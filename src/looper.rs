//! Bar-snapped looping of what was just heard: "loop the last 4 bars".
//!
//! The separated stems (target and band, so the mix levels still apply to a loop) go
//! into a ring buffer of the last few minutes as they are played. Asking for a loop
//! runs Beat This! once over the last minute (src/beats.rs: mel, network, steady
//! decoder), lays the bar lines, and picks the N bars that end at the bar line nearest
//! to the moment of the request. Everything is in the past, so nothing is predicted.
//!
//! The logits are kept: a different meter, half or double time, or another bar 1 is
//! a re-decode (milliseconds), not another network run. The loop's ends are beat
//! indices, so moving them always lands on a beat.
//!
//! Frame numbers count stem frames since the first push (44.1 kHz), the player's own
//! timeline for what it has played.

#![allow(dead_code)]

use crate::beats::{
    self, BeatModel, Decimator, Mel, Settings, DECIM_DELAY, FPS, FRAMES, HOP, MELS,
};
use crate::model::Separator;
use crate::stream::Streamer;
use crate::stretch::Stretch;
use anyhow::{ensure, Result};
use std::sync::Mutex;

const RATE: usize = 44_100;
/// Minutes of stems kept for looping.
pub const KEEP_S: usize = 150;
/// Seconds analysed per request (two network runs).
const ANALYSE_S: usize = 60;
/// Crossfade at the loop point: the end fades into the audio just before the start,
/// so the wrap is as continuous as the music.
const XFADE: usize = RATE / 100;
/// Exact silence at least this long in the buffer is a paused source, skipped by the
/// tail after a loop (music never has a quarter second of exact zeros).
const GAP_SKIP: i64 = RATE as i64 / 4;
/// Beat This! discards this many frames at either edge of a chunk.
const BORDER: usize = 6;

/// Values per frame: target L R, band L R, second stem L R (a two-stem model's
/// second stem, e.g. vocals with bass; silent otherwise). Target + band + second is
/// the mix.
const CH: usize = 6;
/// One frame of the three stems.
pub type Frame = [f32; CH];

/// Stems, interleaved, as i16 (half the memory of f32).
struct Ring {
    data: Vec<i16>,
    cap: usize,
    total: u64,
}

fn to_i16(x: f32) -> i16 {
    (x.clamp(-1.0, 1.0) * 32767.0) as i16
}

fn frame_of(d: &[i16]) -> Frame {
    let s = 1.0 / 32767.0;
    std::array::from_fn(|c| d[c] as f32 * s)
}

/// Interleaved stereo stems (`second` may be empty) into frames of CH values.
fn put(dst: &mut [i16], bass: &[f32], band: &[f32], second: &[f32], i: usize) {
    let z = [0f32; 2];
    let sec = second.get(2 * i..2 * i + 2).unwrap_or(&z);
    let v = [
        bass[2 * i],
        bass[2 * i + 1],
        band[2 * i],
        band[2 * i + 1],
        sec[0],
        sec[1],
    ];
    for c in 0..CH {
        dst[c] = to_i16(v[c]);
    }
}

impl Ring {
    fn new(frames: usize) -> Self {
        Ring {
            data: vec![0; CH * frames],
            cap: frames,
            total: 0,
        }
    }

    fn push(&mut self, bass: &[f32], band: &[f32], second: &[f32]) {
        let n = bass.len() / 2;
        for i in 0..n {
            let k = CH * ((self.total as usize + i) % self.cap);
            put(&mut self.data[k..k + CH], bass, band, second, i);
        }
        self.total += n as u64;
    }

    fn oldest(&self) -> u64 {
        self.total.saturating_sub(self.cap as u64)
    }

    /// Frame `f`; silence outside what is kept.
    fn at(&self, f: i64) -> Frame {
        if f < self.oldest() as i64 || f >= self.total as i64 {
            return [0.0; CH];
        }
        let k = CH * (f as usize % self.cap);
        frame_of(&self.data[k..k + CH])
    }
}

/// A copy of the analysed minute: the loop plays from it, so the ring can keep
/// filling (or fill with the silence of a paused source) for as long as the loop plays.
struct Clip {
    from: u64,
    data: Vec<i16>,
}

impl Clip {
    fn frames(&self) -> usize {
        self.data.len() / CH
    }

    fn at(&self, f: i64) -> Frame {
        let i = f - self.from as i64;
        if i < 0 || i as usize >= self.frames() {
            return [0.0; CH];
        }
        let k = CH * i as usize;
        frame_of(&self.data[k..k + CH])
    }
}

/// Frame `v` into frame `i` of the three interleaved stereo outputs (`second` may be
/// empty when the caller doesn't take it).
fn write(v: &Frame, i: usize, bass: &mut [f32], band: &mut [f32], second: &mut [f32]) {
    bass[2 * i] = v[0];
    bass[2 * i + 1] = v[1];
    band[2 * i] = v[2];
    band[2 * i + 1] = v[3];
    if second.len() >= 2 * i + 2 {
        second[2 * i] = v[4];
        second[2 * i + 1] = v[5];
    }
}

/// The mix (all stems) of a frame, stereo.
fn mixed(v: &Frame) -> [f32; 2] {
    [v[0] + v[2] + v[4], v[1] + v[3] + v[5]]
}

/// The network's view of the analysed stretch, kept for re-decoding.
struct Analysis {
    /// stem frame of 50 fps frame 0
    origin: f64,
    beat: Vec<f32>,
    down: Vec<f32>,
}

impl Analysis {
    fn frame_to_stem(&self, t_s: f64) -> f64 {
        self.origin + t_s * RATE as f64
    }
}

/// The decoded beats over the analysed stretch.
struct Grid {
    beats: Vec<f64>,
    bar: Vec<bool>,
    meter: usize,
    bpm: f32,
}

struct State {
    analysis: Option<Analysis>,
    clip: Option<Clip>,
    settings: Settings,
    grid: Option<Grid>,
    /// the loop is beats[a]..beats[b]
    a: usize,
    b: usize,
    playing: bool,
    /// frames into the loop
    pos: usize,
    click: bool,
    /// set while a request is being analysed
    finding: bool,
    /// After a loop: playing on from the buffer at this frame (see `go_live`).
    tail: Option<i64>,
    /// how far the tail has looked for the end of a silent gap
    scanned: i64,
    /// Practice speed (1 = as recorded), transposition in semitones, and whether a
    /// slower speed keeps the pitch (else it drops with the speed, like tape).
    speed: f32,
    semis: f32,
    keep_pitch: bool,
    /// The stretcher while speed or pitch is changed, and its output not yet played:
    /// CH + 1 values per frame (the stems, then the click).
    stretch: Option<Stretch>,
    fifo: Vec<f32>,
    frac: f64,
}

impl State {
    /// The pitch multiplier for the stretcher.
    fn transpose(&self) -> f32 {
        2f32.powf(self.semis / 12.0) * if self.keep_pitch { 1.0 } else { self.speed }
    }

    fn plain(&self) -> bool {
        (self.speed - 1.0).abs() < 1e-3 && (self.transpose() - 1.0).abs() < 1e-4
    }

    /// The loop's position changed: the stretcher starts again from there.
    fn resync(&mut self) {
        self.stretch = None;
        self.fifo.clear();
        self.frac = 0.0;
    }
}

/// Frame `p` of a loop `len` frames long starting at `start`: near the end it fades
/// into the audio leading up to the start, so the wrap is as continuous as the music.
fn loop_frame(clip: &Clip, start: i64, len: usize, p: usize) -> Frame {
    let mut v = clip.at(start + p as i64);
    if p >= len - XFADE {
        let w = (p - (len - XFADE)) as f32 / XFADE as f32;
        let u = clip.at(start - (len - p) as i64);
        for c in 0..CH {
            v[c] = (1.0 - w) * v[c] + w * u[c];
        }
    }
    v
}

/// Output frames per stretcher call.
const STRETCH_BLOCK: usize = 512;

pub struct Looper {
    ring: Mutex<Ring>,
    net: Mutex<Box<dyn BeatModel>>,
    mel: Mel,
    state: Mutex<State>,
    clicks: [Vec<f32>; 2],
}

/// What the loop view draws (frames are stem frames).
pub struct View {
    pub playing: bool,
    pub finding: bool,
    pub bpm: f32,
    pub meter: usize,
    pub bars: usize,
    pub from: f64,
    pub to: f64,
    pub start: f64,
    pub end: f64,
    pub playhead: f64,
    pub a: usize,
    pub b: usize,
    /// (frame, is a bar line, beat index) for the beats in view
    pub beats: Vec<(f64, bool, usize)>,
    /// peak level per bin across the view, 0..1
    pub peaks: Vec<f32>,
}

fn click_sound(freq: f64) -> Vec<f32> {
    let n = RATE * 4 / 100;
    (0..n)
        .map(|i| {
            let t = i as f64 / RATE as f64;
            ((2.0 * std::f64::consts::PI * freq * t).sin() * (-t * 90.0).exp()) as f32
        })
        .collect()
}

impl Looper {
    pub fn new(net: Box<dyn BeatModel>) -> Self {
        Looper {
            ring: Mutex::new(Ring::new(KEEP_S * RATE)),
            net: Mutex::new(net),
            mel: Mel::new(),
            state: Mutex::new(State {
                analysis: None,
                clip: None,
                settings: Settings::default(),
                grid: None,
                a: 0,
                b: 0,
                playing: false,
                pos: 0,
                click: false,
                finding: false,
                tail: None,
                scanned: 0,
                speed: 1.0,
                semis: 0.0,
                keep_pitch: true,
                stretch: None,
                fifo: Vec::new(),
                frac: 0.0,
            }),
            clicks: [click_sound(1000.0), click_sound(1700.0)],
        }
    }

    /// Interleaved stereo stems, in play order, every frame (`second` empty for a
    /// one-stem model).
    pub fn push(&self, bass: &[f32], band: &[f32], second: &[f32]) {
        self.ring.lock().unwrap().push(bass, band, second);
    }

    pub fn total(&self) -> u64 {
        self.ring.lock().unwrap().total
    }

    /// Analyse the last minute and loop the `bars` bars ending at the bar line nearest
    /// to stem frame `press`. Blocks for the network runs (about a second each on a
    /// phone CPU); playback carries on meanwhile.
    pub fn find(&self, press: u64, bars: usize, s: Settings) -> Result<()> {
        self.state.lock().unwrap().finding = true;
        let r = self.find_inner(press, bars, s);
        self.state.lock().unwrap().finding = false;
        r
    }

    fn find_inner(&self, press: u64, bars: usize, s: Settings) -> Result<()> {
        // The newest minute: a copy to loop from, and its mono mix to analyse.
        let clip = {
            let ring = self.ring.lock().unwrap();
            let to = ring.total;
            let from = to
                .saturating_sub((ANALYSE_S * RATE) as u64)
                .max(ring.oldest());
            let mut data = Vec::with_capacity(CH * (to - from) as usize);
            for f in from..to {
                let k = CH * (f as usize % ring.cap);
                data.extend_from_slice(&ring.data[k..k + CH]);
            }
            Clip { from, data }
        };
        let from = clip.from;
        let mono: Vec<f32> = (0..clip.frames())
            .map(|i| {
                let m = mixed(&clip.at(from as i64 + i as i64));
                0.5 * (m[0] + m[1])
            })
            .collect();
        ensure!(
            mono.len() >= 6 * RATE,
            "not enough audio yet ({:.0} s)",
            mono.len() as f32 / RATE as f32
        );
        let mut x22 = Vec::with_capacity(mono.len() / 2 + 1);
        Decimator::new().push(&mono, &mut x22);
        let spect = self.mel.spect_any(&x22);
        let (beat, down) = self.logits(&spect)?;
        // 22.05 kHz sample k is stem frame from + 2k - delay
        let origin = from as f64 - DECIM_DELAY as f64;
        let an = Analysis { origin, beat, down };
        let mut st = self.state.lock().unwrap();
        st.settings = s;
        st.analysis = Some(an);
        st.clip = Some(clip);
        self.decode(&mut st);
        let g = st.grid.as_ref().unwrap();
        ensure!(g.beats.len() > 2, "no beat found");
        // The bar line nearest to the press ends the loop.
        let b = (0..g.beats.len())
            .filter(|&i| g.bar[i])
            .min_by(|&i, &j| {
                (g.beats[i] - press as f64)
                    .abs()
                    .total_cmp(&(g.beats[j] - press as f64).abs())
            })
            .unwrap_or(g.beats.len() - 1);
        let a = b.saturating_sub(bars.max(1) * g.meter);
        ensure!(b > a, "no bars found");
        st.a = a;
        st.b = b;
        st.pos = 0;
        st.resync();
        st.playing = true;
        st.tail = None;
        st.resync();
        Ok(())
    }

    /// Network over a spectrogram of any length: 1500-frame chunks overlapping by the
    /// discarded borders, the last one aligned to the end (Beat This!'s own chunking);
    /// a short one is zero-padded in front (zero = silence in log-mel).
    fn logits(&self, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = spect.len() / MELS;
        let mut beat = vec![0f32; n];
        let mut down = vec![0f32; n];
        let mut net = self.net.lock().unwrap();
        if n <= FRAMES {
            let mut x = vec![0f32; (FRAMES - n) * MELS];
            x.extend_from_slice(spect);
            let (b, d) = net.run(&x)?;
            beat.copy_from_slice(&b[FRAMES - n..]);
            down.copy_from_slice(&d[FRAMES - n..]);
            return Ok((beat, down));
        }
        let step = FRAMES - 2 * BORDER;
        let mut starts: Vec<usize> = (0..)
            .map(|k| k * step)
            .take_while(|&s| s + FRAMES < n)
            .collect();
        starts.push(n - FRAMES);
        let mut done = 0;
        for (k, &s0) in starts.iter().enumerate() {
            let (b, d) = net.run(&spect[s0 * MELS..(s0 + FRAMES) * MELS])?;
            let lo = if k == 0 { 0 } else { (s0 + BORDER).max(done) };
            let hi = if k + 1 == starts.len() {
                n
            } else {
                s0 + FRAMES - BORDER
            };
            beat[lo..hi].copy_from_slice(&b[lo - s0..hi - s0]);
            down[lo..hi].copy_from_slice(&d[lo - s0..hi - s0]);
            done = hi;
        }
        Ok((beat, down))
    }

    /// Decode the kept logits with the current settings into `st.grid`.
    fn decode(&self, st: &mut State) {
        let Some(an) = &st.analysis else { return };
        let dec = beats::decode(&an.beat, &an.down, &st.settings);
        let beats: Vec<f64> = dec.beats.iter().map(|&t| an.frame_to_stem(t)).collect();
        let bar = (0..beats.len())
            .map(|i| i >= dec.first_bar && (i - dec.first_bar).is_multiple_of(dec.meter))
            .collect();
        st.grid = Some(Grid {
            beats,
            bar,
            meter: dec.meter,
            bpm: dec.bpm,
        });
    }

    /// Another meter or feel: re-decode, keeping the loop's end time and bar count.
    pub fn redecode(&self, meter: usize, feel: f32) {
        let mut st = self.state.lock().unwrap();
        let Some(g) = &st.grid else { return };
        let (end, bars) = (
            g.beats[st.b],
            ((st.b - st.a) as f32 / g.meter as f32).round().max(1.0) as usize,
        );
        st.settings.meter = meter;
        st.settings.feel = feel;
        self.decode(&mut st);
        let g = st.grid.as_ref().unwrap();
        let m = g.meter;
        let b = (0..g.beats.len())
            .filter(|&i| g.bar[i])
            .min_by(|&i, &j| {
                (g.beats[i] - end)
                    .abs()
                    .total_cmp(&(g.beats[j] - end).abs())
            })
            .unwrap_or(g.beats.len() - 1);
        st.b = b;
        st.a = b.saturating_sub(bars * m);
        st.pos = 0;
        st.resync();
    }

    /// Make beat `i` a bar line (the tracker picked the wrong bar 1).
    pub fn bar_one(&self, i: usize) {
        let mut st = self.state.lock().unwrap();
        let Some(g) = st.grid.as_mut() else { return };
        let m = g.meter;
        for (k, f) in g.bar.iter_mut().enumerate() {
            *f = (k + m - i % m).is_multiple_of(m);
        }
    }

    /// Move both ends by `k` beats.
    pub fn shift(&self, k: i32) {
        let mut st = self.state.lock().unwrap();
        let Some(n) = st.grid.as_ref().map(|g| g.beats.len()) else {
            return;
        };
        let (a, b) = (st.a as i64 + k as i64, st.b as i64 + k as i64);
        if a >= 0 && (b as usize) < n {
            st.a = a as usize;
            st.b = b as usize;
            st.resync();
        }
    }

    /// Set the ends to beats `a` and `b` (from dragging the handles).
    pub fn set_edges(&self, a: usize, b: usize) {
        let mut st = self.state.lock().unwrap();
        let Some(n) = st.grid.as_ref().map(|g| g.beats.len()) else {
            return;
        };
        if a < b && b < n {
            st.a = a;
            st.b = b;
            if st.pos >= self.loop_len(&st) {
                st.pos = 0;
                st.resync();
            }
        }
    }

    /// `n` bars ending where the loop ends now.
    pub fn set_bars(&self, n: usize) {
        let mut st = self.state.lock().unwrap();
        let Some(m) = st.grid.as_ref().map(|g| g.meter) else {
            return;
        };
        st.a = st.b.saturating_sub(n.max(1) * m);
        st.resync();
    }

    pub fn set_click(&self, on: bool) {
        self.state.lock().unwrap().click = on;
    }

    /// Back to live: the music carries on from where the loop is, through what was
    /// buffered after it, and into the source once it plays again (the paused stretch
    /// is skipped), so going live never waits out the separator's delay.
    pub fn go_live(&self) {
        let mut st = self.state.lock().unwrap();
        if !st.playing {
            return;
        }
        st.playing = false;
        let here = st
            .grid
            .as_ref()
            .map(|g| g.beats[st.a].round() as i64 + st.pos as i64);
        st.tail = here;
        st.scanned = here.unwrap_or(0);
    }

    /// Straight to the live stream, dropping the buffered tail.
    pub fn stop(&self) {
        let mut st = self.state.lock().unwrap();
        st.playing = false;
        st.tail = None;
    }

    /// Start the loop again from the top (or resume after `stop`).
    pub fn play(&self) {
        let mut st = self.state.lock().unwrap();
        if st.grid.is_some() {
            st.playing = true;
            st.pos = 0;
            st.resync();
        }
    }

    pub fn playing(&self) -> bool {
        self.state.lock().unwrap().playing
    }

    fn loop_len(&self, st: &State) -> usize {
        st.grid
            .as_ref()
            .map_or(0, |g| (g.beats[st.b] - g.beats[st.a]).round() as usize)
    }

    /// The next `n` frames of the loop (or of the tail after it) into `bass`, `band`
    /// (interleaved stereo) and `click` (mono), then move on by `advance` frames (the
    /// rest is the limiter's lookahead). Returns the buffer frame the block starts at,
    /// or None when the live stream should play.
    pub fn read(
        &self,
        bass: &mut [f32],
        band: &mut [f32],
        second: &mut [f32],
        click: &mut [f32],
        n: usize,
        advance: usize,
    ) -> Option<i64> {
        let mut st = self.state.lock().unwrap();
        if !st.playing && st.tail.is_some() {
            return Some(self.read_tail(&mut st, bass, band, second, click, n, advance));
        }
        if !st.playing || st.grid.is_none() {
            return None;
        }
        let len = self.loop_len(&st);
        if len < 2 * XFADE {
            return None;
        }
        if !st.plain() {
            return self.read_stretched(&mut st, len, bass, band, second, click, n, advance);
        }
        if st.stretch.is_some() {
            st.resync();
        }
        let g = st.grid.as_ref().unwrap();
        let start = g.beats[st.a].round() as i64;
        let ring = st.clip.as_ref()?;
        let first = start + st.pos as i64;
        click[..n].fill(0.0);
        for i in 0..n {
            let p = (st.pos + i) % len;
            let v = loop_frame(ring, start, len, p);
            write(&v, i, bass, band, second);
        }
        if st.click {
            for k in st.a..st.b {
                let at = (g.beats[k] - g.beats[st.a]).round() as usize;
                let snd = &self.clicks[g.bar[k] as usize];
                for (i, c) in click[..n].iter_mut().enumerate() {
                    let p = (st.pos + i) % len;
                    if p >= at && p - at < snd.len() {
                        *c += snd[p - at];
                    }
                }
            }
        }
        st.pos = (st.pos + advance) % len;
        Some(first)
    }

    /// The loop at another speed or pitch: source frames from the loop position go
    /// through the stretcher in blocks into a FIFO, from which `n` frames are read and
    /// `advance` used up. The click is laid after stretching, at the beats' stretched
    /// times, so it stays a click.
    #[allow(clippy::too_many_arguments)]
    fn read_stretched(
        &self,
        st: &mut State,
        len: usize,
        bass: &mut [f32],
        band: &mut [f32],
        second: &mut [f32],
        click: &mut [f32],
        n: usize,
        advance: usize,
    ) -> Option<i64> {
        let speed = st.speed as f64;
        let factor = st.transpose();
        let g = st.grid.as_ref()?;
        let start = g.beats[st.a].round() as i64;
        let beats: Vec<(usize, bool)> = (st.a..st.b)
            .map(|k| ((g.beats[k] - g.beats[st.a]).round() as usize, g.bar[k]))
            .collect();
        let with_click = st.click;
        let State {
            clip,
            stretch,
            fifo,
            frac,
            pos,
            ..
        } = st;
        let clip = clip.as_ref()?;
        if stretch.is_none() {
            let mut s = Stretch::new(CH, RATE as f32, false);
            s.set_transpose(factor);
            // pre-roll: the audio just before the position, so it starts at full level
            let pre = s.input_latency();
            let mut x = Vec::with_capacity(CH * pre);
            for k in 0..pre {
                let p = (*pos + len * (pre / len + 1) - pre + k) % len;
                x.extend(loop_frame(clip, start, len, p));
            }
            s.seek(&x, speed);
            *stretch = Some(s);
            fifo.clear();
            *frac = 0.0;
        }
        let s = stretch.as_mut().unwrap();
        s.set_transpose(factor);
        let lag = s.input_latency() as f64 + s.output_latency() as f64 * speed;
        let mut input = Vec::new();
        let mut out = vec![0f32; CH * STRETCH_BLOCK];
        while fifo.len() / (CH + 1) < n {
            let want = STRETCH_BLOCK as f64 * speed + *frac;
            let k = want.floor() as usize;
            *frac = want - k as f64;
            let src0 = *pos as f64;
            input.clear();
            for j in 0..k {
                input.extend(loop_frame(clip, start, len, (*pos + j) % len));
            }
            *pos = (*pos + k) % len;
            s.process(&input, &mut out);
            for j in 0..STRETCH_BLOCK {
                // where in the loop this output frame comes from
                let src = (src0 - lag + j as f64 * speed).rem_euclid(len as f64);
                let mut c = 0f32;
                if with_click {
                    for &(at, bar) in &beats {
                        let d = (src - at as f64).rem_euclid(len as f64) / speed;
                        let snd = &self.clicks[bar as usize];
                        if (d as usize) < snd.len() {
                            c += snd[d as usize];
                        }
                    }
                }
                fifo.extend_from_slice(&out[CH * j..CH * j + CH]);
                fifo.push(c);
            }
        }
        const W: usize = CH + 1;
        for i in 0..n {
            let f = &fifo[W * i..W * i + W];
            let v: Frame = std::array::from_fn(|c| f[c]);
            write(&v, i, bass, band, second);
            click[i] = f[CH];
        }
        fifo.drain(..W * advance.min(fifo.len() / W));
        Some(start + *pos as i64)
    }

    /// Practice speed (0.25..2), transposition (semitones, ±12) and whether slowing
    /// down keeps the pitch. Applies to loops.
    pub fn set_speed(&self, speed: f32, semis: f32, keep_pitch: bool) {
        let mut st = self.state.lock().unwrap();
        st.speed = speed.clamp(0.25, 2.0);
        st.semis = semis.clamp(-12.0, 12.0);
        st.keep_pitch = keep_pitch;
    }

    /// The tail: the buffer from the cursor on, skipping stretches of exact silence (a
    /// paused source: the separator sends zeros) once there is sound after them, and
    /// waiting (silence) where the buffer hasn't caught up.
    #[allow(clippy::too_many_arguments)]
    fn read_tail(
        &self,
        st: &mut State,
        bass: &mut [f32],
        band: &mut [f32],
        second: &mut [f32],
        click: &mut [f32],
        n: usize,
        advance: usize,
    ) -> i64 {
        let ring = self.ring.lock().unwrap();
        let total = ring.total as i64;
        let mut c = st.tail.unwrap().max(ring.oldest() as i64);
        let first = c;
        click[..n].fill(0.0);
        for i in 0..n {
            if i == advance {
                st.tail = Some(c);
            }
            let mut v = if c < total { ring.at(c) } else { [0.0; CH] };
            if c < total && v == [0.0; CH] {
                // a silent stretch: where does it end?
                let mut j = st.scanned.max(c);
                while j < total && ring.at(j) == [0.0; CH] {
                    j += 1;
                }
                st.scanned = j;
                if j < total && j - c >= GAP_SKIP {
                    c = j; // sound again after a pause: jump over the pause
                    v = ring.at(c);
                } else if j >= total && j - c >= GAP_SKIP {
                    write(&[0.0; CH], i, bass, band, second); // still paused: wait here
                    continue;
                }
            }
            write(&v, i, bass, band, second);
            if c < total {
                c += 1;
            }
        }
        if advance >= n {
            st.tail = Some(c);
        }
        first
    }

    /// The loop with a bar of context either side, for drawing.
    pub fn view(&self, bins: usize) -> Option<View> {
        let st = self.state.lock().unwrap();
        let g = st.grid.as_ref()?;
        let (start, end) = (g.beats[st.a], g.beats[st.b]);
        // A bar either side (at least 15% of the loop), whether or not there is audio
        // there: the ends never sit at the edge of the view, where they're hard to grab.
        let bar = (end - start) / ((st.b - st.a) as f64 / g.meter as f64).max(1.0);
        let ctx = bar.max(0.15 * (end - start));
        let (from, to) = (start - ctx, end + ctx);
        let beats = g
            .beats
            .iter()
            .enumerate()
            .filter(|(_, &b)| b >= from && b <= to)
            .map(|(i, &b)| (b, g.bar[i], i))
            .collect();
        let ring = st.clip.as_ref()?;
        let span = (to - from).max(1.0);
        let peaks = (0..bins)
            .map(|k| {
                let f0 = (from + span * k as f64 / bins as f64) as i64;
                let f1 = (from + span * (k + 1) as f64 / bins as f64) as i64;
                // every 8th frame is plenty for a peak meter
                (f0..f1.max(f0 + 1))
                    .step_by(8)
                    .map(|f| {
                        let v = ring.at(f);
                        let m = mixed(&v);
                        m[0].abs().max(m[1].abs())
                    })
                    .fold(0f32, f32::max)
                    .min(1.0)
            })
            .collect();
        Some(View {
            playing: st.playing,
            finding: st.finding,
            bpm: g.bpm,
            meter: g.meter,
            bars: ((st.b - st.a) as f32 / g.meter as f32).round() as usize,
            from,
            to,
            start,
            end,
            playhead: start + st.pos as f64,
            a: st.a,
            b: st.b,
            beats,
            peaks,
        })
    }

    /// The loop in buffer frames, for a bookmark: (start, end, first bar line in it,
    /// beats per bar, bars, feel), or None without a loop.
    pub fn times(&self) -> Option<(f64, f64, f64, usize, usize, f32)> {
        let st = self.state.lock().unwrap();
        let g = st.grid.as_ref()?;
        let bar1 = (st.a..st.b)
            .find(|&i| g.bar[i])
            .map_or(g.beats[st.a], |i| g.beats[i]);
        let bars = ((st.b - st.a) as f32 / g.meter as f32).round().max(1.0) as usize;
        Some((
            g.beats[st.a],
            g.beats[st.b],
            bar1,
            g.meter,
            bars,
            st.settings.feel,
        ))
    }

    /// Put the loop's ends on the beats nearest frames `start` and `end`, and bar 1 on
    /// the beat nearest `bar1` (a recalled bookmark, after `find` laid the beats).
    pub fn set_times(&self, start: f64, end: f64, bar1: f64) {
        let mut st = self.state.lock().unwrap();
        let Some(g) = st.grid.as_mut() else { return };
        let near = |t: f64| {
            (0..g.beats.len())
                .min_by(|&i, &j| (g.beats[i] - t).abs().total_cmp(&(g.beats[j] - t).abs()))
                .unwrap_or(0)
        };
        let (a, b, one) = (near(start), near(end), near(bar1));
        let m = g.meter;
        for (k, f) in g.bar.iter_mut().enumerate() {
            *f = (k + m - one % m) % m == 0;
        }
        if a < b {
            st.a = a;
            st.b = b;
            st.pos = 0;
            st.resync();
        }
    }

    /// The stretch to re-separate after a model switch: the loop and a second either
    /// side (for the wrap's crossfade and small edits), or None without a loop.
    pub fn restem_range(&self) -> Option<(i64, i64)> {
        let st = self.state.lock().unwrap();
        let g = st.grid.as_ref()?;
        st.clip.as_ref()?;
        let pad = RATE as f64;
        Some((
            (g.beats[st.a] - pad).round() as i64,
            (g.beats[st.b] + pad).round() as i64,
        ))
    }

    /// The original mix (target + band) of the loop's copy, `n` frames from `from`,
    /// interleaved stereo (silence outside the copy).
    pub fn clip_mix(&self, from: i64, n: usize) -> Vec<f32> {
        let st = self.state.lock().unwrap();
        let mut out = vec![0f32; 2 * n];
        if let Some(c) = st.clip.as_ref() {
            for i in 0..n {
                let v = c.at(from + i as i64);
                let m = mixed(&v);
                out[2 * i] = m[0];
                out[2 * i + 1] = m[1];
            }
        }
        out
    }

    /// Put re-separated stems into the loop's copy from frame `from` (what is outside
    /// the copy is dropped; `second` empty for a one-stem model).
    pub fn replace(&self, from: i64, bass: &[f32], band: &[f32], second: &[f32]) {
        let mut st = self.state.lock().unwrap();
        let Some(c) = st.clip.as_mut() else { return };
        for i in 0..bass.len() / 2 {
            let k = from + i as i64 - c.from as i64;
            if k < 0 || k as usize >= c.frames() {
                continue;
            }
            let k = CH * k as usize;
            put(&mut c.data[k..k + CH], bass, band, second, i);
        }
    }

    pub fn finding(&self) -> bool {
        self.state.lock().unwrap().finding
    }
}

/// Re-separate the loop (and a second either side) with `eng`, a newly loaded model:
/// a window of context before it, then hop by hop, as live. The streamer is reset
/// before and after, so the live stream starts clean. False without a loop.
pub fn restem<S: Separator>(l: &Looper, eng: &mut Streamer<S>) -> Result<bool> {
    let Some((from, to)) = l.restem_range() else {
        return Ok(false);
    };
    let (hop, delay, warm) = (eng.hop, eng.delay(), eng.window());
    let s0 = from - warm as i64;
    // enough input that the output (delayed by `delay`) reaches `to`
    let need = (to - s0) as usize + delay;
    let n = need.div_ceil(hop) * hop;
    let mix = l.clip_mix(s0, n);
    eng.reset();
    let (mut bass, mut band, mut second) = (
        Vec::with_capacity(2 * n),
        Vec::with_capacity(2 * n),
        Vec::new(),
    );
    for chunk in mix.chunks_exact(2 * hop) {
        let st = eng.process(chunk)?;
        bass.extend(st.bass);
        band.extend(st.band);
        if let Some(e) = st.extra.into_iter().next() {
            second.extend(e);
        }
    }
    eng.reset();
    // output frame k is input frame s0 + k - delay
    let k0 = (from - s0) as usize + delay;
    let k1 = (to - s0) as usize + delay;
    let sec = if second.len() >= 2 * k1 {
        &second[2 * k0..2 * k1]
    } else {
        &[][..]
    };
    l.replace(from, &bass[2 * k0..2 * k1], &band[2 * k0..2 * k1], sec);
    Ok(true)
}

#[allow(unused)]
const _: usize = FPS + HOP;

#[cfg(test)]
mod tests {
    use super::*;

    struct NoNet;
    impl BeatModel for NoNet {
        fn run(&mut self, _: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
            unreachable!()
        }
    }

    /// Live after a loop: plays on from the loop, skips the pause, then the new audio.
    #[test]
    fn tail_skips_the_pause() {
        let l = Looper::new(Box::new(NoNet));
        // 1 s of music (value 0.5), 1 s of exact silence (paused), then 0.5 s of music (0.25)
        let mk = |n: usize, v: f32| vec![v; 2 * n];
        l.push(&mk(RATE, 0.5), &mk(RATE, 0.0), &[]);
        l.push(&mk(RATE, 0.0), &mk(RATE, 0.0), &[]);
        {
            let mut st = l.state.lock().unwrap();
            st.grid = Some(Grid {
                beats: vec![0.0, 1000.0, 2000.0],
                bar: vec![true; 3],
                meter: 1,
                bpm: 120.0,
            });
            st.a = 0;
            st.b = 2;
            st.playing = true;
            st.pos = 500;
        }
        l.go_live();
        let n = 4096;
        let (mut b, mut r, mut c) = (vec![0f32; 2 * n], vec![0f32; 2 * n], vec![0f32; n]);
        // up to the pause: the music after the loop
        assert_eq!(l.read(&mut b, &mut r, &mut [], &mut c, n, n), Some(500));
        assert!(b.iter().all(|&v| (v - 0.5).abs() < 1e-3));
        let mut played = n + 500;
        while played + n < RATE {
            l.read(&mut b, &mut r, &mut [], &mut c, n, n);
            played += n;
        }
        // reaching the pause with nothing after it yet: silence, waiting
        for _ in 0..3 {
            l.read(&mut b, &mut r, &mut [], &mut c, n, n);
        }
        assert!(b.iter().all(|&v| v == 0.0));
        // the source plays again: straight into it, the pause skipped
        l.push(&mk(RATE / 2, 0.25), &mk(RATE / 2, 0.0), &[]);
        let at = l.read(&mut b, &mut r, &mut [], &mut c, n, n).unwrap();
        assert!(at >= RATE as i64, "still before the pause: {at}");
        assert!(
            b.iter().all(|&v| (v - 0.25).abs() < 1e-3),
            "first {:?}",
            &b[..4]
        );
    }

    #[test]
    fn ring_wraps() {
        let mut r = Ring::new(4);
        let x: Vec<f32> = (0..12).map(|i| i as f32 / 100.0).collect();
        r.push(&x, &x, &[]);
        assert_eq!(r.total, 6);
        assert_eq!(r.oldest(), 2);
        assert_eq!(r.at(1), [0.0; CH]);
        assert!((r.at(5)[0] - 0.10).abs() < 1e-3);
    }
}
