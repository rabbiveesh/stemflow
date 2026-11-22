//! Live mode: capture the target app, separate, and play the stems back.
//!
//! The platform part (`pulse` on Linux, `coreaudio` on macOS) only moves audio:
//! it takes the app's sound away from the speakers, hands it to us as 44.1 kHz
//! stereo, and plays what we give back. Everything else is shared.

#[cfg(target_os = "macos")]
mod coreaudio;
#[cfg(target_os = "linux")]
mod pulse;
#[cfg(target_os = "macos")]
use coreaudio as backend;
#[cfg(target_os = "linux")]
use pulse as backend;

use crate::control::Control;
use crate::model::{Separator, SAMPLE_RATE};
use crate::stream::{Stems, Streamer};
use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

/// Blocking source of interleaved 44.1 kHz stereo, about 20 ms per call.
pub trait Capture: Send {
    fn read(&mut self) -> Result<Vec<f32>>;
}

/// Blocking sink of interleaved 44.1 kHz stereo: returns once `x` is queued, which
/// waits while the device buffer (`play_frames` of [`backend::open`]) is full.
pub trait Playback: Send {
    fn write(&mut self, x: &[f32]) -> Result<()>;
}

/// An open route: the app's audio diverted to `capture`, `playback` to the speakers.
pub struct Io {
    pub capture: Box<dyn Capture>,
    pub playback: Box<dyn Playback>,
    /// Called every 2 s: pick up streams or processes the app opened since.
    pub regrab: Box<dyn FnMut() + Send>,
    /// Called from the Ctrl-C handler, just before exiting: give the app its sound back.
    pub cleanup: Box<dyn Fn() + Send + Sync>,
    /// What was grabbed, for the start-up line.
    pub route: String,
    /// The real output it plays to.
    pub output: String,
    /// Printed under the start-up line, e.g. when nothing was grabbed yet.
    pub hint: Option<String>,
}

/// Open the platform route for `grab` (an app name, case-insensitive substring).
pub fn open(grab: &str, output: Option<String>, name: &str) -> Result<Io> {
    // Keep the device buffer small and the jitter buffer in the player instead, so
    // the mix levels (applied as samples leave for the device) change within ~0.1 s.
    backend::open(grab, output, name, 5 * BLOCK)
}

/// The player's block: 20 ms.
const BLOCK: usize = SAMPLE_RATE as usize / 50;

/// Exit on Ctrl-C after `cleanup`.
fn on_ctrlc(running: Arc<AtomicBool>, cleanup: Box<dyn Fn() + Send + Sync>) -> Result<()> {
    ctrlc::set_handler(move || {
        running.store(false, Ordering::SeqCst);
        cleanup();
        // Skip atexit/static destructors: tearing down CUDA under live audio threads segfaults.
        unsafe { libc::_exit(0) };
    })?;
    Ok(())
}

/// No model: play the captured app straight back, printing its level every second.
/// Checks the route (capture, muting the original, playback) on its own.
pub fn passthrough(grab: &str, output: Option<String>, seconds: Option<f32>) -> Result<()> {
    let Io {
        mut capture,
        mut playback,
        mut regrab,
        cleanup,
        route,
        output,
        hint,
    } = open(grab, output, "passthrough")?;
    on_ctrlc(Arc::new(AtomicBool::new(true)), cleanup)?;
    eprintln!("stemflow: {route} -> passthrough -> {output} (Ctrl-C to stop)");
    if let Some(h) = hint {
        eprintln!("  {h}");
    }
    let (mut n, mut peak, mut total) = (0usize, 0f32, 0usize);
    let mut last_grab = Instant::now();
    loop {
        let v = capture.read()?;
        peak = v.iter().fold(peak, |m, x| m.max(x.abs()));
        n += v.len() / 2;
        playback.write(&v)?;
        if n >= SAMPLE_RATE as usize {
            total += n;
            let db = if peak > 0.0 {
                format!("{:6.1} dBFS", 20.0 * peak.log10())
            } else {
                "silence".into()
            };
            eprintln!("{:5.1}s  peak {db}", total as f32 / SAMPLE_RATE as f32);
            (n, peak) = (0, 0.0);
            if seconds.is_some_and(|s| total as f32 >= s * SAMPLE_RATE as f32) {
                return Ok(());
            }
        }
        if last_grab.elapsed().as_secs() >= 2 {
            last_grab = Instant::now();
            regrab();
        }
    }
}

pub type Live = Streamer<Box<dyn Separator + Send>>;

/// `load` builds the streamer for a model id when the menu switches models.
#[allow(clippy::too_many_arguments)]
pub fn live(
    st: Live,
    ctl: Arc<Control>,
    beats: Option<Arc<crate::beats::Tracker>>,
    deck: Option<Arc<crate::loopdeck::Deck>>,
    grab: &str,
    output: Option<String>,
    slack: f32,
    load: impl Fn(&str) -> Result<(Live, &'static str, Option<&'static str>)>,
) -> Result<()> {
    let mut st = Some(st);
    let Io {
        mut capture,
        mut playback,
        mut regrab,
        cleanup,
        route,
        output,
        hint,
    } = open(grab, output, ctl.target())?;
    let running = Arc::new(AtomicBool::new(true));
    on_ctrlc(running.clone(), cleanup)?;
    let hop = st.as_ref().unwrap().hop;
    let block = BLOCK;
    eprintln!(
        "stemflow: {route} -> {} -> {output}; latency ≈ {:.1}s (Ctrl-C to stop)",
        ctl.target(),
        (st.as_ref().unwrap().latency() as f32 / SAMPLE_RATE as f32) + slack
    );
    if let Some(h) = hint {
        eprintln!("  {h}");
    }

    // capture -> worker -> playback, each on its own thread.
    let (tx_in, rx_in) = mpsc::channel::<Vec<f32>>();
    // stems, and whether they are the plain mix (stems only when looping)
    let (tx_out, rx_out) = mpsc::channel::<(Stems, bool)>();
    let r = running.clone();
    std::thread::spawn(move || {
        while r.load(Ordering::Relaxed) {
            let Ok(v) = capture.read() else { break };
            if tx_in.send(v).is_err() {
                break;
            }
        }
    });
    // The output trails the input by the separator's delay; the player maps what it
    // plays onto the tracker's timeline (input frames) with it.
    let delay = Arc::new(std::sync::atomic::AtomicUsize::new(
        st.as_ref().unwrap().delay(),
    ));
    if let Some(tr) = beats.clone() {
        let (c, r) = (ctl.clone(), running.clone());
        std::thread::spawn(move || beat_loop(&tr, &c, &r));
    }
    let (pbeats, pdelay) = (beats.clone(), delay.clone());
    let pctl = ctl.clone();
    let pdeck = deck.clone();
    let refill = hop + (slack * SAMPLE_RATE as f32) as usize;
    let player = std::thread::spawn(move || -> Result<()> {
        use mpsc::{RecvTimeoutError, TryRecvError};
        // bass, band and a second stem (all zero for one-stem models)
        let (mut bass, mut band, mut sec) =
            (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new());
        let mut pos = 0; // frames already played from the front of the buffers
        let mut buffering = true;
        let mut g = pctl.gains3();
        let mut out = vec![0f32; 2 * block];
        let mut lim = Limiter::new();
        let mut played = 0i64; // stem frames played: input frame (played - delay)
                               // a loop block plus the limiter's lookahead, and its click
        let (mut lb, mut lr, mut ls, mut lc) = (
            vec![0f32; 4 * block],
            vec![0f32; 4 * block],
            vec![0f32; 4 * block],
            vec![0f32; 2 * block],
        );
        let mut looping = false; // the last block came from the looper
        if let Some(d) = &pdeck {
            d.set_lag(DEVICE_LAG as i64);
        }
        loop {
            // Take whatever the worker has finished. Wait only when the live stream is
            // what plays and it's starved: a loop never waits for the separator (a
            // model switch can hold it for many seconds). While buffering, always take
            // more: otherwise a first hop of more than two blocks but less than
            // `refill` would spin here forever without playing.
            let avail = bass.len() / 2 - pos;
            if buffering || avail < 2 * block {
                let next = if !looping && (buffering || avail < block) {
                    match rx_out.recv_timeout(std::time::Duration::from_millis(200)) {
                        Ok(st) => Some(st),
                        Err(RecvTimeoutError::Timeout) => None,
                        Err(RecvTimeoutError::Disconnected) => return Ok(()),
                    }
                } else {
                    match rx_out.try_recv() {
                        Ok(st) => Some(st),
                        Err(TryRecvError::Empty) => None,
                        Err(TryRecvError::Disconnected) => return Ok(()),
                    }
                };
                if let Some((st, plain)) = next {
                    // every live block goes into the loop buffer, played or not
                    if let Some(d) = &pdeck {
                        d.looper.push(
                            &st.bass,
                            &st.band,
                            st.extra.first().map_or(&[][..], |v| &v[..]),
                        );
                        if plain {
                            d.looper.mark_plain(st.band.len() / 2);
                        }
                    }
                    bass.drain(..2 * pos);
                    band.drain(..2 * pos);
                    sec.drain(..2 * pos);
                    pos = 0;
                    match st.extra.into_iter().next() {
                        Some(v) => sec.extend(v),
                        None => sec.resize(sec.len() + st.bass.len(), 0.0),
                    }
                    bass.extend(st.bass);
                    band.extend(st.band);
                }
            }
            let avail = bass.len() / 2 - pos;
            let mut target = pctl.gains3();
            let src = pdeck.as_ref().and_then(|d| {
                d.looper
                    .read(&mut lb, &mut lr, &mut ls, &mut lc, 2 * block, block)
            });
            looping = src.is_some();
            // The plain mix (not separated: stems only when looping) plays as it is,
            // from the band's place.
            if let Some(d) = &pdeck {
                if (looping && d.looper.plain_now()) || (!looping && d.stems_only()) {
                    target = (0.0, 1.0, 0.0);
                }
            }
            if let (Some(at), Some(d)) = (src, &pdeck) {
                // A loop (or the music playing on after one): from the buffer, at
                // the same levels. Live blocks are used up at the same pace.
                mix_block([&lb, &lr, &ls], 0, block, g, target, &mut out);
                lim.apply(&mut out, mix_peak([&lb, &lr, &ls], block, block, target));
                for (i, c) in lc[..block].iter().enumerate() {
                    for ch in 0..2 {
                        out[2 * i + ch] = (out[2 * i + ch] + CLICK_LEVEL * c).clamp(-1.0, 1.0);
                    }
                }
                if avail >= block {
                    pos += block;
                    played += block as i64;
                }
                d.set_heard(at - DEVICE_LAG as i64);
            } else {
                if buffering {
                    if avail < refill {
                        continue;
                    }
                    buffering = false;
                }
                if avail < block {
                    eprintln!("warning: playback ran dry; rebuffering");
                    buffering = true;
                    continue;
                }
                mix_block([&bass, &band, &sec], pos, block, g, target, &mut out);
                // Look one block ahead (it's already buffered) so the limiter is down
                // before a peak arrives rather than clipping it.
                let ahead = (avail - block).min(block);
                let next = mix_peak([&bass, &band, &sec], pos + block, ahead, target);
                lim.apply(&mut out, next);
                if let Some(tr) = pbeats.as_ref().filter(|_| pctl.click()) {
                    let d = pdelay.load(Ordering::Relaxed) as i64;
                    tr.click(&mut out, played - d, CLICK_LEVEL);
                }
                if let Some(d) = &pdeck {
                    d.set_heard(played - DEVICE_LAG as i64);
                }
                played += block as i64;
                pos += block;
            }
            g = target;
            playback.write(&out)?;
        }
    });

    let hop_s = hop as f32 / SAMPLE_RATE as f32;
    let mut pending: Vec<f32> = Vec::new();
    let mut last_grab = Instant::now();
    let mut out_frames = 0i64; // stem frames sent to the player: the buffer's frames
    let mut capturing = false; // sending the plain mix (stems only when looping)
    for v in &rx_in {
        pending.extend(v);
        if let Some(id) = ctl.take_request() {
            if let Some(d) = &deck {
                d.cut("model switch");
            }
            let prev = ctl.model();
            eprintln!("stemflow: switching to {id}");
            drop(st.take()); // free the GPU before loading the next one
            st = Some(match load(&id) {
                Ok((s, target, second)) => {
                    ctl.switched(&id, target, second);
                    delay.store(s.delay(), Ordering::Relaxed);
                    s
                }
                Err(e) => {
                    eprintln!("stemflow: couldn't load {id}: {e:#}; back to {prev}");
                    ctl.switch_failed(&id, &e);
                    load(&prev)?.0
                }
            });
            // A loop that's playing is heard in the new model's stems too.
            if let (Some(d), Some(s)) = (&deck, st.as_mut()) {
                match crate::looper::restem(&d.looper, s, true) {
                    Ok(true) => eprintln!("stemflow: loop separated again with {id}"),
                    Ok(false) => {}
                    Err(e) => eprintln!("stemflow: couldn't separate the loop again: {e:#}"),
                }
            }
            // Skip what queued up while loading, so the delay doesn't grow.
            while let Ok(v) = rx_in.try_recv() {
                pending.extend(v);
            }
            let keep = pending.len().min(2 * hop);
            pending.drain(..pending.len() - keep);
        }
        let Some(st) = st.as_mut() else { continue };
        if let Some(d) = &deck {
            let plain = d.stems_only();
            // A loop to separate: the parts of it still the plain mix.
            if plain && d.looper.restem_range(false).is_some() {
                let t = Instant::now();
                match crate::looper::restem(&d.looper, st, false) {
                    Ok(_) => eprintln!(
                        "stemflow: loop separated in {:.2}s",
                        t.elapsed().as_secs_f32()
                    ),
                    Err(e) => eprintln!("stemflow: couldn't separate the loop: {e:#}"),
                }
            }
            if plain != capturing {
                // the separator's held-back frames are dropped (or start afresh)
                capturing = plain;
                d.cut(if plain {
                    "separating -> capture only"
                } else {
                    "capture only -> separating"
                });
                st.reset();
                delay.store(if plain { 0 } else { st.delay() }, Ordering::Relaxed);
            }
            // where the player is now: the newest input, whose stems land `delay`
            // frames later in the buffer
            let lag = if capturing { 0 } else { st.delay() };
            d.set_input(out_frames + (pending.len() / 2 + lag) as i64);
        }
        while pending.len() >= 2 * hop {
            let backlog = pending.len() as f32 / 2.0 / SAMPLE_RATE as f32;
            if backlog > 2.0 * hop_s {
                eprintln!(
                    "warning: {backlog:.1}s of input queued; compute can't keep up (raise --hop)"
                );
            }
            let chunk: Vec<f32> = pending.drain(..2 * hop).collect();
            // Every hop the separator gets: the tracker's frame count is the timeline.
            if let Some(tr) = &beats {
                tr.push(&chunk);
            }
            out_frames += hop as i64;
            if capturing {
                // capture only: the chunk to the buffer and the speakers as it is
                let stems = Stems {
                    bass: vec![0f32; 2 * hop],
                    band: chunk,
                    extra: Vec::new(),
                };
                if tx_out.send((stems, true)).is_err() {
                    return player.join().unwrap();
                }
                continue;
            }
            // The app is paused (or between songs): silence, and leave the GPU alone.
            // Exact zeros also tell the looper's tail where a pause is, to skip it.
            if chunk.iter().all(|x| x.abs() < 1e-4) {
                let z = vec![0f32; 2 * hop];
                let stems = Stems {
                    bass: z.clone(),
                    band: z,
                    extra: Vec::new(),
                };
                if tx_out.send((stems, false)).is_err() {
                    return player.join().unwrap();
                }
                continue;
            }
            let t = Instant::now();
            let out = st.process(&chunk)?;
            let dt = t.elapsed().as_secs_f32();
            ctl.compute_us.store((dt * 1e6) as u64, Ordering::Relaxed);
            if dt > hop_s {
                ctl.overruns.fetch_add(1, Ordering::Relaxed);
                eprintln!("warning: run took {dt:.2}s > hop {hop_s:.2}s");
            }
            if tx_out.send((out, false)).is_err() {
                return player.join().unwrap();
            }
        }
        // Pick up streams the app opens later (Spotify reopens on device changes).
        if last_grab.elapsed().as_secs() >= 2 {
            last_grab = Instant::now();
            regrab();
        }
    }
    Ok(())
}

const CLICK_LEVEL: f32 = 0.35;
const BEAT_EVERY: std::time::Duration = std::time::Duration::from_secs(3);

/// While the click is on, re-analyse the last 30 s every few seconds.
fn beat_loop(tr: &crate::beats::Tracker, ctl: &Control, running: &AtomicBool) {
    let mut last = Instant::now();
    while running.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(250));
        if !ctl.click() || last.elapsed() < BEAT_EVERY {
            continue;
        }
        last = Instant::now();
        *tr.settings.lock().unwrap() = crate::beats::Settings {
            meter: ctl.meter() as usize,
            feel: ctl.feel(),
            ..Default::default()
        };
        match tr.analyze() {
            Ok(true) => {
                let i = tr.info();
                ctl.set_beats(format!("{:.0} bpm · {}/bar", i.bpm, i.meter));
                eprintln!(
                    "beats: {:.1} bpm, {}/bar, {} beats, analysis {:.2}s, drift {:.0} ms",
                    i.bpm,
                    i.meter,
                    i.beats,
                    i.analysis_s,
                    i.drift_s * 1e3
                );
            }
            Ok(false) => ctl.set_beats(String::new()),
            Err(e) => eprintln!("beats: analysis failed: {e:#}"),
        }
    }
}

/// Frames between what the player writes and what is heard: the device buffer.
const DEVICE_LAG: usize = 5 * BLOCK;

/// `n` frames of the mix from `pos`, the gains ramping from `g0` to `g1` (no zipper
/// noise when they change).
fn mix_block(
    s: [&[f32]; 3],
    pos: usize,
    n: usize,
    g0: (f32, f32, f32),
    g1: (f32, f32, f32),
    out: &mut [f32],
) {
    for i in 0..n {
        let a = (i as f32 + 1.0) / n as f32;
        let gb = g0.0 + (g1.0 - g0.0) * a;
        let gr = g0.1 + (g1.1 - g0.1) * a;
        let gs = g0.2 + (g1.2 - g0.2) * a;
        for ch in 0..2 {
            let k = 2 * (pos + i) + ch;
            out[2 * i + ch] = gb * s[0][k] + gr * s[1][k] + gs * s[2][k];
        }
    }
}

/// Peak of the mix (bass, band, second stem at `gains`) over `n` frames from `start`.
fn mix_peak(s: [&[f32]; 3], start: usize, n: usize, gains: (f32, f32, f32)) -> f32 {
    (2 * start..2 * (start + n)).fold(0f32, |m, k| {
        m.max((gains.0 * s[0][k] + gains.1 * s[1][k] + gains.2 * s[2][k]).abs())
    })
}

/// Block peak limiter with one block of lookahead: the gain ramps down across a
/// block to whatever both it and the next block need, so nothing goes over
/// `CEIL`, and recovers slowly (`RELEASE` per block) so it doesn't pump.
struct Limiter {
    g: f32,
}

impl Limiter {
    const CEIL: f32 = 0.97;
    /// ~6 dB/s at 20 ms blocks.
    const RELEASE: f32 = 1.014;

    fn new() -> Self {
        Limiter { g: 1.0 }
    }

    fn apply(&mut self, out: &mut [f32], next_peak: f32) {
        let peak = out.iter().fold(0f32, |m, x| m.max(x.abs()));
        let need = |p: f32| if p > Self::CEIL { Self::CEIL / p } else { 1.0 };
        let target = (self.g * Self::RELEASE)
            .min(1.0)
            .min(need(peak))
            .min(need(next_peak));
        let n = out.len() / 2;
        if self.g < 1.0 || target < 1.0 {
            for i in 0..n {
                let a = (i as f32 + 1.0) / n as f32;
                let g = self.g + (target - self.g) * a;
                out[2 * i] *= g;
                out[2 * i + 1] *= g;
            }
        }
        self.g = target;
    }
}

#[cfg(test)]
mod limiter_tests {
    use super::*;

    #[test]
    fn never_clips_and_recovers() {
        let mut lim = Limiter::new();
        let block = 960;
        // quiet, then a loud burst (announced one block ahead), then quiet again
        let level = |b: usize| if (5..10).contains(&b) { 3.0 } else { 0.5 };
        for b in 0..400 {
            let mut out: Vec<f32> = (0..2 * block)
                .map(|i| level(b) * ((i / 2) as f32 * 0.05).sin())
                .collect();
            lim.apply(&mut out, level(b + 1));
            let peak = out.iter().fold(0f32, |m, x| m.max(x.abs()));
            assert!(peak <= Limiter::CEIL + 1e-4, "block {b}: {peak}");
        }
        assert!(lim.g > 0.999, "recovered to {}", lim.g);
    }
}
