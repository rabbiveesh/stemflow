//! Live beat and bar tracking for looping and practice.
//!
//! Beat This! (CPJKU, ISMIR 2024, MIT; exported by tools/export_beat_this.py) runs on
//! the last 30 s of input every few seconds, then a steady-tempo decoder lays a beat
//! grid and bar lines over its logits. The decoder is the "steady" one from the beat
//! shootout page (out/beats/page/beats.js): a madmom-style tempo HMM, then bar lines
//! every N beats where the downbeat curve agrees most, N from a meter hint or chosen.
//!
//! Times are input frames at 44.1 kHz, counted from the first frame pushed, so the
//! player can map what it is playing onto the grid (output frame - separator delay).
//!
//! Shared by the desktop and the Android core (included by path); depends only on
//! `ort`, `realfft` and `anyhow`.

#![allow(dead_code)]

use anyhow::{ensure, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Beat This! frame rate and the fixed number of frames per run (its training chunk).
pub const FPS: usize = 50;
pub const FRAMES: usize = 1500;
pub const SR: usize = 22_050;
pub const HOP: usize = 441;
const NFFT: usize = 1024;
pub const MELS: usize = 128;
/// 22.05 kHz samples that give exactly FRAMES centred frames (29.98 s).
pub const WINDOW: usize = (FRAMES - 1) * HOP;
/// Input rate: everything outside this module is 44.1 kHz.
const IN_RATE: f64 = 44_100.0;
/// Least audio before the first analysis; the rest of the window is zero-padded.
const MIN_AUDIO: usize = 6 * SR;
/// Audio an analysis needs before its tempo and meter are held against the next ones
/// (shorter windows often land on half or double time).
const HOLD_AUDIO: usize = 15 * SR;

// ---------------------------------------------------------------- input: 44.1 -> 22.05 kHz mono

/// Low-pass FIR (Blackman-windowed sinc, cutoff 10.5 kHz) and keep every other sample.
/// Output k is centred on input sample 2k - DECIM_DELAY.
pub(crate) struct Decimator {
    taps: Vec<f32>,
    hist: Vec<f32>,
    n: u64,
}

const DECIM_TAPS: usize = 63;
pub(crate) const DECIM_DELAY: i64 = (DECIM_TAPS as i64 - 1) / 2;

impl Decimator {
    pub(crate) fn new() -> Self {
        let fc = 10_500.0 / IN_RATE; // cycles per input sample
        let m = (DECIM_TAPS - 1) as f64;
        let mut taps: Vec<f64> = (0..DECIM_TAPS)
            .map(|i| {
                let x = i as f64 - m / 2.0;
                let sinc = if x == 0.0 {
                    2.0 * fc
                } else {
                    (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x)
                };
                let a = 2.0 * std::f64::consts::PI * i as f64 / m;
                sinc * (0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos())
            })
            .collect();
        let s: f64 = taps.iter().sum();
        taps.iter_mut().for_each(|t| *t /= s);
        Decimator {
            taps: taps.into_iter().map(|t| t as f32).collect(),
            hist: Vec::new(),
            n: 0,
        }
    }

    pub(crate) fn push(&mut self, mono: &[f32], out: &mut Vec<f32>) {
        let l = self.taps.len();
        for &s in mono {
            self.hist.push(s);
            if self.n.is_multiple_of(2) {
                let h = &self.hist;
                let len = h.len();
                let mut acc = 0f32;
                for j in 0..l.min(len) {
                    acc += self.taps[j] * h[len - 1 - j];
                }
                out.push(acc);
            }
            self.n += 1;
            if self.hist.len() > 8 * l {
                self.hist.drain(..self.hist.len() - l);
            }
        }
    }
}

// ---------------------------------------------------------------- log-mel spectrogram

/// Beat This!'s LogMelSpect: torchaudio MelSpectrogram(22050, n_fft 1024, hop 441,
/// 30-11000 Hz, 128 slaney mels, no mel norm, normalized="frame_length", power 1),
/// centred frames with reflect padding, then log1p(1000 x).
pub struct Mel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// per mel: first bin and the weights from there
    fb: Vec<(usize, Vec<f32>)>,
}

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    if f >= 1000.0 {
        1000.0 / f_sp + (f / 1000.0).ln() / (6.4f64.ln() / 27.0)
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_mel = 1000.0 / f_sp;
    if m >= min_log_mel {
        1000.0 * ((6.4f64.ln() / 27.0) * (m - min_log_mel)).exp()
    } else {
        m * f_sp
    }
}

impl Mel {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(NFFT);
        // periodic Hann, scaled by 1/sqrt(n_fft): torchaudio's "frame_length" is
        // torch.stft(normalized=True), not a window-energy norm
        let norm = (NFFT as f64).sqrt();
        let window = (0..NFFT)
            .map(|n| {
                ((0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / NFFT as f64).cos()) / norm)
                    as f32
            })
            .collect();
        let bins = NFFT / 2 + 1;
        let freqs: Vec<f64> = (0..bins)
            .map(|i| i as f64 * (SR as f64 / 2.0) / (bins - 1) as f64)
            .collect();
        let (m0, m1) = (hz_to_mel(30.0), hz_to_mel(11_000.0));
        let pts: Vec<f64> = (0..MELS + 2)
            .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (MELS + 1) as f64))
            .collect();
        let fb = (0..MELS)
            .map(|m| {
                let w: Vec<f32> = freqs
                    .iter()
                    .map(|&f| {
                        let down = (f - pts[m]) / (pts[m + 1] - pts[m]);
                        let up = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1]);
                        down.min(up).max(0.0) as f32
                    })
                    .collect();
                let first = w.iter().position(|&x| x > 0.0).unwrap_or(0);
                let last = w.iter().rposition(|&x| x > 0.0).unwrap_or(0);
                (first, w[first..=last.max(first)].to_vec())
            })
            .collect();
        Mel { fft, window, fb }
    }

    /// `x`: WINDOW samples at 22.05 kHz. Returns FRAMES x MELS, frame-major.
    pub fn spect(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), WINDOW);
        self.spect_any(x)
    }

    /// Any length (at least NFFT/2 + 1 samples): 1 + len / HOP frames, frame-major.
    pub fn spect_any(&self, x: &[f32]) -> Vec<f32> {
        let frames = 1 + x.len() / HOP;
        let pad = NFFT as i64 / 2;
        let len = x.len() as i64;
        let at = |i: i64| {
            let j = i - pad;
            let j = if j < 0 {
                -j
            } else if j >= len {
                2 * (len - 1) - j
            } else {
                j
            };
            x[j as usize]
        };
        let mut out = vec![0f32; frames * MELS];
        let mut buf = vec![0f32; NFFT];
        let mut spec = vec![Complex32::default(); NFFT / 2 + 1];
        let mut scratch = self.fft.make_scratch_vec();
        let mut mag = vec![0f32; NFFT / 2 + 1];
        for t in 0..frames {
            let s = (t * HOP) as i64;
            for (n, (b, w)) in buf.iter_mut().zip(&self.window).enumerate() {
                *b = at(s + n as i64) * w;
            }
            self.fft
                .process_with_scratch(&mut buf, &mut spec, &mut scratch)
                .unwrap();
            for (m, c) in mag.iter_mut().zip(&spec) {
                *m = c.norm();
            }
            for (k, (first, w)) in self.fb.iter().enumerate() {
                let v: f32 = w.iter().zip(&mag[*first..]).map(|(a, b)| a * b).sum();
                out[t * MELS + k] = (1000.0 * v).ln_1p();
            }
        }
        out
    }
}

// ---------------------------------------------------------------- the network

/// Load libonnxruntime once per process (Android: the app's copy; desktop: as model.rs).
pub fn ort_env(lib: &Path) -> Result<&'static ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init_from(lib)
        .map_err(|e| anyhow::anyhow!("loading {}: {e}", lib.display()))?
        .with_name("stemflow-beats")
        .build()?;
    Ok(ENV.get_or_init(|| env))
}

/// A Beat This! runtime: ONNX Runtime here (desktop), LiteRT on the phone
/// (android/bench/src/litert.rs).
pub trait BeatModel: Send {
    /// FRAMES x MELS spectrogram in; beat and downbeat logits (FRAMES each) out.
    fn run(&mut self, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>)>;
}

pub struct BeatNet {
    session: Session,
}

impl BeatNet {
    /// `gpu`: run on CUDA (desktop builds with the `cuda` feature; ignored elsewhere).
    pub fn load(
        env: &ort::environment::Environment,
        path: &Path,
        threads: usize,
        gpu: bool,
    ) -> Result<Self> {
        let b = Session::builder(env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        #[cfg(feature = "cuda")]
        let b = if gpu {
            b.with_execution_providers([ort::ep::CUDA::default()
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                .with_conv_max_workspace(false)
                .build()
                .error_on_failure()])
                .map_err(|e| anyhow::anyhow!("{e}"))?
        } else {
            b
        };
        #[cfg(not(feature = "cuda"))]
        let _ = gpu;
        let mut b = b;
        let session = b
            .commit_from_file(path)
            .with_context(|| format!("loading beat model {}", path.display()))?;
        Ok(BeatNet { session })
    }
}

impl BeatModel for BeatNet {
    fn run(&mut self, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        let out = self.session.run(
            ort::inputs!["spect" => Tensor::from_array(([1usize, FRAMES, MELS], spect.to_vec()))?],
        )?;
        let (_, b) = out["beat"].try_extract_tensor::<f32>()?;
        let (_, d) = out["downbeat"].try_extract_tensor::<f32>()?;
        Ok((b.to_vec(), d.to_vec()))
    }
}

// ---------------------------------------------------------------- decoding

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Beat times (s) from beat logits: Viterbi over (beat period tau in frames, phase).
/// The tempo can only change at a beat, and slowly (madmom's DBN, lambda 100).
pub fn steady_beats(logits: &[f32], min_bpm: f32, max_bpm: f32) -> Vec<f64> {
    let (lambda, obs_lambda) = (100.0f64, 16.0f64);
    let fps = FPS as f64;
    let t_min = ((60.0 * fps / max_bpm as f64).floor() as usize).max(2);
    let t_max = ((60.0 * fps / min_bpm as f64).ceil() as usize).max(t_min);
    let taus: Vec<usize> = (t_min..=t_max).collect();
    let k_n = taus.len();
    let mut off = Vec::with_capacity(k_n);
    let mut s_n = 0;
    for &t in &taus {
        off.push(s_n);
        s_n += t;
    }
    let n = logits.len();
    if n == 0 {
        return vec![];
    }
    let p: Vec<f64> = logits
        .iter()
        .map(|&x| sig(x as f64).clamp(1e-6, 1.0 - 1e-6))
        .collect();
    let lb: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let ln: Vec<f64> = p
        .iter()
        .map(|x| ((1.0 - x) / (obs_lambda - 1.0)).ln())
        .collect();
    let mut trans = vec![0f64; k_n * k_n];
    for a in 0..k_n {
        let r = |b: usize| -lambda * (taus[b] as f64 / taus[a] as f64 - 1.0).abs();
        let z: f64 = (0..k_n).map(|b| r(b).exp()).sum();
        for b in 0..k_n {
            trans[a * k_n + b] = r(b) - z.ln();
        }
    }
    let in_beat = |k: usize, phi: usize| (phi as f64) < (taus[k] as f64 / obs_lambda).max(1.0);
    let mut v = vec![0f64; s_n];
    let mut w = vec![0f64; s_n];
    for k in 0..k_n {
        for phi in 0..taus[k] {
            v[off[k] + phi] = -(s_n as f64).ln() + if in_beat(k, phi) { lb[0] } else { ln[0] };
        }
    }
    let mut back = vec![0u16; n * k_n];
    for t in 1..n {
        for k in 0..k_n {
            let o = off[k];
            for phi in 1..taus[k] {
                w[o + phi] = v[o + phi - 1] + if in_beat(k, phi) { lb[t] } else { ln[t] };
            }
        }
        for b in 0..k_n {
            let (mut best, mut arg) = (f64::NEG_INFINITY, 0);
            for a in 0..k_n {
                let s = v[off[a] + taus[a] - 1] + trans[a * k_n + b];
                if s > best {
                    best = s;
                    arg = a;
                }
            }
            w[off[b]] = best + lb[t];
            back[t * k_n + b] = arg as u16;
        }
        std::mem::swap(&mut v, &mut w);
    }
    let (mut best, mut k, mut phi) = (f64::NEG_INFINITY, 0, 0);
    for kk in 0..k_n {
        for pp in 0..taus[kk] {
            if v[off[kk] + pp] > best {
                best = v[off[kk] + pp];
                k = kk;
                phi = pp;
            }
        }
    }
    let mut beats = Vec::new();
    for t in (0..n).rev() {
        if phi == 0 {
            beats.push(t as f64 / fps);
            if t > 0 {
                k = back[t * k_n + k] as usize;
                phi = taus[k] - 1;
            }
        } else {
            phi -= 1;
        }
    }
    beats.reverse();
    beats
}

/// Probability of the curve near `time` (max logit within ±2 frames).
fn at(curve: &[f32], time: f64) -> f64 {
    let i = (time * FPS as f64).round() as i64;
    let lo = (i - 2).max(0) as usize;
    let hi = ((i + 2).min(curve.len() as i64 - 1)).max(0) as usize;
    let m = curve[lo.min(hi)..=hi]
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    sig(m as f64)
}

/// Bar lines every `m` beats at the offset the downbeat curve supports most.
/// Returns (offset of the first bar line in `beats`, contrast vs the other beats).
pub fn bars(beats: &[f64], down: &[f32], m: usize) -> (usize, f64) {
    let pd: Vec<f64> = beats.iter().map(|&b| at(down, b)).collect();
    let (mut best_k, mut best) = (0, f64::NEG_INFINITY);
    for k in 0..m {
        let sel: Vec<f64> = pd.iter().skip(k).step_by(m).copied().collect();
        if !sel.is_empty() {
            let s = sel.iter().sum::<f64>() / sel.len() as f64;
            if s > best {
                best = s;
                best_k = k;
            }
        }
    }
    let other: Vec<f64> = pd
        .iter()
        .enumerate()
        .filter(|(i, _)| !(*i + m - best_k % m).is_multiple_of(m))
        .map(|(_, &p)| p)
        .collect();
    let mean_other = if other.is_empty() {
        0.0
    } else {
        other.iter().sum::<f64>() / other.len() as f64
    };
    (best_k, best - mean_other)
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub min_bpm: f32,
    pub max_bpm: f32,
    /// Beats per bar; 0 = choose 3-7 from the downbeat curve (slight bias to 4).
    pub meter: usize,
    /// 0.5 = half time (every other beat), 2 = double time, 1 = as decoded.
    pub feel: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            min_bpm: 55.0,
            max_bpm: 300.0,
            meter: 0,
            feel: 1.0,
        }
    }
}

pub struct Decoded {
    /// seconds from the start of the analysed window
    pub beats: Vec<f64>,
    /// index in `beats` of the first bar line; bar lines every `meter` beats from there
    pub first_bar: usize,
    pub meter: usize,
    pub bpm: f32,
}

pub fn decode(beat: &[f32], down: &[f32], s: &Settings) -> Decoded {
    let mut beats = steady_beats(beat, s.min_bpm, s.max_bpm);
    if s.feel >= 1.5 && beats.len() >= 2 {
        let mut out = Vec::with_capacity(2 * beats.len());
        for i in 0..beats.len() {
            out.push(beats[i]);
            if i + 1 < beats.len() {
                out.push((beats[i] + beats[i + 1]) / 2.0);
            }
        }
        beats = out;
    } else if s.feel <= 0.75 && beats.len() >= 2 {
        let sum = |k: usize| {
            beats
                .iter()
                .skip(k)
                .step_by(2)
                .map(|&b| at(beat, b))
                .sum::<f64>()
        };
        let k = if sum(0) >= sum(1) { 0 } else { 1 };
        beats = beats.into_iter().skip(k).step_by(2).collect();
    }
    let (first_bar, meter) = if s.meter == 0 {
        let mut best = (0, 4, f64::NEG_INFINITY);
        for m in 3..=7 {
            let (k, c) = bars(&beats, down, m);
            let score = c + if m == 4 { 0.03 } else { 0.0 };
            if score > best.2 {
                best = (k, m, score);
            }
        }
        (best.0, best.1)
    } else {
        (bars(&beats, down, s.meter).0, s.meter)
    };
    let mut d: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bpm = if d.is_empty() {
        0.0
    } else {
        (60.0 / d[d.len() / 2]) as f32
    };
    Decoded {
        beats,
        first_bar,
        meter,
        bpm,
    }
}

// ---------------------------------------------------------------- the live tracker

/// A beat grid in input frames (44.1 kHz since the first push).
pub struct Grid {
    pub beats: Vec<f64>,
    /// whether each beat is a bar line
    pub bar: Vec<bool>,
    /// beat period at the end of the window (median of the last LOCAL beats), input
    /// frames: the click carries on at this tempo past the analysed audio
    pub period: f64,
    /// input frame where the analysed audio ends
    pub end: f64,
    pub bpm: f32,
    pub meter: usize,
}

impl Grid {
    /// Beats in [from, to), continuing past the analysed end at the last period for up to
    /// `extend` frames, so the click carries on until the next analysis lands.
    pub fn beats_between(&self, from: f64, to: f64, extend: f64) -> Vec<(f64, bool)> {
        let mut out: Vec<(f64, bool)> = self
            .beats
            .iter()
            .zip(&self.bar)
            .filter(|(&b, _)| b >= from && b < to)
            .map(|(&b, &d)| (b, d))
            .collect();
        let Some(&last) = self.beats.last() else {
            return out;
        };
        if self.period <= 0.0 {
            return out;
        }
        // beats since the last bar line, to keep counting bars past the end
        let since = self.bar.iter().rev().position(|&d| d).unwrap_or(0);
        let mut i = 1;
        loop {
            let b = last + i as f64 * self.period;
            if b >= to || b > self.end + extend {
                break;
            }
            if b >= from {
                out.push((b, (since + i) % self.meter.max(1) == 0));
            }
            i += 1;
        }
        out
    }
}

/// Index in `ring` just after the last gap: at least GAP of near silence (a pause,
/// the space between tracks) or, failing that, the first sound.
const GAP: usize = SR / 2;

fn start_after_gap(ring: &VecDeque<f32>) -> usize {
    let block = SR / 50; // 20 ms
    let n = ring.len() / block;
    let quiet = |i: usize| {
        let e: f32 = ring.range(i * block..(i + 1) * block).map(|v| v * v).sum();
        (e / block as f32).sqrt() < 1e-3 // -60 dBFS
    };
    let mut run = 0;
    for i in (0..n).rev() {
        if quiet(i) {
            run += 1;
            if run * block >= GAP {
                return ((i + run) * block).min(ring.len());
            }
        } else {
            run = 0;
        }
    }
    ring.iter()
        .position(|v| v.abs() > 1e-4)
        .unwrap_or(ring.len())
}

struct Input {
    decim: Decimator,
    ring: VecDeque<f32>,
    /// 22.05 kHz samples produced so far
    produced: u64,
    mono: Vec<f32>,
    tmp: Vec<f32>,
}

/// What the last analyses settled on. A different tempo (half, double, ...) or meter
/// must come up in three analyses in a row (9 s) before the grid follows it; until then the
/// decoder is held near the old one. One odd window (a break, a fill, a tag) then
/// doesn't halve the click for a few seconds.
#[derive(Default)]
struct Memory {
    /// tempo before the feel is applied, and the settings it was found with
    bpm: Option<f32>,
    meter: Option<usize>,
    settings: Option<(f32, f32, usize, f32)>,
    tempo_doubts: u32,
    meter_doubts: u32,
}

const CONFIRM: u32 = 3;
/// Beats the carry-on tempo is taken from.
const LOCAL: usize = 8;

pub struct Info {
    pub bpm: f32,
    pub meter: usize,
    pub analysis_s: f32,
    pub beats: usize,
    /// how far the click had drifted when this analysis landed, s (-1: no previous grid)
    pub drift_s: f32,
}

/// Feed from the separator's thread, analyse from a background thread, render clicks
/// from the player: each part locks only what it needs, briefly.
pub struct Tracker {
    input: Mutex<Input>,
    memory: Mutex<Memory>,
    net: Mutex<Box<dyn BeatModel>>,
    mel: Mel,
    grid: Mutex<Option<Arc<Grid>>>,
    pub settings: Mutex<Settings>,
    last_s: Mutex<f32>,
    /// worst gap between the previous grid's carried-on beats and this one's, s (-1: none)
    drift_s: Mutex<f32>,
    /// padding of the last analysed window: it grows again after a gap
    last_pad: AtomicUsize,
    clicks: [Vec<f32>; 2],
}

fn click_sound(freq: f64) -> Vec<f32> {
    let n = (IN_RATE * 0.04) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / IN_RATE;
            ((2.0 * std::f64::consts::PI * freq * t).sin() * (-t * 90.0).exp()) as f32
        })
        .collect()
}

impl Tracker {
    pub fn new(net: Box<dyn BeatModel>) -> Self {
        Tracker {
            memory: Mutex::new(Memory::default()),
            input: Mutex::new(Input {
                decim: Decimator::new(),
                ring: VecDeque::with_capacity(WINDOW + SR),
                produced: 0,
                mono: Vec::new(),
                tmp: Vec::new(),
            }),
            net: Mutex::new(net),
            mel: Mel::new(),
            grid: Mutex::new(None),
            settings: Mutex::new(Settings::default()),
            last_s: Mutex::new(0.0),
            drift_s: Mutex::new(-1.0),
            last_pad: AtomicUsize::new(WINDOW),
            clicks: [click_sound(1000.0), click_sound(1700.0)],
        }
    }

    /// Interleaved stereo at 44.1 kHz, in order, with nothing skipped (the frame count
    /// is the timeline the grid is in).
    pub fn push(&self, stereo: &[f32]) {
        let mut g = self.input.lock().unwrap();
        let inp = &mut *g;
        inp.mono.clear();
        inp.mono.extend(
            stereo
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| 0.5 * (c[0] + c[1])),
        );
        inp.tmp.clear();
        inp.decim.push(&inp.mono, &mut inp.tmp);
        inp.produced += inp.tmp.len() as u64;
        inp.ring.extend(inp.tmp.iter());
        let excess = inp.ring.len().saturating_sub(WINDOW);
        inp.ring.drain(..excess);
    }

    /// Run the model and the decoder on the latest window and swap in the new grid.
    /// Returns false if there wasn't enough (non-silent) audio to analyse.
    pub fn analyze(&self) -> Result<bool> {
        let t0 = std::time::Instant::now();
        let (x, produced, pad) = {
            let g = self.input.lock().unwrap();
            // Audio before the last gap (the music starting, a new track, a pause)
            // counts as padding, not audio: one song's tempo says nothing about the next.
            let lead = start_after_gap(&g.ring);
            let audio = g.ring.len() - lead;
            if audio < MIN_AUDIO {
                // just after a gap: the old grid belongs to the last piece; stop clicking
                *self.grid.lock().unwrap() = None;
                return Ok(false);
            }
            let pad = WINDOW - audio;
            let mut x = vec![0f32; pad]; // zero-pad the front at the start
            x.extend(g.ring.iter().skip(lead));
            (x, g.produced, pad)
        };
        // the last 10 s silent (paused): drop the grid, stop clicking
        let tail = &x[WINDOW - 10 * SR..];
        let rms = (tail.iter().map(|v| v * v).sum::<f32>() / tail.len() as f32).sqrt();
        if rms < 1e-4 {
            *self.grid.lock().unwrap() = None;
            *self.memory.lock().unwrap() = Memory::default();
            return Ok(false);
        }
        let spect = self.mel.spect(&x);
        let (b, d) = self.net.lock().unwrap().run(&spect)?;
        ensure!(
            b.len() == FRAMES && d.len() == FRAMES,
            "beat model gave {} frames",
            b.len()
        );
        let s = *self.settings.lock().unwrap();
        // Decode only the frames over real audio: over the zero padding at the start
        // the decoder would lay beats at any tempo and skew the median.
        let skip = (pad / HOP + 1).min(FRAMES - 1);
        // A window that starts after a gap is a new piece: forget the old tempo.
        if pad > 0 && self.last_pad.swap(pad, Ordering::Relaxed) < pad {
            *self.memory.lock().unwrap() = Memory::default();
        }
        let dec = self.decode_steadily(&b[skip..], &d[skip..], s, WINDOW - pad >= HOLD_AUDIO);
        // seconds in the window -> input frames: 22.05 kHz sample k is input frame 2k - delay
        let start22 = produced as f64 - WINDOW as f64 + (skip * HOP) as f64;
        let to_input = |t: f64| 2.0 * (start22 + t * SR as f64) - DECIM_DELAY as f64;
        let beats: Vec<f64> = dec.beats.iter().map(|&t| to_input(t)).collect();
        let bar = (0..beats.len())
            .map(|i| i >= dec.first_bar && (i - dec.first_bar).is_multiple_of(dec.meter))
            .collect();
        // The tempo now, not the 30 s average: a band pushes and pulls, and the decoder's
        // tempo follows slowly, so its last beats say where the next ones will fall.
        let tail = &beats[beats.len().saturating_sub(LOCAL + 1)..];
        let mut d: Vec<f64> = tail.windows(2).map(|w| w[1] - w[0]).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let period = if d.is_empty() { 0.0 } else { d[d.len() / 2] };
        // How far the old grid's carried-on beats were from where this analysis puts
        // them (what was heard as drift), over the audio this one adds.
        let end = 2.0 * produced as f64 - DECIM_DELAY as f64; // the newest input analysed
        let drift = self.grid().and_then(|old| {
            let ext = old.beats_between(old.end, end, f64::INFINITY);
            let errs: Vec<f64> = beats
                .iter()
                .filter(|&&b| b > old.end && b <= end)
                .filter_map(|&b| {
                    ext.iter()
                        .map(|&(e, _)| b - e)
                        .min_by(|x, y| x.abs().total_cmp(&y.abs()))
                })
                .filter(|e| e.abs() < period / 2.0)
                .collect();
            (!errs.is_empty()).then(|| errs.iter().map(|e| e.abs()).fold(0.0, f64::max) / IN_RATE)
        });
        *self.drift_s.lock().unwrap() = drift.map_or(-1.0, |d| d as f32);
        *self.grid.lock().unwrap() = Some(Arc::new(Grid {
            beats,
            bar,
            period,
            end,
            bpm: dec.bpm,
            meter: dec.meter,
        }));
        *self.last_s.lock().unwrap() = t0.elapsed().as_secs_f32();
        Ok(true)
    }

    /// `decode`, held to the previous tempo and meter until a change is confirmed.
    fn decode_steadily(&self, b: &[f32], d: &[f32], s: Settings, enough: bool) -> Decoded {
        let mut m = self.memory.lock().unwrap();
        let key = (s.min_bpm, s.max_bpm, s.meter, s.feel);
        if m.settings != Some(key) {
            *m = Memory {
                settings: Some(key),
                ..Default::default()
            };
        }
        let free = decode(b, d, &s);
        let raw = |dec: &Decoded| dec.bpm / s.feel; // the tempo before half/double time
        let mut dec = free;
        if let Some(prev) = m.bpm {
            let ratio = raw(&dec) / prev;
            if (0.87..1.15).contains(&ratio) {
                m.tempo_doubts = 0;
            } else {
                m.tempo_doubts += 1;
                if m.tempo_doubts < CONFIRM {
                    let held = Settings {
                        min_bpm: prev / 1.15,
                        max_bpm: prev * 1.15,
                        ..s
                    };
                    dec = decode(b, d, &held);
                } else {
                    m.tempo_doubts = 0;
                }
            }
        }
        if s.meter == 0 {
            if let Some(pm) = m.meter.filter(|&pm| pm != dec.meter) {
                m.meter_doubts += 1;
                if m.meter_doubts < CONFIRM {
                    let (first_bar, _) = bars(&dec.beats, d, pm);
                    dec.first_bar = first_bar;
                    dec.meter = pm;
                } else {
                    m.meter_doubts = 0;
                }
            } else {
                m.meter_doubts = 0;
            }
        }
        // Only a long enough window sets what later ones are held to.
        if dec.bpm > 0.0 && enough {
            m.bpm = Some(raw(&dec));
            m.meter = Some(dec.meter);
        }
        dec
    }

    pub fn grid(&self) -> Option<Arc<Grid>> {
        self.grid.lock().unwrap().clone()
    }

    pub fn info(&self) -> Info {
        let g = self.grid();
        Info {
            bpm: g.as_ref().map_or(0.0, |g| g.bpm),
            meter: g.as_ref().map_or(0, |g| g.meter),
            analysis_s: *self.last_s.lock().unwrap(),
            beats: g.as_ref().map_or(0, |g| g.beats.len()),
            drift_s: *self.drift_s.lock().unwrap(),
        }
    }

    /// Add clicks (high on bar lines) to interleaved stereo `out`, whose first frame is
    /// input frame `start`. Keeps clicking at the last tempo for up to 8 s past the
    /// analysed audio.
    pub fn click(&self, out: &mut [f32], start: i64, level: f32) {
        let Some(g) = self.grid() else { return };
        let n = out.len() / 2;
        let len = self.clicks[0].len();
        let from = start as f64 - len as f64;
        let to = start as f64 + n as f64;
        for (b, bar) in g.beats_between(from, to, 8.0 * IN_RATE) {
            let c = &self.clicks[bar as usize];
            let b = b.round() as i64;
            for i in 0..n {
                let k = start + i as i64 - b;
                if k >= 0 && (k as usize) < len {
                    let v = level * c[k as usize];
                    out[2 * i] = (out[2 * i] + v).clamp(-1.0, 1.0);
                    out[2 * i + 1] = (out[2 * i + 1] + v).clamp(-1.0, 1.0);
                }
            }
        }
    }
}
