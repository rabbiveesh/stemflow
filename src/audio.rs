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
pub fn live(
    st: Live,
    ctl: Arc<Control>,
    beats: Option<Arc<crate::beats::Tracker>>,
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
    let (tx_out, rx_out) = mpsc::channel::<Stems>();
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
    let refill = hop + (slack * SAMPLE_RATE as f32) as usize;
    let player = std::thread::spawn(move || -> Result<()> {
        // bass, band and a second stem (all zero for one-stem models)
        let (mut bass, mut band, mut sec) =
            (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new());
        let mut pos = 0; // frames already played from the front of the buffers
        let mut buffering = true;
        let mut g = pctl.gains3();
        let mut out = vec![0f32; 2 * block];
        let mut lim = Limiter::new();
        let mut played = 0i64; // stem frames played: input frame (played - delay)
        loop {
            // Take whatever the worker has finished; block only when starved.
            let avail = bass.len() / 2 - pos;
            let next = if buffering || avail < block {
                rx_out.recv().ok()
            } else {
                rx_out.try_recv().ok()
            };
            if let Some(st) = next {
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
            } else if buffering || avail < block {
                return Ok(()); // worker gone
            }
            let avail = bass.len() / 2 - pos;
            if buffering {
                if avail < refill {
                    continue;
                }
                buffering = false;
            }
            while bass.len() / 2 - pos >= block {
                let target = pctl.gains3();
                for i in 0..block {
                    let a = (i as f32 + 1.0) / block as f32; // ramp: no zipper noise
                    let gb = g.0 + (target.0 - g.0) * a;
                    let gr = g.1 + (target.1 - g.1) * a;
                    let gs = g.2 + (target.2 - g.2) * a;
                    for ch in 0..2 {
                        let k = 2 * (pos + i) + ch;
                        out[2 * i + ch] = gb * bass[k] + gr * band[k] + gs * sec[k];
                    }
                }
                // Look one block ahead (it's already buffered) so the limiter is down
                // before a peak arrives rather than clipping it.
                let ahead = (bass.len() / 2 - pos - block).min(block);
                let next = mix_peak([&bass, &band, &sec], pos + block, ahead, target);
                lim.apply(&mut out, next);
                if let Some(tr) = pbeats.as_ref().filter(|_| pctl.click()) {
                    let d = pdelay.load(Ordering::Relaxed) as i64;
                    tr.click(&mut out, played - d, CLICK_LEVEL);
                }
                played += block as i64;
                g = target;
                pos += block;
                playback.write(&out)?;
                if running_low(&bass, pos, block) {
                    break; // go fetch the next chunk once we're down to one block
                }
            }
            if bass.len() / 2 - pos == 0 {
                eprintln!("warning: playback ran dry; rebuffering");
                buffering = true;
            }
        }
    });

    let hop_s = hop as f32 / SAMPLE_RATE as f32;
    let mut pending: Vec<f32> = Vec::new();
    let mut last_grab = Instant::now();
    for v in &rx_in {
        pending.extend(v);
        if let Some(id) = ctl.take_request() {
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
            // Skip what queued up while loading, so the delay doesn't grow.
            while let Ok(v) = rx_in.try_recv() {
                pending.extend(v);
            }
            let keep = pending.len().min(2 * hop);
            pending.drain(..pending.len() - keep);
        }
        let Some(st) = st.as_mut() else { continue };
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
            let t = Instant::now();
            let out = st.process(&chunk)?;
            let dt = t.elapsed().as_secs_f32();
            ctl.compute_us.store((dt * 1e6) as u64, Ordering::Relaxed);
            if dt > hop_s {
                ctl.overruns.fetch_add(1, Ordering::Relaxed);
                eprintln!("warning: run took {dt:.2}s > hop {hop_s:.2}s");
            }
            if tx_out.send(out).is_err() {
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

/// The player drains its queue a block at a time and goes back for more input
/// when only one block is left.
fn running_low(bass: &[f32], pos: usize, block: usize) -> bool {
    bass.len() / 2 - pos < 2 * block
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
