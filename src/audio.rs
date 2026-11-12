//! Live I/O through the PulseAudio API (served by pipewire-pulse, or real PulseAudio).
//!
//! We create a null sink called `stemflow`, move the target app's streams into it,
//! capture its monitor, run the separator, and play the result to the real output.

use crate::control::Control;
use crate::model::{Separator, SAMPLE_RATE};
use crate::stream::{Stems, Streamer};
use anyhow::{bail, Context, Result};
use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

const SINK: &str = "stemflow";

fn pactl(args: &[&str]) -> Result<String> {
    let out = Command::new("pactl")
        .args(args)
        .output()
        .context("running pactl")?;
    if !out.status.success() {
        bail!(
            "pactl {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Returns the module index if we created the sink (so we unload it on exit).
fn ensure_sink() -> Result<Option<String>> {
    let sinks = pactl(&["list", "short", "sinks"])?;
    if sinks.lines().any(|l| l.split('\t').nth(1) == Some(SINK)) {
        return Ok(None);
    }
    let idx = pactl(&[
        "load-module",
        "module-null-sink",
        &format!("sink_name={SINK}"),
        "sink_properties=device.description=stemflow",
        "rate=44100",
        "channels=2",
    ])?;
    Ok(Some(idx.trim().to_string()))
}

/// Sink-input ids whose application.name contains `needle` (case-insensitive).
fn find_streams(needle: &str) -> Result<Vec<String>> {
    let needle = needle.to_lowercase();
    let mut ids = Vec::new();
    let mut cur: Option<String> = None;
    for line in pactl(&["list", "sink-inputs"])?.lines() {
        let l = line.trim();
        if let Some(id) = l.strip_prefix("Sink Input #") {
            cur = Some(id.to_string());
        } else if let Some(name) = l.strip_prefix("application.name = ") {
            if name.to_lowercase().contains(&needle) {
                ids.extend(cur.clone());
            }
        }
    }
    Ok(ids)
}

fn bytes_to_f32(b: &[u8], out: &mut Vec<f32>) {
    out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
}

fn f32_to_bytes(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub type Live = Streamer<Box<dyn Separator + Send>>;

/// `load` builds the streamer for a model id when the menu switches models.
pub fn live(
    st: Live,
    ctl: Arc<Control>,
    grab: &str,
    output: Option<String>,
    slack: f32,
    load: impl Fn(&str) -> Result<(Live, &'static str)>,
) -> Result<()> {
    let mut st = Some(st);
    let output = match output {
        Some(o) => o,
        None => pactl(&["get-default-sink"])?.trim().to_string(),
    };
    if output == SINK {
        bail!("default sink is `{SINK}` itself; pass --output <real sink> (see `pactl list short sinks`)");
    }
    let module = ensure_sink()?;
    let running = Arc::new(AtomicBool::new(true));
    {
        let (r, module) = (running.clone(), module.clone());
        ctrlc::set_handler(move || {
            r.store(false, Ordering::SeqCst);
            // Unloading the sink makes PipeWire move grabbed streams back to the default.
            if let Some(m) = &module {
                let _ = pactl(&["unload-module", m]);
            }
            // Skip atexit/static destructors: tearing down CUDA under live audio threads segfaults.
            unsafe { libc::_exit(0) };
        })?;
    }

    let spec = Spec {
        format: Format::F32le,
        channels: 2,
        rate: SAMPLE_RATE,
    };
    let frame = 8usize; // bytes per stereo f32 frame
    let rec = Simple::new(
        None,
        "stemflow",
        Direction::Record,
        Some(&format!("{SINK}.monitor")),
        "capture",
        &spec,
        None,
        Some(&BufferAttr {
            maxlength: u32::MAX,
            tlength: u32::MAX,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize: (SAMPLE_RATE as usize / 50 * frame) as u32, // 20 ms
        }),
    )
    .map_err(|e| anyhow::anyhow!("opening capture: {e}"))?;
    let hop = st.as_ref().unwrap().hop;
    // Keep the device buffer small and the jitter buffer here instead, so the mix
    // levels (applied as samples leave for the device) change within ~0.1 s.
    let block = SAMPLE_RATE as usize / 50; // 20 ms
    let play_buf = (5 * block * frame) as u32;
    let play = Simple::new(
        None,
        "stemflow",
        Direction::Playback,
        Some(&output),
        ctl.target(),
        &spec,
        None,
        Some(&BufferAttr {
            maxlength: u32::MAX,
            tlength: play_buf,
            prebuf: u32::MAX,
            minreq: u32::MAX,
            fragsize: u32::MAX,
        }),
    )
    .map_err(|e| anyhow::anyhow!("opening playback on {output}: {e}"))?;

    let grabbed = find_streams(grab)?;
    for id in &grabbed {
        pactl(&["move-sink-input", id, SINK])?;
    }
    eprintln!(
        "stemflow: {} `{grab}` stream(s) -> {SINK} -> {} -> {output}; latency ≈ {:.1}s (Ctrl-C to stop)",
        grabbed.len(),
        ctl.target(),
        (st.as_ref().unwrap().latency() as f32 / SAMPLE_RATE as f32) + slack
    );
    if grabbed.is_empty() {
        eprintln!("  (no `{grab}` stream yet; start playback, or move any app to \"stemflow\" in pavucontrol)");
    }

    // capture -> worker -> playback, each on its own thread.
    let (tx_in, rx_in) = mpsc::channel::<Vec<f32>>();
    let (tx_out, rx_out) = mpsc::channel::<Stems>();
    let r = running.clone();
    std::thread::spawn(move || {
        let mut buf = vec![0u8; SAMPLE_RATE as usize / 50 * frame];
        while r.load(Ordering::Relaxed) {
            if rec.read(&mut buf).is_err() {
                break;
            }
            let mut v = Vec::with_capacity(buf.len() / 4);
            bytes_to_f32(&buf, &mut v);
            if tx_in.send(v).is_err() {
                break;
            }
        }
    });
    let pctl = ctl.clone();
    let refill = hop + (slack * SAMPLE_RATE as f32) as usize;
    let player = std::thread::spawn(move || -> Result<()> {
        let (mut bass, mut band) = (Vec::<f32>::new(), Vec::<f32>::new());
        let mut pos = 0; // frames already played from the front of bass/band
        let mut buffering = true;
        let mut g = pctl.gains();
        let mut out = vec![0f32; 2 * block];
        let mut lim = Limiter::new();
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
                pos = 0;
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
                let target = pctl.gains();
                for i in 0..block {
                    let a = (i as f32 + 1.0) / block as f32; // ramp: no zipper noise
                    let gb = g.0 + (target.0 - g.0) * a;
                    let gr = g.1 + (target.1 - g.1) * a;
                    for ch in 0..2 {
                        let k = 2 * (pos + i) + ch;
                        out[2 * i + ch] = gb * bass[k] + gr * band[k];
                    }
                }
                // Look one block ahead (it's already buffered) so the limiter is down
                // before a peak arrives rather than clipping it.
                let ahead = (bass.len() / 2 - pos - block).min(block);
                let next = mix_peak(&bass, &band, pos + block, ahead, target);
                lim.apply(&mut out, next);
                g = target;
                pos += block;
                play.write(&f32_to_bytes(&out))
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
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
                Ok((s, target)) => {
                    ctl.switched(&id, target);
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
            if let Ok(ids) = find_streams(grab) {
                for id in ids {
                    let _ = pactl(&["move-sink-input", &id, SINK]);
                }
            }
        }
    }
    Ok(())
}

/// The player drains its queue a block at a time and goes back for more input
/// when only one block is left.
fn running_low(bass: &[f32], pos: usize, block: usize) -> bool {
    bass.len() / 2 - pos < 2 * block
}

/// Peak of `gains.0 * bass + gains.1 * band` over `n` frames from `start`.
fn mix_peak(bass: &[f32], band: &[f32], start: usize, n: usize, gains: (f32, f32)) -> f32 {
    bass[2 * start..2 * (start + n)]
        .iter()
        .zip(&band[2 * start..2 * (start + n)])
        .fold(0f32, |m, (b, r)| m.max((gains.0 * b + gains.1 * r).abs()))
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
