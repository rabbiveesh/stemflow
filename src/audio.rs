//! Live I/O through the PulseAudio API (served by pipewire-pulse, or real PulseAudio).
//!
//! We create a null sink called `stemflow`, move the target app's streams into it,
//! capture its monitor, run the separator, and play the result to the real output.

use crate::model::{Separator, SAMPLE_RATE};
use crate::stream::Streamer;
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

pub fn live<S: Separator + Send + 'static>(
    mut st: Streamer<S>,
    grab: &str,
    output: Option<String>,
    slack: f32,
) -> Result<()> {
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
    let hop = st.hop;
    let play_buf = ((2 * hop + (slack * SAMPLE_RATE as f32) as usize) * frame) as u32;
    let play = Simple::new(
        None,
        "stemflow",
        Direction::Playback,
        Some(&output),
        "bass",
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
        "stemflow: {} `{grab}` stream(s) -> {SINK} -> bass -> {output}; latency ≈ {:.1}s (Ctrl-C to stop)",
        grabbed.len(),
        (st.latency() as f32 / SAMPLE_RATE as f32) + slack
    );
    if grabbed.is_empty() {
        eprintln!("  (no `{grab}` stream yet; start playback, or move any app to \"stemflow\" in pavucontrol)");
    }

    // capture -> worker -> playback, each on its own thread.
    let (tx_in, rx_in) = mpsc::channel::<Vec<f32>>();
    let (tx_out, rx_out) = mpsc::channel::<Vec<f32>>();
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
    let player = std::thread::spawn(move || -> Result<()> {
        let silence = vec![0f32; 2 * (slack * SAMPLE_RATE as f32) as usize];
        play.write(&f32_to_bytes(&silence))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for chunk in rx_out {
            play.write(&f32_to_bytes(&chunk))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(())
    });

    let hop_s = hop as f32 / SAMPLE_RATE as f32;
    let mut pending: Vec<f32> = Vec::new();
    let mut last_grab = Instant::now();
    for v in rx_in {
        pending.extend(v);
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
            if dt > hop_s {
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
