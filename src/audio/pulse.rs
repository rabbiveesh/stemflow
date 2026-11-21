//! Linux I/O through the PulseAudio API (served by pipewire-pulse, or real PulseAudio).
//!
//! We create a null sink called `stemflow`, move the target app's streams into it,
//! capture its monitor, and play the result to the real output.

use super::{Capture, Io, Playback};
use crate::model::SAMPLE_RATE;
use anyhow::{bail, Context, Result};
use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use std::process::Command;

const SINK: &str = "stemflow";
const FRAME: usize = 8; // bytes per stereo f32 frame

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

struct Rec(Simple, Vec<u8>);

impl Capture for Rec {
    fn read(&mut self) -> Result<Vec<f32>> {
        self.0
            .read(&mut self.1)
            .map_err(|e| anyhow::anyhow!("capture: {e}"))?;
        Ok(self
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }
}

struct Play(Simple);

impl Playback for Play {
    fn write(&mut self, x: &[f32]) -> Result<()> {
        let b: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.0.write(&b).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

pub fn open(grab: &str, output: Option<String>, name: &str, play_frames: usize) -> Result<Io> {
    let output = match output {
        Some(o) => o,
        None => pactl(&["get-default-sink"])?.trim().to_string(),
    };
    if output == SINK {
        bail!("default sink is `{SINK}` itself; pass --output <real sink> (see `pactl list short sinks`)");
    }
    let module = ensure_sink()?;
    let spec = Spec {
        format: Format::F32le,
        channels: 2,
        rate: SAMPLE_RATE,
    };
    let block = SAMPLE_RATE as usize / 50; // 20 ms
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
            fragsize: (block * FRAME) as u32,
        }),
    )
    .map_err(|e| anyhow::anyhow!("opening capture: {e}"))?;
    let play = Simple::new(
        None,
        "stemflow",
        Direction::Playback,
        Some(&output),
        name,
        &spec,
        None,
        Some(&BufferAttr {
            maxlength: u32::MAX,
            tlength: (play_frames * FRAME) as u32,
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
    let hint = grabbed.is_empty().then(|| {
        format!("(no `{grab}` stream yet; start playback, or move any app to \"stemflow\" in pavucontrol)")
    });
    let needle = grab.to_string();
    Ok(Io {
        capture: Box::new(Rec(rec, vec![0u8; block * FRAME])),
        playback: Box::new(Play(play)),
        // Pick up streams the app opens later (Spotify reopens on device changes).
        regrab: Box::new(move || {
            if let Ok(ids) = find_streams(&needle) {
                for id in ids {
                    let _ = pactl(&["move-sink-input", &id, SINK]);
                }
            }
        }),
        // Unloading the sink makes PipeWire move grabbed streams back to the default.
        cleanup: Box::new(move || {
            if let Some(m) = &module {
                let _ = pactl(&["unload-module", m]);
            }
        }),
        route: format!("{} `{grab}` stream(s) -> {SINK}", grabbed.len()),
        output,
        hint,
    })
}
