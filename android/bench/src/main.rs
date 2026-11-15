//! stemflow-bench: how fast does a separator run on this device?
//!
//! Reuses the desktop's STFT, model and streaming code verbatim (via #[path]), so the
//! numbers are for exactly what the live app would run. A `.onnx` model runs on ONNX
//! Runtime (CPU), a `.tflite` one on LiteRT (CPU, or the GPU with --gpu). Over adb:
//!
//!   export LD_LIBRARY_PATH=. ORT_DYLIB_PATH=./libonnxruntime.so
//!   ./stemflow-bench htdemucs_ft_bass.onnx --wav clip.wav --threads 4
//!   ./stemflow-bench htdemucs_ft_bass_344064.tflite --gpu fp16 --wav clip.wav --out bass.wav

#![allow(dead_code)]

mod litert;
#[path = "../../../src/model.rs"]
mod model;
#[path = "../../../src/spec.rs"]
mod spec;
#[path = "../../../src/stream.rs"]
mod stream;

use anyhow::{bail, Context, Result};
use clap::Parser;
use model::{Device, HtDemucs, MonoFold, Separator, SAMPLE_RATE};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    /// htdemucs_ft-style model, .onnx or .tflite (bass or vocals; fine-tunes too).
    model: PathBuf,
    /// Run a .tflite model on the GPU: `fp32`, or `fp16` (faster, less exact).
    #[arg(long)]
    gpu: Option<String>,
    /// 44.1 kHz WAV to run over (default: 30 s of noise, timing only).
    #[arg(long)]
    wav: Option<PathBuf>,
    /// CPU threads (0 = the runtime's default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Seconds of new audio per model run (desktop live: 1).
    #[arg(long, default_value_t = 1.0)]
    hop: f32,
    /// Seconds of future context per output sample.
    #[arg(long, default_value_t = 1.0)]
    lookahead: f32,
    /// Crossfade between runs, seconds.
    #[arg(long, default_value_t = 0.25)]
    xfade: f32,
    /// Stop after this many model runs (0 = the whole file).
    #[arg(long, default_value_t = 0)]
    runs: usize,
    /// Write the isolated stem here.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn read_wav(p: &Path) -> Result<Vec<f32>> {
    let mut r = hound::WavReader::open(p).with_context(|| format!("opening {}", p.display()))?;
    let spec = r.spec();
    if spec.sample_rate != SAMPLE_RATE {
        bail!("{} is {} Hz; resample to 44100", p.display(), spec.sample_rate);
    }
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(match spec.channels {
        1 => raw.iter().flat_map(|&s| [s, s]).collect(),
        2 => raw,
        n => bail!("{n} channels not supported"),
    })
}

fn peak_rss_mb() -> Option<f32> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f32 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn main() -> Result<()> {
    let a = Args::parse();
    let sr = SAMPLE_RATE as f32;
    let secs = |s: f32| (s * sr).round() as usize;

    let stereo = match &a.wav {
        Some(p) => read_wav(p)?,
        None => {
            // Cheap LCG noise: content doesn't change the compute.
            let mut x = 12345u32;
            (0..2 * secs(30.0))
                .map(|_| {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    (x >> 8) as f32 / (1u32 << 24) as f32 * 0.2 - 0.1
                })
                .collect()
        }
    };

    let t = Instant::now();
    let sep: Box<dyn Separator> = if a.model.extension().is_some_and(|e| e == "tflite") {
        let accel = match a.gpu.as_deref() {
            None => litert::Accel::Cpu,
            Some("fp32") => litert::Accel::Gpu,
            Some("fp16") => litert::Accel::GpuFp16,
            Some(o) => bail!("--gpu {o}: expected fp32 or fp16"),
        };
        // Told apart by file name, as written by the two export scripts.
        if a.model.file_name().is_some_and(|n| n.to_string_lossy().starts_with("bsrofo")) {
            Box::new(litert::LiteRtRoformer::load(&a.model, accel, a.threads)?)
        } else {
            Box::new(litert::LiteRtDemucs::load(&a.model, accel, a.threads)?)
        }
    } else {
        Box::new(HtDemucs::load(&a.model, Device::Cpu, a.threads)?)
    };
    let sep = MonoFold(sep);
    eprintln!(
        "loaded {} + warm-up run in {:.2}s (window {:.2}s)",
        a.model.display(),
        t.elapsed().as_secs_f32(),
        sep.window_len() as f32 / sr
    );

    let (hop, la) = (secs(a.hop), secs(a.lookahead));
    let mut s = stream::Streamer::new(sep, hop, la, secs(a.xfade));
    let frames = stereo.len() / 2;
    let mut padded = stereo;
    padded.resize(2 * (frames + la).div_ceil(hop) * hop, 0.0);

    let mut out = Vec::with_capacity(padded.len());
    let mut times = Vec::new();
    for chunk in padded.chunks(2 * hop) {
        let t = Instant::now();
        out.extend(s.process(chunk)?.bass);
        let dt = t.elapsed().as_secs_f32();
        times.push(dt);
        eprintln!("run {:3}: {:.3}s", times.len(), dt);
        if a.runs > 0 && times.len() >= a.runs {
            break;
        }
    }

    let n = times.len();
    times.sort_by(f32::total_cmp);
    let pct = |p: f32| times[((n - 1) as f32 * p) as usize];
    let late = times.iter().filter(|&&t| t > a.hop).count();
    println!(
        "model={} gpu={} threads={} hop={:.2}s runs={n}: min {:.3}s p50 {:.3}s p90 {:.3}s max {:.3}s; {late} over budget; realtime x{:.2}; peak RSS {:.0} MB",
        a.model.file_name().unwrap_or_default().to_string_lossy(),
        a.gpu.as_deref().unwrap_or("no"),
        a.threads,
        a.hop,
        times[0],
        pct(0.5),
        pct(0.9),
        times[n - 1],
        a.hop / pct(0.5),
        peak_rss_mb().unwrap_or(f32::NAN),
    );

    if let Some(p) = &a.out {
        // Drop the streamer's delay so the stem lines up with the input.
        let d = 2 * s.delay();
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(p, spec)?;
        for &v in out.iter().skip(d).take(2 * frames) {
            w.write_sample(v)?;
        }
        w.finalize()?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}
