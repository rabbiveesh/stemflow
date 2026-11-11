mod audio;
mod model;
mod spec;
mod stream;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use model::{Device, HtDemucs, Separator, HTDEMUCS_SEGMENT, SAMPLE_RATE};
use std::path::{Path, PathBuf};
use std::time::Instant;
use stream::{Mix, Streamer};

/// Live bass isolation for whatever your desktop is playing.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Route an app (default: Spotify) through stemflow and play the bass stem.
    Live {
        #[command(flatten)]
        opts: Opts,
        /// Move streams whose application name contains this (case-insensitive).
        #[arg(long, default_value = "spotify")]
        grab: String,
        /// Output sink (default: the current default sink).
        #[arg(long)]
        output: Option<String>,
        /// Extra playback buffer in seconds to absorb compute jitter.
        #[arg(long, default_value_t = 0.3)]
        slack: f32,
    },
    /// Run the streaming pipeline over a WAV file (44.1 kHz) and write the result.
    File {
        #[command(flatten)]
        opts: Opts,
        input: PathBuf,
        output: PathBuf,
    },
    /// Check the Rust STFT + ONNX pipeline against PyTorch reference vectors.
    Verify {
        #[arg(long, default_value = "models/htdemucs_ft_bass.onnx")]
        model: PathBuf,
        #[arg(long, default_value = "models/ref")]
        reference: PathBuf,
        #[arg(long, value_enum, default_value_t = Device::Cpu)]
        device: Device,
    },
}

#[derive(Args, Clone)]
struct Opts {
    #[arg(long, default_value = "models/htdemucs_ft_bass.onnx")]
    model: PathBuf,
    #[arg(long, value_enum, default_value_t = default_device())]
    device: Device,
    /// Seconds of new audio per model run. Compute time per run must stay below this.
    #[arg(long, default_value_t = 1.0)]
    hop: f32,
    /// Seconds of future context each output sample gets. More = cleaner, laggier.
    #[arg(long, default_value_t = 1.5)]
    lookahead: f32,
    /// Crossfade between consecutive runs, seconds (must be <= lookahead).
    #[arg(long, default_value_t = 0.1)]
    xfade: f32,
    #[arg(long, default_value_t = 1.0)]
    bass_gain: f32,
    /// Level of everything except the bass (0 = bass solo, 1 = original mix).
    #[arg(long, default_value_t = 0.0)]
    rest_gain: f32,
    /// CPU threads for inference (0 = ONNX Runtime default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

fn default_device() -> Device {
    if cfg!(feature = "cuda") {
        Device::Cuda
    } else {
        Device::Cpu
    }
}

fn secs(s: f32) -> usize {
    (s * SAMPLE_RATE as f32).round() as usize
}

fn streamer(o: &Opts) -> Result<Streamer<HtDemucs>> {
    let sep = HtDemucs::load(&o.model, o.device, o.threads)?;
    let (hop, la, xf) = (secs(o.hop), secs(o.lookahead), secs(o.xfade));
    if xf > la {
        bail!("--xfade must be <= --lookahead");
    }
    if hop + la > HTDEMUCS_SEGMENT {
        bail!(
            "--hop + --lookahead must be <= {:.2}s",
            HTDEMUCS_SEGMENT as f32 / SAMPLE_RATE as f32
        );
    }
    Ok(Streamer::new(
        sep,
        hop,
        la,
        xf,
        Mix {
            bass_gain: o.bass_gain,
            rest_gain: o.rest_gain,
        },
    ))
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Live {
            opts,
            grab,
            output,
            slack,
        } => audio::live(streamer(&opts)?, &grab, output, slack),
        Cmd::File {
            opts,
            input,
            output,
        } => file(&opts, &input, &output),
        Cmd::Verify {
            model,
            reference,
            device,
        } => verify(&model, &reference, device),
    }
}

fn file(o: &Opts, input: &Path, output: &Path) -> Result<()> {
    let mut r =
        hound::WavReader::open(input).with_context(|| format!("opening {}", input.display()))?;
    let spec = r.spec();
    if spec.sample_rate != SAMPLE_RATE {
        bail!(
            "{} is {} Hz; resample to 44100 first (e.g. ffmpeg -ar 44100)",
            input.display(),
            spec.sample_rate
        );
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
    let stereo: Vec<f32> = match spec.channels {
        1 => raw.iter().flat_map(|&s| [s, s]).collect(),
        2 => raw,
        n => bail!("{n} channels not supported"),
    };

    let mut s = streamer(o)?;
    let (hop, lat) = (s.hop, s.delay());
    let frames = stereo.len() / 2;
    let mut padded = stereo;
    padded.resize(2 * (frames + lat).div_ceil(hop) * hop, 0.0);

    let mut out = Vec::with_capacity(padded.len());
    let mut times = Vec::new();
    for chunk in padded.chunks(2 * hop) {
        let t = Instant::now();
        out.extend(s.process(chunk)?);
        times.push(t.elapsed().as_secs_f32());
    }
    let hop_s = hop as f32 / SAMPLE_RATE as f32;
    times.sort_by(f32::total_cmp);
    let pct = |p: f32| times[((times.len() - 1) as f32 * p) as usize];
    let late = times.iter().filter(|&&t| t > hop_s).count();
    eprintln!(
        "{} runs per {hop_s:.2}s hop: p50 {:.3}s  p99 {:.3}s  max {:.3}s  -> {late} run(s) over budget{}",
        times.len(),
        pct(0.5),
        pct(0.99),
        pct(1.0),
        if late == 0 { " (keeps up live)" } else { "" }
    );

    let mut w = hound::WavWriter::create(
        output,
        hound::WavSpec {
            channels: 2,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    // Drop the pipeline latency so the output lines up with the input.
    for &v in &out[2 * lat..2 * (lat + frames)] {
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn verify(model: &Path, reference: &Path, device: Device) -> Result<()> {
    let mix = read_f32(&reference.join("mix.f32"))?;
    let want_mag = read_f32(&reference.join("mag.f32"))?;
    let want = read_f32(&reference.join("bass.f32"))?;
    let mut m = HtDemucs::load(model, device, 0)?;

    let mag = m.mag(&mix);
    let mag_err = mag
        .iter()
        .zip(&want_mag)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let t = Instant::now();
    let got = m.separate(&mix)?;
    let dt = t.elapsed();
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    println!("STFT  max abs err vs torch: {mag_err:.2e}");
    println!("bass  max abs err vs torch: {err:.2e} (signal peak {peak:.3})");
    println!(
        "one 7.8s window took {:.3}s on {device:?}",
        dt.as_secs_f32()
    );
    if mag_err > 1e-3 || err > 1e-3 {
        bail!("mismatch");
    }
    println!("OK");
    Ok(())
}
