mod audio;
mod control;
mod model;
mod models;
mod spec;
mod stream;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use model::{BsRoformer, Device, Ensemble, HtDemucs, MonoFold, ScNet, Separator, SAMPLE_RATE};
use std::path::{Path, PathBuf};
use std::time::Instant;
use stream::Streamer;

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
    /// List the models in the models folder (* = the one live mode last used).
    Models {
        #[arg(long, default_value = "models")]
        models: PathBuf,
    },
    /// Check the Rust STFT + ONNX pipeline against PyTorch reference vectors.
    Verify {
        /// demucs, scnet or bsrofo (an ensemble is verified by verifying its members).
        #[arg(long, value_enum, default_value_t = Sep::Demucs)]
        sep: Sep,
        #[arg(long, default_value = "models")]
        models: PathBuf,
        #[arg(long, value_enum, default_value_t = Device::Cpu)]
        device: Device,
        /// Which stem to verify (bsrofo and demucs).
        #[arg(long, value_enum, default_value_t = Target::Bass)]
        target: Target,
    },
}

/// The stem the mix controls act on. BS-RoFormer SW and htdemucs_ft have vocals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Target {
    Bass,
    /// karaoke: solo / full / remove act on the vocals
    Vocals,
}

impl Target {
    fn name(self) -> &'static str {
        match self {
            Target::Bass => "bass",
            Target::Vocals => "vocals",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Sep {
    /// htdemucs_ft bass specialist (7.8 s window)
    Demucs,
    /// SCNet XL IHF (11 s window)
    Scnet,
    /// average of both
    Ensemble,
    /// BS-RoFormer SW (window set by --window)
    Bsrofo,
}

#[derive(Args, Clone)]
struct Opts {
    /// Architecture (default bsrofo). Live, without --sep/--target/--model, the model
    /// last picked in the menu is used.
    #[arg(long, value_enum)]
    sep: Option<Sep>,
    /// Stem to isolate or remove (default bass). `vocals` (karaoke) needs --sep bsrofo or demucs.
    #[arg(long, value_enum)]
    target: Option<Target>,
    /// A model by id: its path under --models without `.onnx`, e.g. `ft2/htdemucs_ft_bass`
    /// (overrides --sep and --target). `stemflow models` lists them.
    #[arg(long)]
    model: Option<String>,
    /// Directory with the exported .onnx models.
    #[arg(long, default_value = "models")]
    models: PathBuf,
    /// Feed the model the original stereo instead of a mono fold. Mono is the default
    /// because the models miss bass that's panned off-centre (wide 50s-60s stereo).
    #[arg(long)]
    stereo_in: bool,
    /// BS-RoFormer context window, seconds (trained at 13.35). 3 keeps up live on an
    /// 8 GB laptop GPU; longer needs more VRAM and compute but not more latency.
    #[arg(long, default_value_t = 3.0)]
    window: f32,
    /// Ensemble weight of SCNet (Demucs gets 1 - this).
    #[arg(long, default_value_t = 0.5)]
    scnet_weight: f32,
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
    /// Starting mix on one line: -1 = target (bass) solo, 0 = full mix, +1 = no target
    /// (play-along). Default: the last level set (live), else bass solo.
    #[arg(long, allow_hyphen_values = true)]
    balance: Option<f32>,
    /// Target (bass, or vocals in karaoke) loudness in percent of the original (50 sounds
    /// half as loud, up to 200 = twice as loud). Overrides --balance.
    #[arg(long)]
    bass: Option<f32>,
    /// Band (everything but the target) loudness in percent of full. Overrides --balance.
    #[arg(long)]
    band: Option<f32>,
    /// CPU threads for inference (0 = ONNX Runtime default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

impl Opts {
    fn sep(&self) -> Sep {
        self.sep.unwrap_or(Sep::Bsrofo)
    }
    fn target(&self) -> Target {
        self.target.unwrap_or(Target::Bass)
    }
    /// The model id `--sep`/`--target` name, if they name a single file.
    fn sep_model_id(&self) -> Option<String> {
        match self.sep() {
            Sep::Bsrofo => Some(bsrofo_onnx(self.target())),
            Sep::Demucs => Some(demucs_onnx(self.target())),
            Sep::Scnet => Some(SCNET_ONNX.into()),
            Sep::Ensemble => None,
        }
        .map(|f| f.trim_end_matches(".onnx").to_string())
    }
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

fn demucs_onnx(t: Target) -> String {
    format!("htdemucs_ft_{}.onnx", t.name())
}
const SCNET_ONNX: &str = "scnet_xl_ihf_bass.onnx";
fn bsrofo_onnx(t: Target) -> String {
    format!("bsrofo_sw_{}.onnx", t.name())
}

fn separator(o: &Opts) -> Result<Box<dyn Separator + Send>> {
    let demucs = || HtDemucs::load(&o.models.join(demucs_onnx(o.target())), o.device, o.threads);
    let scnet = || ScNet::load(&o.models.join(SCNET_ONNX), o.device, o.threads);
    if o.target() != Target::Bass && !matches!(o.sep(), Sep::Bsrofo | Sep::Demucs) {
        bail!(
            "--target {} needs --sep bsrofo or demucs",
            o.target().name()
        );
    }
    let sep: Box<dyn Separator + Send> = match o.sep() {
        Sep::Demucs => Box::new(demucs()?),
        Sep::Scnet => Box::new(scnet()?),
        Sep::Ensemble => Box::new(Ensemble::new(vec![
            (Box::new(demucs()?), 1.0 - o.scnet_weight),
            (Box::new(scnet()?), o.scnet_weight),
        ])),
        Sep::Bsrofo => Box::new(BsRoformer::load(
            &o.models.join(bsrofo_onnx(o.target())),
            o.device,
            o.threads,
            secs(o.window),
        )?),
    };
    Ok(fold(o, sep))
}

fn fold(o: &Opts, sep: Box<dyn Separator + Send>) -> Box<dyn Separator + Send> {
    if o.stereo_in {
        sep
    } else {
        Box::new(MonoFold(sep))
    }
}

/// Load one model file (from the models folder scan).
fn load_model(o: &Opts, m: &models::Model) -> Result<Box<dyn Separator + Send>> {
    let sep: Box<dyn Separator + Send> = match m.arch {
        models::Arch::Demucs => Box::new(HtDemucs::load(&m.path, o.device, o.threads)?),
        models::Arch::Scnet => Box::new(ScNet::load(&m.path, o.device, o.threads)?),
        models::Arch::Bsrofo => Box::new(BsRoformer::load(
            &m.path,
            o.device,
            o.threads,
            secs(o.window),
        )?),
    };
    Ok(fold(o, sep))
}

fn streamer(o: &Opts) -> Result<Streamer<Box<dyn Separator + Send>>> {
    match &o.model {
        Some(id) => {
            let m = models::find(&o.models, id).with_context(|| {
                format!(
                    "no model `{id}` in {} (see `stemflow models`)",
                    o.models.display()
                )
            })?;
            wrap(o, load_model(o, &m)?)
        }
        None => wrap(o, separator(o)?),
    }
}

/// The model live mode starts with: --model, else --sep/--target if given, else the
/// one last picked in the menu (if it still exists), else the default.
fn live_model(o: &Opts) -> Option<models::Model> {
    let id = o
        .model
        .clone()
        .or_else(|| {
            if o.sep.is_none() && o.target.is_none() {
                control::load_model().filter(|id| models::find(&o.models, id).is_some())
            } else {
                None
            }
        })
        .or_else(|| o.sep_model_id())?;
    models::find(&o.models, &id)
}

fn wrap(o: &Opts, sep: Box<dyn Separator + Send>) -> Result<Streamer<Box<dyn Separator + Send>>> {
    let (hop, la, xf) = (secs(o.hop), secs(o.lookahead), secs(o.xfade));
    if xf > la {
        bail!("--xfade must be <= --lookahead");
    }
    if hop + la > sep.valid_len() {
        bail!(
            "--hop + --lookahead must be <= {:.2}s",
            sep.valid_len() as f32 / SAMPLE_RATE as f32
        );
    }
    Ok(Streamer::new(sep, hop, la, xf))
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Live {
            opts,
            grab,
            output,
            slack,
        } => {
            let m = live_model(&opts);
            let st = match &m {
                Some(m) => wrap(&opts, load_model(&opts, m)?)?,
                None => streamer(&opts)?, // the ensemble
            };
            let target = m.as_ref().map_or(opts.target().name(), |m| m.target);
            let ctl = levels(&opts, target, true);
            ctl.set_model(m.as_ref().map_or("ensemble", |m| &m.id));
            let (lat, hop) = (
                st.latency() as f64 / SAMPLE_RATE as f64,
                st.hop as f64 / SAMPLE_RATE as f64,
            );
            // Off the bus is fine (e.g. no session bus over ssh): CLI flags still work.
            let _bus = control::serve(
                ctl.clone(),
                opts.models.clone(),
                &grab,
                lat + slack as f64,
                hop,
            )
            .map_err(|e| eprintln!("stemflow: no D-Bus control ({e})"))
            .ok();
            // Switching models from the menu: drop the running one (frees the GPU),
            // then load the new one with the same settings.
            let load =
                move |id: &str| -> Result<(Streamer<Box<dyn Separator + Send>>, &'static str)> {
                    let m = models::find(&opts.models, id)
                        .with_context(|| format!("no model `{id}`"))?;
                    Ok((wrap(&opts, load_model(&opts, &m)?)?, m.target))
                };
            audio::live(st, ctl, &grab, output, slack, load)
        }
        Cmd::Models { models } => {
            let cur = control::load_model();
            for m in models::scan(&models) {
                let mark = if cur.as_deref() == Some(m.id.as_str()) {
                    "*"
                } else {
                    " "
                };
                println!("{mark} {:32} {}", m.id, m.label);
            }
            Ok(())
        }
        Cmd::File {
            opts,
            input,
            output,
        } => file(&opts, &input, &output),
        Cmd::Verify {
            sep,
            models,
            device,
            target,
        } => match sep {
            Sep::Demucs => verify_demucs(&models, device, target),
            Sep::Scnet => verify_scnet(&models, device),
            Sep::Ensemble => {
                verify_demucs(&models, device, Target::Bass).and(verify_scnet(&models, device))
            }
            Sep::Bsrofo => verify_bsrofo(&models, device, target),
        },
    }
}

/// Starting levels: explicit flags, else the remembered ones, else bass solo.
fn levels(o: &Opts, target: &'static str, remember: bool) -> std::sync::Arc<control::Control> {
    let ctl = control::Control::new(target, 1.0, 0.0);
    if let Some((a, b)) = remember.then(|| control::load_levels(target)).flatten() {
        ctl.set_levels(a, b);
    }
    if let Some(b) = o.balance {
        ctl.set_balance(b);
    }
    let (a, b) = ctl.levels();
    ctl.set_levels(
        o.bass.map_or(a, |v| v / 100.0),
        o.band.map_or(b, |v| v / 100.0),
    );
    ctl
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
    let ctl = levels(o, o.target().name(), false);
    let (hop, lat) = (s.hop, s.delay());
    let frames = stereo.len() / 2;
    let mut padded = stereo;
    padded.resize(2 * (frames + lat).div_ceil(hop) * hop, 0.0);

    let mut out = Vec::with_capacity(padded.len());
    let mut times = Vec::new();
    for chunk in padded.chunks(2 * hop) {
        let t = Instant::now();
        let (bg, rg) = ctl.gains();
        out.extend(s.process(chunk)?.mix(bg, rg));
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

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    let s: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (s / a.len() as f32).sqrt()
}

fn report(
    name: &str,
    input_err: f32,
    out_err: f32,
    peak: f32,
    dt: f32,
    device: Device,
) -> Result<()> {
    println!("[{name}] input spectrogram max abs err vs torch: {input_err:.2e}");
    println!("[{name}] stem max abs err vs torch: {out_err:.2e} (signal peak {peak:.3})");
    println!("[{name}] one window took {dt:.3}s on {device:?} (after warmup)");
    if input_err > 1e-3 || out_err > 1e-3 {
        bail!("[{name}] mismatch");
    }
    println!("[{name}] OK");
    Ok(())
}

fn verify_demucs(models: &Path, device: Device, target: Target) -> Result<()> {
    let t = target.name();
    let r = models.join(match target {
        Target::Bass => "ref".to_string(),
        _ => format!("ref_demucs_{t}"),
    });
    let (mix, want_mag, want) = (
        read_f32(&r.join("mix.f32"))?,
        read_f32(&r.join("mag.f32"))?,
        read_f32(&r.join(format!("{t}.f32")))?,
    );
    let mut m = HtDemucs::load(&models.join(demucs_onnx(target)), device, 0)?;
    let mag_err = max_err(&m.mag(&mix), &want_mag);
    let t = Instant::now();
    let got = m.separate(&mix)?;
    let dt = t.elapsed().as_secs_f32();
    let peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    report("demucs", mag_err, max_err(&got, &want), peak, dt, device)
}

fn verify_scnet(models: &Path, device: Device) -> Result<()> {
    let r = models.join("ref_scnet");
    let (mix, want_in, want) = (
        read_f32(&r.join("mix.f32"))?,
        read_f32(&r.join("spec_in.f32"))?,
        read_f32(&r.join("bass.f32"))?,
    );
    let mut m = ScNet::load(&models.join(SCNET_ONNX), device, 0)?;
    let in_err = max_err(&m.spec_in(&mix), &want_in);
    let t = Instant::now();
    let got = m.separate(&mix)?;
    let dt = t.elapsed().as_secs_f32();
    let peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    report("scnet", in_err, max_err(&got, &want), peak, dt, device)
}

fn verify_bsrofo(models: &Path, device: Device, target: Target) -> Result<()> {
    let t = target.name();
    let r = models.join(match target {
        Target::Bass => "ref_bsrofo".to_string(),
        _ => format!("ref_bsrofo_{t}"),
    });
    let (mix, want_in, want) = (
        read_f32(&r.join("mix.f32"))?,
        read_f32(&r.join("spec_in.f32"))?,
        read_f32(&r.join(format!("{t}.f32")))?,
    );
    let mut m = BsRoformer::load(&models.join(bsrofo_onnx(target)), device, 0, mix.len() / 2)?;
    let in_err = max_err(&m.spec_in(&mix), &want_in);
    // The model amplifies float32 STFT rounding (1e-5 input jitter moves the output
    // by up to 1e-2), so the strict check feeds torch's own spectrogram: that tests
    // the ONNX graph and the Rust iSTFT exactly. The end-to-end error is reported.
    let t = Instant::now();
    let got = m.separate_spec(want_in)?;
    let dt = t.elapsed().as_secs_f32();
    let peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    let e2e = m.separate(&mix)?;
    let (e, d) = (max_err(&e2e, &want), rms_diff(&e2e, &want));
    println!("[bsrofo] end to end (Rust STFT): max err {e:.2e}, rms {d:.2e}");
    report("bsrofo", in_err, max_err(&got, &want), peak, dt, device)
}
