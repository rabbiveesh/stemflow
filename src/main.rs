mod audio;
mod beats;
mod control;
#[cfg(target_os = "macos")]
mod coreml;
mod looper;
mod model;
mod models;
mod socket;
mod spec;
mod stream;
mod stretch;

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
    /// Check the live route without a model: capture the app (taking it off the
    /// speakers) and play it straight back, printing its level every second.
    Passthrough {
        /// Capture apps whose name contains this (case-insensitive).
        #[arg(long, default_value = "spotify")]
        grab: String,
        /// Output (default: the current default output).
        #[arg(long)]
        output: Option<String>,
        /// Stop after this many seconds (default: run until Ctrl-C).
        #[arg(long)]
        seconds: Option<f32>,
    },
    /// Run the streaming pipeline over a WAV file (44.1 kHz) and write the result.
    File {
        #[command(flatten)]
        opts: Opts,
        input: PathBuf,
        output: PathBuf,
        /// Mix in the beat click as live mode would (analysis every 3 s of input, from
        /// the audio that has arrived so far). Prints the beats it clicked.
        #[arg(long)]
        click: bool,
        /// Test looping: at this second, ask for the last --bars bars, then write the
        /// loop three times (with the click) instead of the rest of the file.
        #[arg(long)]
        loop_at: Option<f32>,
        #[arg(long, default_value_t = 4)]
        bars: usize,
        /// With --loop-at: play the loop at this speed (0.5 = half)
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        /// With --loop-at: transpose by this many semitones
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        semis: f32,
    },
    /// Beat-track the last minute of a WAV file (44.1 kHz stereo) as the looper does,
    /// and print what the decoder saw: tempo, meter scores, bar lines.
    Beats {
        #[command(flatten)]
        opts: Opts,
        input: PathBuf,
        /// Beats per bar (0 = choose)
        #[arg(long, default_value_t = 0)]
        meter: usize,
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

/// The stem the mix controls act on. BS-RoFormer SW has a head for each; htdemucs_ft
/// has bass, vocals, other and drums.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Target {
    Bass,
    /// karaoke: solo / full / remove act on the vocals
    Vocals,
    /// what's left without the other stems: in a jazz combo, the horns
    Other,
    Piano,
    Guitar,
    Drums,
}

impl Target {
    fn name(self) -> &'static str {
        match self {
            Target::Bass => "bass",
            Target::Vocals => "vocals",
            Target::Other => "other",
            Target::Piano => "piano",
            Target::Guitar => "guitar",
            Target::Drums => "drums",
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
    /// Architecture (default bsrofo; on macOS demucs, unless BS-RoFormer has a Core ML
    /// package for the window: tools/export_bsrofo_coreml.py).
    /// Live, without --sep/--target/--model, the model last picked in the menu is used.
    #[arg(long, value_enum)]
    sep: Option<Sep>,
    /// Stem to isolate or remove (default bass). Anything but bass needs --sep bsrofo or
    /// demucs, and that stem's model file.
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
    /// 8 GB laptop GPU; longer needs more VRAM and compute but not more latency. With
    /// Core ML, each window length needs its own package (tools/export_bsrofo_coreml.py).
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
    /// With a two-stem model (bass+vocals): how much of the second stem stays in the
    /// band, in percent (0 = take it out).
    #[arg(long)]
    second: Option<f32>,
    /// CPU threads for inference (0 = ONNX Runtime default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Beat tracker for the beat click (a Beat This! export, tools/export_beat_this.py),
    /// relative to --models. Runs every few seconds while the click is on, on --device.
    #[arg(long, default_value = "beats/beat_this_final0_1500.onnx")]
    beats: PathBuf,
    /// CPU threads for the beat tracker (when it runs on the CPU).
    #[arg(long, default_value_t = 4)]
    beat_threads: usize,
}

impl Opts {
    fn sep(&self) -> Sep {
        // On macOS BS-RoFormer only keeps up with a Core ML package for its window.
        self.sep
            .unwrap_or(if cfg!(target_os = "macos") && !self.bsrofo_coreml() {
                Sep::Demucs
            } else {
                Sep::Bsrofo
            })
    }
    fn bsrofo_coreml(&self) -> bool {
        let onnx = self.models.join(bsrofo_onnx(self.target()));
        self.device == Device::Coreml
            && model::coreml_package(&onnx, &format!(".t{}", 1 + secs(self.window) / 512)).is_some()
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
    if cfg!(target_os = "macos") {
        Device::Coreml
    } else if cfg!(feature = "cuda") {
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
            let second = m.as_ref().and_then(|m| m.second);
            eprintln!(
                "stemflow: model {} on {:?}",
                m.as_ref().map_or("ensemble", |m| &m.id),
                opts.device
            );
            let ctl = levels(&opts, target, second, true);
            ctl.set_model(m.as_ref().map_or("ensemble", |m| &m.id));
            control::load_beat_options(&ctl);
            // The separator's runtime is loaded by now; the tracker shares it.
            let bpath = opts.models.join(&opts.beats);
            let tracker = if bpath.exists() {
                match model::runtime(opts.device).and_then(|env| {
                    beats::BeatNet::load(
                        env,
                        &bpath,
                        opts.beat_threads,
                        opts.device == Device::Cuda,
                    )
                }) {
                    Ok(net) => Some(std::sync::Arc::new(beats::Tracker::new(Box::new(net)))),
                    Err(e) => {
                        eprintln!("stemflow: no beat click ({e:#})");
                        None
                    }
                }
            } else {
                eprintln!("stemflow: no beat click ({} not found)", bpath.display());
                None
            };
            let (lat, hop) = (
                st.latency() as f64 / SAMPLE_RATE as f64,
                st.hop as f64 / SAMPLE_RATE as f64,
            );
            // Off the bus is fine (e.g. no session bus over ssh): CLI flags still work.
            #[cfg(target_os = "linux")]
            let _bus = control::serve(
                ctl.clone(),
                opts.models.clone(),
                &grab,
                lat + slack as f64,
                hop,
            )
            .map_err(|e| eprintln!("stemflow: no D-Bus control ({e})"))
            .ok();
            // The same controls on a Unix socket (the macOS menu bar app, scripts).
            match socket::serve(
                ctl.clone(),
                socket::Info {
                    models: opts.models.clone(),
                    app: grab.clone(),
                    latency: lat + slack as f64,
                    hop,
                    beats: tracker.is_some(),
                },
            ) {
                Ok(p) => eprintln!("stemflow: control socket {}", p.display()),
                Err(e) => eprintln!("stemflow: no control socket ({e:#})"),
            }
            // Switching models from the menu: drop the running one (frees the GPU),
            // then load the new one with the same settings.
            let load =
                move |id: &str| -> Result<(audio::Live, &'static str, Option<&'static str>)> {
                    let m = models::find(&opts.models, id)
                        .with_context(|| format!("no model `{id}`"))?;
                    Ok((wrap(&opts, load_model(&opts, &m)?)?, m.target, m.second))
                };
            audio::live(st, ctl, tracker, &grab, output, slack, load)
        }
        Cmd::Passthrough {
            grab,
            output,
            seconds,
        } => audio::passthrough(&grab, output, seconds),
        Cmd::Beats { opts, input, meter } => {
            let r = hound::WavReader::open(&input)?;
            let spec = r.spec();
            anyhow::ensure!(
                spec.sample_rate == SAMPLE_RATE && spec.channels == 2,
                "need 44.1 kHz stereo"
            );
            let x: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => r.into_samples::<f32>().collect::<Result<_, _>>()?,
                hound::SampleFormat::Int => {
                    let k = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
                    r.into_samples::<i32>()
                        .map(|v| v.map(|v| v as f32 * k))
                        .collect::<Result<_, _>>()?
                }
            };
            let net = beats::BeatNet::load(
                model::runtime(opts.device)?,
                &opts.models.join(&opts.beats),
                opts.beat_threads,
                opts.device == Device::Cuda,
            )?;
            let lp = looper::Looper::new(Box::new(net));
            let zeros = vec![0f32; 2 * 4096];
            for c in x.chunks(2 * 4096) {
                lp.push(c, &zeros[..c.len()], &[]);
            }
            let s = beats::Settings {
                meter,
                ..Default::default()
            };
            lp.find(lp.total(), 4, s)?;
            println!("{}", lp.explain());
            Ok(())
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
            click,
            loop_at,
            bars,
            speed,
            semis,
        } => match loop_at {
            Some(t) => loop_test(&opts, &input, &output, t, bars, speed, semis),
            None => file(&opts, &input, &output, click),
        },
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
fn levels(
    o: &Opts,
    target: &'static str,
    second: Option<&'static str>,
    remember: bool,
) -> std::sync::Arc<control::Control> {
    let ctl = control::Control::new(target, 1.0, 0.0);
    ctl.set_second(second);
    if let Some((a, b, c)) = remember
        .then(|| control::load_levels(&ctl.levels_key()).or_else(|| control::load_levels(target)))
        .flatten()
    {
        ctl.set_levels(a, b);
        ctl.set_second_level(c.unwrap_or(1.0));
    }
    if let Some(v) = o.second {
        ctl.set_second_level(v / 100.0);
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

/// `stemflow file --loop-at`: the looper on a file, to check bar finding and the wrap.
fn loop_test(
    o: &Opts,
    input: &Path,
    output: &Path,
    at: f32,
    bars: usize,
    speed: f32,
    semis: f32,
) -> Result<()> {
    let r = hound::WavReader::open(input)?;
    let spec = r.spec();
    anyhow::ensure!(
        spec.sample_rate == SAMPLE_RATE && spec.channels == 2,
        "need 44.1 kHz stereo"
    );
    let x: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.into_samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let k = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.into_samples::<i32>()
                .map(|v| v.map(|v| v as f32 * k))
                .collect::<Result<_, _>>()?
        }
    };
    let mut s = streamer(o)?;
    let net = beats::BeatNet::load(
        model::runtime(o.device)?,
        &o.models.join(&o.beats),
        o.beat_threads,
        o.device == Device::Cuda,
    )?;
    let lp = looper::Looper::new(Box::new(net));
    let (hop, lat) = (s.hop, s.delay());
    let press = (at * SAMPLE_RATE as f32) as u64 + lat as u64; // stem frames trail input by lat
    let mut played = 0u64;
    for chunk in x.chunks_exact(2 * hop) {
        let st = s.process(chunk)?;
        lp.push(
            &st.bass,
            &st.band,
            st.extra.first().map_or(&[][..], |v| &v[..]),
        );
        played += hop as u64;
        if played >= press + 2 * SAMPLE_RATE as u64 {
            break;
        }
    }
    let t = Instant::now();
    lp.find(press, bars, beats::Settings::default())?;
    let v = lp.view(10).unwrap();
    let secs = |f: f64| (f - lat as f64) / SAMPLE_RATE as f64; // back to input time
    eprintln!(
        "found in {:.2}s: {:.1} bpm, {}/bar, {} bars, loop {:.3}s..{:.3}s ({:.3}s long), press at {at:.2}s",
        t.elapsed().as_secs_f32(), v.bpm, v.meter, v.bars, secs(v.start), secs(v.end), (v.end - v.start) / SAMPLE_RATE as f64
    );
    // A bookmark's way back: keep the grid, lay it again without the network.
    let (g, a, b, one, m, bpm) = lp.grid_around(2).unwrap();
    let times = lp.times();
    let t = Instant::now();
    lp.lay(g, a, b, one, m, bpm, 1.0)?;
    eprintln!(
        "laid again in {:.3}s: times {}",
        t.elapsed().as_secs_f32(),
        if lp.times() == times {
            "match"
        } else {
            "DIFFER"
        }
    );
    let t = Instant::now();
    lp.redecode(m, 1.0);
    let w = lp.view(10).unwrap();
    eprintln!(
        "lazy redecode in {:.2}s: loop {:.3}s..{:.3}s",
        t.elapsed().as_secs_f32(),
        secs(w.start),
        secs(w.end)
    );
    // Next phrase: not heard yet, so no move; then play on (as the source does
    // after a loop is made) and move.
    let (t0, t1) = lp.times().map(|t| (t.0, t.1)).unwrap();
    eprintln!(
        "next phrase before hearing it: moved = {}",
        lp.next_phrase()?
    );
    let more = (t1 + (t1 - t0)) as u64 + SAMPLE_RATE as u64;
    for chunk in x[2 * played as usize..].chunks_exact(2 * hop) {
        if played >= more {
            break;
        }
        let st = s.process(chunk)?;
        lp.push(
            &st.bass,
            &st.band,
            st.extra.first().map_or(&[][..], |v| &v[..]),
        );
        played += hop as u64;
    }
    let t = Instant::now();
    let moved = lp.next_phrase()?;
    let (n0, n1) = lp.times().map(|t| (t.0, t.1)).unwrap();
    eprintln!(
        "next phrase in {:.2}s: moved = {moved}, {:.3}s..{:.3}s (was {:.3}s..{:.3}s; starts {:+.0} ms from the old end, length {:+.0} ms)",
        t.elapsed().as_secs_f32(), secs(n0), secs(n1), secs(t0), secs(t1),
        (n0 - t1) * 1000.0 / SAMPLE_RATE as f64,
        ((n1 - n0) - (t1 - t0)) * 1000.0 / SAMPLE_RATE as f64
    );
    // Refresh: more audio arrives, the copy and beats are re-taken with the loop
    // kept; then a bar more at the end needs no analysis.
    let (r0, r1, _, m, _, _) = lp.times().unwrap();
    let more = played + 3 * SAMPLE_RATE as u64;
    for chunk in x[2 * played as usize..].chunks_exact(2 * hop) {
        if played >= more {
            break;
        }
        let st = s.process(chunk)?;
        lp.push(
            &st.bass,
            &st.band,
            st.extra.first().map_or(&[][..], |v| &v[..]),
        );
        played += hop as u64;
    }
    let t = Instant::now();
    let refreshed = lp.refresh()?;
    let (q0, q1, _, _, _, _) = lp.times().unwrap();
    let t2 = Instant::now();
    let grew = lp.nudge(0, m as i64);
    let (_, e1, _, _, _, _) = lp.times().unwrap();
    eprintln!(
        "refresh in {:.2}s ({refreshed}): loop moved {:+.0}/{:+.0} ms; a bar more in {:.1} ms ({grew}): {:.3}s..{:.3}s",
        t.elapsed().as_secs_f32(),
        (q0 - r0) * 1000.0 / SAMPLE_RATE as f64,
        (q1 - r1) * 1000.0 / SAMPLE_RATE as f64,
        t2.elapsed().as_secs_f64() * 1000.0,
        secs(q0),
        secs(e1)
    );
    // Re-stem with the same model: should give back nearly the same stems.
    let before = lp.clip_mix(v.start as i64, 44100);
    let (b0, _) = {
        let (mut b, mut r, mut c) = (
            vec![0f32; 2 * 44100],
            vec![0f32; 2 * 44100],
            vec![0f32; 44100],
        );
        lp.read(&mut b, &mut r, &mut [], &mut c, 44100, 0);
        (b, r)
    };
    let t = Instant::now();
    looper::restem(&lp, &mut s, true)?;
    let (b1, _) = {
        let (mut b, mut r, mut c) = (
            vec![0f32; 2 * 44100],
            vec![0f32; 2 * 44100],
            vec![0f32; 44100],
        );
        lp.read(&mut b, &mut r, &mut [], &mut c, 44100, 0);
        (b, r)
    };
    let after = lp.clip_mix(v.start as i64, 44100);
    let e = |a: &[f32], b: &[f32]| {
        let n: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let d: f32 = a.iter().map(|x| x * x).sum();
        10.0 * (d / n.max(1e-12)).log10()
    };
    eprintln!(
        "re-stem in {:.2}s: target vs before {:.1} dB SNR, mix kept {:.1} dB",
        t.elapsed().as_secs_f32(),
        e(&b0, &b1),
        e(&before, &after)
    );
    lp.set_click(true);
    lp.set_speed(speed, semis, true);
    // three times round the loop at the practice speed
    let len = ((v.end - v.start) / speed as f64).round() as usize;
    let t_out = Instant::now();
    let (mut b, mut r, mut c) = (vec![0f32; 2 * 1024], vec![0f32; 2 * 1024], vec![0f32; 1024]);
    let mut w = hound::WavWriter::create(
        output,
        hound::WavSpec {
            channels: 2,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    let mut n = 0;
    while n < 3 * len {
        let k = 1024.min(3 * len - n);
        lp.read(&mut b, &mut r, &mut [], &mut c, k, k);
        for i in 0..k {
            for ch in 0..2 {
                w.write_sample(b[2 * i + ch] + r[2 * i + ch] + 0.35 * c[i])?;
            }
        }
        n += k;
    }
    w.finalize()?;
    eprintln!(
        "wrote {:.2}s at {speed}x {semis:+} st in {:.3}s of compute",
        3.0 * len as f64 / SAMPLE_RATE as f64,
        t_out.elapsed().as_secs_f32()
    );
    Ok(())
}

fn file(o: &Opts, input: &Path, output: &Path, click: bool) -> Result<()> {
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
    let second = o
        .model
        .as_ref()
        .and_then(|id| models::find(&o.models, id))
        .and_then(|m| m.second);
    let ctl = levels(o, o.target().name(), second, false);
    let (hop, lat) = (s.hop, s.delay());
    let frames = stereo.len() / 2;
    let mut padded = stereo;
    padded.resize(2 * (frames + lat).div_ceil(hop) * hop, 0.0);

    let tracker = if click {
        let net = beats::BeatNet::load(
            model::runtime(o.device)?,
            &o.models.join(&o.beats),
            o.beat_threads,
            o.device == Device::Cuda,
        )?;
        Some(beats::Tracker::new(Box::new(net)))
    } else {
        None
    };
    let mut out = Vec::with_capacity(padded.len());
    let mut times = Vec::new();
    let mut since = 0;
    for chunk in padded.chunks(2 * hop) {
        let t = Instant::now();
        let (bg, rg, sg) = ctl.gains3();
        let mut m = s.process(chunk)?.mix(bg, rg, sg);
        times.push(t.elapsed().as_secs_f32());
        if let Some(tr) = &tracker {
            // As live: the output of this hop trails the input by the delay.
            let start = (out.len() / 2) as i64 - lat as i64;
            tr.push(chunk);
            since += hop;
            if since >= 3 * SAMPLE_RATE as usize {
                since = 0;
                if tr.analyze()? {
                    let i = tr.info();
                    eprintln!(
                        "beats at {:.1}s: {:.1} bpm, {}/bar, drift {:.0} ms",
                        (start + hop as i64) as f32 / SAMPLE_RATE as f32
                            + lat as f32 / SAMPLE_RATE as f32,
                        i.bpm,
                        i.meter,
                        i.drift_s * 1e3
                    );
                }
            }
            tr.click(&mut m, start, 0.35);
        }
        out.extend(m);
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
