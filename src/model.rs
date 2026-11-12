//! Source separators. Everything downstream only sees "fixed-size stereo window in,
//! bass for that window out", so other models (BS-RoFormer, SCNet, ...) can slot in
//! as further `Separator` impls.

use crate::spec::{frames_for, HannStft, PlainStft, Stft, BINS, FULL_BINS, HOP};
use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use realfft::num_complex::Complex32;
use std::path::Path;

pub const SAMPLE_RATE: u32 = 44_100;

/// Optional per-model GPU memory cap in MiB ($STEMFLOW_GPU_MEM_MB; 0 = no cap).
/// Measured needs in fp32: htdemucs_ft bass ~3 GB, SCNet XL ~5+ GB, so the
/// ensemble doesn't fit on an 8 GB card yet (fp16 SCNet would fix that).
fn gpu_mem_mb() -> usize {
    std::env::var("STEMFLOW_GPU_MEM_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

impl<S: Separator + ?Sized> Separator for Box<S> {
    fn window_len(&self) -> usize {
        (**self).window_len()
    }
    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        (**self).separate(mix)
    }
    fn valid_len(&self) -> usize {
        (**self).valid_len()
    }
}

pub trait Separator {
    /// Window length in samples (per channel) the model is run on.
    fn window_len(&self) -> usize;
    /// `mix` is planar stereo `[L.. , R..]` of `window_len()` each; returns the
    /// bass stem in the same layout.
    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>>;
    /// How many trailing samples of the output are real estimates (an ensemble of
    /// models with different windows only fully covers the shortest one).
    fn valid_len(&self) -> usize {
        self.window_len()
    }
}

/// htdemucs_ft bass specialist exported by tools/export_htdemucs.py.
pub struct HtDemucs {
    session: Session,
    stft: Stft,
    seg: usize,
    frames: usize,
}

pub const HTDEMUCS_SEGMENT: usize = 343_980; // int(7.8 s * 44100)

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    Cpu,
    Cuda,
}

/// Load libonnxruntime: $ORT_DYLIB_PATH, else vendor/ort-{cpu,cuda} (tools/fetch-ort.sh).
fn runtime(device: Device) -> Result<&'static ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    // The CUDA provider finds libcudart/cublas/cudnn via the loader search path, which
    // is fixed at process start; so for CUDA, re-exec once with vendor/ort-cuda/lib on it.
    if device == Device::Cuda
        && std::env::var_os("ORT_DYLIB_PATH").is_none()
        && std::env::var_os("STEMFLOW_REEXEC").is_none()
    {
        use std::os::unix::process::CommandExt;
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/ort-cuda/lib");
        let mut path = std::ffi::OsString::from(&dir);
        if let Some(old) = std::env::var_os("LD_LIBRARY_PATH") {
            path.push(":");
            path.push(old);
        }
        let err = std::process::Command::new(std::env::current_exe()?)
            .args(std::env::args_os().skip(1))
            .env("LD_LIBRARY_PATH", path)
            .env("STEMFLOW_REEXEC", "1")
            .exec();
        anyhow::bail!("re-exec with CUDA libs failed: {err}");
    }
    let lib = std::env::var_os("ORT_DYLIB_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let flavor = if device == Device::Cuda {
                "ort-cuda"
            } else {
                "ort-cpu"
            };
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("vendor")
                .join(flavor)
                .join("lib/libonnxruntime.so")
        });
    let env = ort::init_from(&lib)
        .map_err(|e| anyhow::anyhow!("loading {}: {e} (run tools/fetch-ort.sh)", lib.display()))?
        .with_name("stemflow")
        .build()?;
    Ok(ENV.get_or_init(|| env))
}

fn session(path: &Path, device: Device, threads: usize, gpu_mem_mb: usize) -> Result<Session> {
    let _ = gpu_mem_mb; // unused without the cuda feature
    let env = runtime(device)?;
    let mut b = Session::builder(env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if device == Device::Cuda {
        #[cfg(feature = "cuda")]
        {
            b = b
                .with_execution_providers([ort::ep::CUDA::default()
                    // Optional cap on this model's GPU pool (0 = no limit).
                    .with_memory_limit(gpu_mem_mb * 1024 * 1024)
                    // Exhaustive cuDNN search + max workspace balloons VRAM (5-7 GB per
                    // model); heuristic picks are near-identical in speed here.
                    .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                    .with_conv_max_workspace(false)
                    .build()
                    .error_on_failure()])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        #[cfg(not(feature = "cuda"))]
        anyhow::bail!("built without the `cuda` feature; rebuild with `--features cuda`");
    }
    b.commit_from_file(path).with_context(|| {
        format!(
            "loading model {} (see README: model export)",
            path.display()
        )
    })
}

impl HtDemucs {
    pub fn load(path: &Path, device: Device, threads: usize) -> Result<Self> {
        let session = session(path, device, threads, gpu_mem_mb())?;
        let seg = HTDEMUCS_SEGMENT;
        let mut m = HtDemucs {
            session,
            stft: Stft::new(),
            seg,
            frames: frames_for(seg),
        };
        // First run pays for cuDNN autotuning / allocations; do it before audio starts.
        m.separate(&vec![0.0; 2 * seg])?;
        Ok(m)
    }

    /// The model's input spectrogram ("mag"), `[4, BINS, frames]` = L re, L im, R re, R im.
    pub fn mag(&self, mix: &[f32]) -> Vec<f32> {
        let (seg, fr) = (self.seg, self.frames);
        let plane = BINS * fr;
        let mut mag = vec![0f32; 4 * plane];
        let mut z = vec![Complex32::new(0.0, 0.0); plane];
        for ch in 0..2 {
            self.stft.spec(&mix[ch * seg..(ch + 1) * seg], &mut z);
            let (re, im) = mag[2 * ch * plane..(2 * ch + 2) * plane].split_at_mut(plane);
            for (i, c) in z.iter().enumerate() {
                re[i] = c.re;
                im[i] = c.im;
            }
        }
        mag
    }
}

impl Separator for HtDemucs {
    fn window_len(&self) -> usize {
        self.seg
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let (seg, fr) = (self.seg, self.frames);
        assert_eq!(mix.len(), 2 * seg);
        let mag = self.mag(mix);
        let outputs = self.session.run(ort::inputs![
            "mix" => Tensor::from_array(([1usize, 2, seg], mix.to_vec()))?,
            "mag" => Tensor::from_array(([1usize, 4, BINS, fr], mag))?,
        ])?;
        let (_, spec) = outputs["spec"].try_extract_tensor::<f32>()?;
        let (_, wave) = outputs["wave"].try_extract_tensor::<f32>()?;

        let plane = BINS * fr;
        let mut out = wave.to_vec(); // time branch, [2, seg]
        let mut z = vec![Complex32::new(0.0, 0.0); plane];
        let mut y = vec![0f32; seg];
        for ch in 0..2 {
            let re = &spec[2 * ch * plane..(2 * ch + 1) * plane];
            let im = &spec[(2 * ch + 1) * plane..(2 * ch + 2) * plane];
            for i in 0..plane {
                z[i] = Complex32::new(re[i], im[i]);
            }
            self.stft.ispec(&z, seg, &mut y);
            for (o, v) in out[ch * seg..(ch + 1) * seg].iter_mut().zip(&y) {
                *o += v;
            }
        }
        Ok(out)
    }
}

/// SCNet XL IHF bass, exported by tools/export_scnet.py. Its STFT is plain torch.stft
/// (rectangular window) on the chunk zero-padded so the frame count is even.
pub struct ScNet {
    session: Session,
    stft: PlainStft,
    seg: usize,
    padding: usize,
}

pub const SCNET_SEGMENT: usize = 485_100; // 11 s

impl ScNet {
    pub fn load(path: &Path, device: Device, threads: usize) -> Result<Self> {
        let seg = SCNET_SEGMENT;
        let mut padding = HOP - seg % HOP;
        if ((seg + padding) / HOP).is_multiple_of(2) {
            padding += HOP;
        }
        let mut m = ScNet {
            session: session(path, device, threads, gpu_mem_mb())?,
            stft: PlainStft::new(),
            seg,
            padding,
        };
        m.separate(&vec![0.0; 2 * seg])?;
        Ok(m)
    }

    fn frames(&self) -> usize {
        PlainStft::frames_for(self.seg + self.padding)
    }

    /// Model input: `[4, FULL_BINS, frames]` = L re, L im, R re, R im.
    pub fn spec_in(&self, mix: &[f32]) -> Vec<f32> {
        let (seg, fr) = (self.seg, self.frames());
        let plane = FULL_BINS * fr;
        let mut out = vec![0f32; 4 * plane];
        let mut x = vec![0f32; seg + self.padding];
        let mut z = vec![Complex32::new(0.0, 0.0); plane];
        for ch in 0..2 {
            x[..seg].copy_from_slice(&mix[ch * seg..(ch + 1) * seg]);
            self.stft.stft(&x, &mut z);
            let (re, im) = out[2 * ch * plane..(2 * ch + 2) * plane].split_at_mut(plane);
            for (i, c) in z.iter().enumerate() {
                re[i] = c.re;
                im[i] = c.im;
            }
        }
        out
    }
}

impl Separator for ScNet {
    fn window_len(&self) -> usize {
        self.seg
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let (seg, fr) = (self.seg, self.frames());
        assert_eq!(mix.len(), 2 * seg);
        let spec_in = self.spec_in(mix);
        let outputs = self.session.run(
            ort::inputs!["spec_in" => Tensor::from_array(([1usize, 4, FULL_BINS, fr], spec_in))?],
        )?;
        let (_, spec) = outputs["spec"].try_extract_tensor::<f32>()?;
        let plane = FULL_BINS * fr;
        let mut out = vec![0f32; 2 * seg];
        let mut z = vec![Complex32::new(0.0, 0.0); plane];
        let mut y = vec![0f32; seg + self.padding];
        for ch in 0..2 {
            let re = &spec[2 * ch * plane..(2 * ch + 1) * plane];
            let im = &spec[(2 * ch + 1) * plane..(2 * ch + 2) * plane];
            for i in 0..plane {
                z[i] = Complex32::new(re[i], im[i]);
            }
            self.stft.istft(&z, fr, &mut y);
            out[ch * seg..(ch + 1) * seg].copy_from_slice(&y[..seg]);
        }
        Ok(out)
    }
}

/// BS-RoFormer SW, bass head, exported by tools/export_bsrofo.py. Its STFT
/// (n_fft 2048, hop 512, Hann, unnormalized) runs here; the graph's frame axis is
/// dynamic, so the window length is a free choice (trained at 588800 = 13.35 s).
pub struct BsRoformer {
    session: Session,
    stft: HannStft,
    seg: usize,
}

impl BsRoformer {
    pub fn load(path: &Path, device: Device, threads: usize, seg: usize) -> Result<Self> {
        let mut m = BsRoformer {
            session: session(path, device, threads, gpu_mem_mb())?,
            stft: HannStft::new(2048, 512),
            seg,
        };
        m.separate(&vec![0.0; 2 * seg])?;
        Ok(m)
    }

    fn frames(&self) -> usize {
        self.stft.frames_for(self.seg)
    }

    /// Model input: `[4, 1025, frames]` = L re, L im, R re, R im.
    pub fn spec_in(&self, mix: &[f32]) -> Vec<f32> {
        let (seg, fr, bins) = (self.seg, self.frames(), self.stft.bins());
        let plane = bins * fr;
        let mut out = vec![0f32; 4 * plane];
        let mut z = vec![Complex32::new(0.0, 0.0); plane];
        for ch in 0..2 {
            self.stft.stft(&mix[ch * seg..(ch + 1) * seg], &mut z);
            let (re, im) = out[2 * ch * plane..(2 * ch + 2) * plane].split_at_mut(plane);
            for (i, c) in z.iter().enumerate() {
                re[i] = c.re;
                im[i] = c.im;
            }
        }
        out
    }
}

impl Separator for BsRoformer {
    fn window_len(&self) -> usize {
        self.seg
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let seg = self.seg;
        assert_eq!(mix.len(), 2 * seg);
        let spec_in = self.spec_in(mix);
        self.separate_spec(spec_in)
    }
}

impl BsRoformer {
    /// The model + iSTFT half of `separate`, from a ready-made input spectrogram.
    pub fn separate_spec(&mut self, spec_in: Vec<f32>) -> Result<Vec<f32>> {
        let (seg, fr, bins) = (self.seg, self.frames(), self.stft.bins());
        let outputs = self.session.run(
            ort::inputs!["spec_in" => Tensor::from_array(([1usize, 4, bins, fr], spec_in))?],
        )?;
        let (_, spec) = outputs["spec"].try_extract_tensor::<f32>()?;
        let plane = bins * fr;
        let mut out = vec![0f32; 2 * seg];
        let mut z = vec![Complex32::new(0.0, 0.0); plane];
        for ch in 0..2 {
            let re = &spec[2 * ch * plane..(2 * ch + 1) * plane];
            let im = &spec[(2 * ch + 1) * plane..(2 * ch + 2) * plane];
            for i in 0..plane {
                z[i] = Complex32::new(re[i], im[i]);
            }
            self.stft.istft(&z, fr, &mut out[ch * seg..(ch + 1) * seg]);
        }
        Ok(out)
    }
}

/// Weighted average of several separators. The window is the longest member's; each
/// member sees the trailing part of it that fits its own window, so only the
/// shortest member's span (`valid_len`) has every model's vote.
pub struct Ensemble {
    members: Vec<(Box<dyn Separator + Send>, f32)>,
    seg: usize,
    valid: usize,
}

impl Ensemble {
    pub fn new(members: Vec<(Box<dyn Separator + Send>, f32)>) -> Self {
        let seg = members
            .iter()
            .map(|(m, _)| m.window_len())
            .max()
            .expect("empty ensemble");
        let valid = members.iter().map(|(m, _)| m.valid_len()).min().unwrap();
        Ensemble {
            members,
            seg,
            valid,
        }
    }
}

impl Separator for Ensemble {
    fn window_len(&self) -> usize {
        self.seg
    }

    fn valid_len(&self) -> usize {
        self.valid
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let seg = self.seg;
        let mut acc = vec![0f32; 2 * seg];
        let mut wsum = vec![0f32; seg];
        let mut sub = Vec::new();
        for (m, w) in &mut self.members {
            let n = m.window_len();
            sub.clear();
            for ch in 0..2 {
                sub.extend_from_slice(&mix[ch * seg + seg - n..(ch + 1) * seg]);
            }
            let y = m.separate(&sub)?;
            for ch in 0..2 {
                for i in 0..n {
                    acc[ch * seg + seg - n + i] += *w * y[ch * n + i];
                }
            }
            for v in &mut wsum[seg - n..] {
                *v += *w;
            }
        }
        for ch in 0..2 {
            for i in 0..seg {
                if wsum[i] > 0.0 {
                    acc[ch * seg + i] /= wsum[i];
                }
            }
        }
        Ok(acc)
    }
}

/// Feeds the wrapped separator `(L + R) / 2` on both channels.
///
/// The models were trained on pop mixes with the bass in the centre. On wide 1950s-60s
/// jazz stereo (e.g. the Oscar Peterson Trio on Verve) the bass sits off to one side
/// and htdemucs_ft returned silence for whole clips; folding to mono fixed it
/// (Night Train: 30/30 silent seconds -> 0/30). Centre-panned material loses nothing.
pub struct MonoFold<S>(pub S);

impl<S: Separator> Separator for MonoFold<S> {
    fn window_len(&self) -> usize {
        self.0.window_len()
    }

    fn valid_len(&self) -> usize {
        self.0.valid_len()
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let n = mix.len() / 2;
        let (l, r) = mix.split_at(n);
        let m: Vec<f32> = l.iter().zip(r).map(|(a, b)| 0.5 * (a + b)).collect();
        let mut folded = Vec::with_capacity(2 * n);
        folded.extend_from_slice(&m);
        folded.extend_from_slice(&m);
        self.0.separate(&folded)
    }
}
