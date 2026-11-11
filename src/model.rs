//! Source separators. Everything downstream only sees "fixed-size stereo window in,
//! bass for that window out", so other models (BS-RoFormer, SCNet, ...) can slot in
//! as further `Separator` impls.

use crate::spec::{frames_for, Stft, BINS};
use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use realfft::num_complex::Complex32;
use std::path::Path;

pub const SAMPLE_RATE: u32 = 44_100;

pub trait Separator {
    /// Window length in samples (per channel) the model is run on.
    fn window_len(&self) -> usize;
    /// `mix` is planar stereo `[L.. , R..]` of `window_len()` each; returns the
    /// bass stem in the same layout.
    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>>;
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

impl HtDemucs {
    pub fn load(path: &Path, device: Device, threads: usize) -> Result<Self> {
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
                    .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("built without the `cuda` feature; rebuild with `--features cuda`");
        }
        let session = b
            .commit_from_file(path)
            .with_context(|| format!("loading model {}", path.display()))?;
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
