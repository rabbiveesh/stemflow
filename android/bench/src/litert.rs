//! htdemucs on LiteRT (TFLite), optionally on the GPU delegate, through the C API in
//! Google's litert / litert-gpu AARs. Runs models from ../tools/export_litert.py:
//! inputs `mix` [1, 2, seg] and `mag` [1, 4, BINS, frames], outputs the stem's
//! spectrogram [1, 4, BINS, frames] and time-branch waveform [1, 2, seg]. The STFT,
//! iSTFT and final sum are the desktop's (spec.rs), as in model.rs' HtDemucs.

use crate::model::Separator;
use crate::spec::{frames_for, Stft, BINS};
use anyhow::{bail, ensure, Context, Result};
use realfft::num_complex::Complex32;
use std::ffi::{c_char, c_void, CString};
use std::path::Path;

/// TfLiteGpuDelegateOptionsV2 (delegates/gpu/delegate_options.h, LiteRT 1.4).
#[repr(C)]
struct GpuOptions {
    is_precision_loss_allowed: i32,
    inference_preference: i32,
    inference_priority1: i32,
    inference_priority2: i32,
    inference_priority3: i32,
    experimental_flags: i64,
    max_delegated_partitions: i32,
    serialization_dir: *const c_char,
    model_token: *const c_char,
}
const FAST_SINGLE_ANSWER: i32 = 0;
const MAX_PRECISION: i32 = 1;
const MIN_LATENCY: i32 = 2;
const MIN_MEMORY_USAGE: i32 = 3;

#[link(name = "tensorflowlite_jni")]
extern "C" {
    fn TfLiteModelCreateFromFile(path: *const c_char) -> *mut c_void;
    fn TfLiteInterpreterOptionsCreate() -> *mut c_void;
    fn TfLiteInterpreterOptionsSetNumThreads(o: *mut c_void, n: i32);
    fn TfLiteInterpreterOptionsAddDelegate(o: *mut c_void, d: *mut c_void);
    fn TfLiteInterpreterCreate(model: *const c_void, o: *const c_void) -> *mut c_void;
    fn TfLiteInterpreterDelete(i: *mut c_void);
    fn TfLiteInterpreterOptionsDelete(o: *mut c_void);
    fn TfLiteModelDelete(m: *mut c_void);
    fn TfLiteInterpreterAllocateTensors(i: *mut c_void) -> i32;
    fn TfLiteInterpreterInvoke(i: *mut c_void) -> i32;
    fn TfLiteInterpreterGetInputTensorCount(i: *const c_void) -> i32;
    fn TfLiteInterpreterGetOutputTensorCount(i: *const c_void) -> i32;
    fn TfLiteInterpreterGetInputTensor(i: *const c_void, idx: i32) -> *mut c_void;
    fn TfLiteInterpreterGetOutputTensor(i: *const c_void, idx: i32) -> *const c_void;
    fn TfLiteTensorNumDims(t: *const c_void) -> i32;
    fn TfLiteTensorDim(t: *const c_void, d: i32) -> i32;
    fn TfLiteTensorCopyFromBuffer(t: *mut c_void, data: *const c_void, bytes: usize) -> i32;
    fn TfLiteTensorCopyToBuffer(t: *const c_void, data: *mut c_void, bytes: usize) -> i32;
    fn TfLiteTensorData(t: *const c_void) -> *mut c_void;
    fn TfLiteTensorByteSize(t: *const c_void) -> usize;
}

#[link(name = "tensorflowlite_gpu_jni")]
extern "C" {
    fn TfLiteGpuDelegateOptionsV2Default() -> GpuOptions;
    fn TfLiteGpuDelegateV2Create(o: *const GpuOptions) -> *mut c_void;
    fn TfLiteGpuDelegateV2Delete(d: *mut c_void);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Accel {
    Cpu,
    /// GPU delegate, full fp32.
    Gpu,
    /// GPU delegate, fp16 allowed (about twice as fast on Mali).
    GpuFp16,
}

fn dims(t: *const c_void) -> Vec<usize> {
    unsafe { (0..TfLiteTensorNumDims(t)).map(|d| TfLiteTensorDim(t, d) as usize).collect() }
}

pub struct LiteRtDemucs {
    interp: *mut c_void,
    // Owned by this engine and freed with it (see Drop); the delegate is null on the CPU.
    model: *mut c_void,
    opts: *mut c_void,
    delegate: *mut c_void,
    // Tensor indices, told apart by rank: the waveforms are 3-D, the spectrograms 4-D.
    in_mix: i32,
    in_mag: i32,
    out_wave: i32,
    out_spec: i32,
    stft: Stft,
    seg: usize,
    frames: usize,
    // Scratch kept between runs, rather than ~30 MB allocated afresh each time
    // (tidier on a phone; it made no measurable difference to the run time).
    z: Vec<Complex32>,
    y: Vec<f32>,
}

impl LiteRtDemucs {
    pub fn load(path: &Path, accel: Accel, threads: usize) -> Result<Self> {
        let cpath = CString::new(path.to_str().context("model path")?)?;
        let null = std::ptr::null_mut();
        // Filled in step by step, so that Drop frees whatever exists if a step fails.
        let mut m = LiteRtDemucs {
            interp: null,
            model: null,
            opts: null,
            delegate: null,
            in_mix: -1,
            in_mag: -1,
            out_wave: -1,
            out_spec: -1,
            stft: Stft::new(),
            seg: 0,
            frames: 0,
            z: Vec::new(),
            y: Vec::new(),
        };
        unsafe {
            m.model = TfLiteModelCreateFromFile(cpath.as_ptr());
            ensure!(!m.model.is_null(), "loading {}", path.display());
            m.opts = TfLiteInterpreterOptionsCreate();
            if threads > 0 {
                TfLiteInterpreterOptionsSetNumThreads(m.opts, threads as i32);
            }
            if accel != Accel::Cpu {
                let mut g = TfLiteGpuDelegateOptionsV2Default();
                // Not SUSTAINED_SPEED: its exhaustive OpenCL tuning took 218 s at
                // start-up on a Pixel 10a and made the runs no faster (1.24 s vs 1.22 s).
                g.inference_preference = FAST_SINGLE_ANSWER;
                if accel == Accel::GpuFp16 {
                    g.is_precision_loss_allowed = 1;
                    g.inference_priority1 = MIN_LATENCY;
                    g.inference_priority2 = MIN_MEMORY_USAGE;
                    g.inference_priority3 = MAX_PRECISION;
                } else {
                    g.is_precision_loss_allowed = 0;
                    g.inference_priority1 = MAX_PRECISION;
                    g.inference_priority2 = MIN_LATENCY;
                    g.inference_priority3 = MIN_MEMORY_USAGE;
                }
                m.delegate = TfLiteGpuDelegateV2Create(&g);
                ensure!(!m.delegate.is_null(), "creating the GPU delegate");
                TfLiteInterpreterOptionsAddDelegate(m.opts, m.delegate);
            }
            m.interp = TfLiteInterpreterCreate(m.model, m.opts);
            ensure!(!m.interp.is_null(), "creating the interpreter (GPU delegate rejected the graph?)");
            ensure!(TfLiteInterpreterAllocateTensors(m.interp) == 0, "allocating tensors");
            ensure!(TfLiteInterpreterGetInputTensorCount(m.interp) == 2, "expected 2 inputs");
            ensure!(TfLiteInterpreterGetOutputTensorCount(m.interp) == 2, "expected 2 outputs");
            for i in 0..2 {
                let d = dims(TfLiteInterpreterGetInputTensor(m.interp, i));
                match d.as_slice() {
                    [1, 2, s] => (m.in_mix, m.seg) = (i, *s),
                    [1, 4, BINS, f] => (m.in_mag, m.frames) = (i, *f),
                    _ => bail!("unexpected input shape {d:?}"),
                }
                match dims(TfLiteInterpreterGetOutputTensor(m.interp, i)).len() {
                    3 => m.out_wave = i,
                    4 => m.out_spec = i,
                    n => bail!("unexpected output rank {n}"),
                }
            }
        }
        ensure!(m.in_mix >= 0 && m.in_mag >= 0 && m.out_wave >= 0 && m.out_spec >= 0, "model I/O");
        ensure!(m.frames == frames_for(m.seg), "{} frames for a {}-sample segment", m.frames, m.seg);
        m.z = vec![Complex32::new(0.0, 0.0); BINS * m.frames];
        m.y = vec![0f32; m.seg];
        m.separate(&vec![0.0; 2 * m.seg])?; // first run compiles the GPU kernels
        Ok(m)
    }
}

impl Drop for LiteRtDemucs {
    /// Frees the engine (about 1 GB with the GPU delegate), so models can be swapped
    /// at run time. The interpreter goes first: it uses the other three.
    fn drop(&mut self) {
        unsafe {
            if !self.interp.is_null() {
                TfLiteInterpreterDelete(self.interp);
            }
            if !self.delegate.is_null() {
                TfLiteGpuDelegateV2Delete(self.delegate);
            }
            if !self.opts.is_null() {
                TfLiteInterpreterOptionsDelete(self.opts);
            }
            if !self.model.is_null() {
                TfLiteModelDelete(self.model);
            }
        }
    }
}

impl Separator for LiteRtDemucs {
    fn window_len(&self) -> usize {
        self.seg
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let (seg, fr) = (self.seg, self.frames);
        assert_eq!(mix.len(), 2 * seg);
        let plane = BINS * fr;
        let z = &mut self.z;
        let mut out = vec![0f32; 2 * seg]; // time branch
        // The spectrograms are written to and read from the tensors' own memory.
        unsafe {
            let i = self.interp;
            let bytes = |v: &[f32]| std::mem::size_of_val(v);
            let t = TfLiteInterpreterGetInputTensor(i, self.in_mix);
            ensure!(TfLiteTensorCopyFromBuffer(t, mix.as_ptr().cast(), bytes(mix)) == 0, "mix in");
            let t = TfLiteInterpreterGetInputTensor(i, self.in_mag);
            ensure!(TfLiteTensorByteSize(t) == 16 * plane, "mag in");
            // [4, BINS, frames] = L re, L im, R re, R im
            let mag = std::slice::from_raw_parts_mut(TfLiteTensorData(t).cast::<f32>(), 4 * plane);
            for ch in 0..2 {
                self.stft.spec(&mix[ch * seg..(ch + 1) * seg], z);
                let (re, im) = mag[2 * ch * plane..(2 * ch + 2) * plane].split_at_mut(plane);
                for (i, c) in z.iter().enumerate() {
                    re[i] = c.re;
                    im[i] = c.im;
                }
            }
            ensure!(TfLiteInterpreterInvoke(i) == 0, "LiteRT invoke failed");
            let t = TfLiteInterpreterGetOutputTensor(i, self.out_wave);
            ensure!(TfLiteTensorCopyToBuffer(t, out.as_mut_ptr().cast(), bytes(&out)) == 0, "wave out");
            let t = TfLiteInterpreterGetOutputTensor(i, self.out_spec);
            ensure!(TfLiteTensorByteSize(t) == 16 * plane, "spec out");
            let spec = std::slice::from_raw_parts(TfLiteTensorData(t).cast::<f32>(), 4 * plane);
            for ch in 0..2 {
                let re = &spec[2 * ch * plane..(2 * ch + 1) * plane];
                let im = &spec[(2 * ch + 1) * plane..(2 * ch + 2) * plane];
                for i in 0..plane {
                    z[i] = Complex32::new(re[i], im[i]);
                }
                self.stft.ispec(z, seg, &mut self.y);
                for (o, v) in out[ch * seg..(ch + 1) * seg].iter_mut().zip(&self.y) {
                    *o += v;
                }
            }
        }
        Ok(out)
    }
}
