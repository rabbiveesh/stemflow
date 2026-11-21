//! htdemucs on LiteRT (TFLite), optionally on the GPU, through the C API in Google's
//! AARs. Runs models from ../tools/export_litert.py: inputs `mix` [1, 2, seg] and
//! `mag` [1, 4, BINS, frames], outputs the stem's spectrogram [1, 4, BINS, frames]
//! and time-branch waveform [1, 2, seg]. The STFT, iSTFT and final sum are the
//! desktop's (spec.rs), as in model.rs' HtDemucs.
//! LiteRtRoformer does the same for BS-RoFormer (../tools/export_bsrofo_litert.py).
//! LiteRtBeatNet runs Beat This! (../tools/export_beat_this_litert.py), which needs no
//! transform here: log-mel in, logits out.
//!
//! Two runtimes, chosen at build time (their libraries export clashing symbols):
//!   - default: LiteRT 1.4.2's interpreter and GPU delegate
//!     (libtensorflowlite_jni.so, libtensorflowlite_gpu_jni.so);
//!   - feature `litert2`: LiteRT 2.3.2's CompiledModel and its GPU accelerator
//!     ("ML Drift": libLiteRt.so, plus libLiteRtClGlAccelerator.so where the dynamic
//!     linker finds it). 10-14% faster on a Pixel 10a for the same model files.

use crate::model::Separator;
use crate::spec::{frames_for, HannStft, Stft, BINS};
use anyhow::{bail, ensure, Result};
use engine::Engine;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Accel {
    Cpu,
    /// GPU delegate, full fp32.
    Gpu,
    /// GPU delegate, fp16 allowed (about twice as fast on Mali).
    GpuFp16,
}

/// LiteRT 1.4.2: the TFLite interpreter with the GPU delegate.
#[cfg(not(feature = "litert2"))]
mod engine {
    use super::Accel;
    use anyhow::{ensure, Context, Result};
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

    fn dims(t: *const c_void) -> Vec<usize> {
        unsafe {
            (0..TfLiteTensorNumDims(t))
                .map(|d| TfLiteTensorDim(t, d) as usize)
                .collect()
        }
    }

    /// An interpreter with everything it owns; freed on drop (about 1 GB with the GPU
    /// delegate), so models can be swapped at run time.
    pub struct Engine {
        interp: *mut c_void,
        model: *mut c_void,
        opts: *mut c_void,
        delegate: *mut c_void, // null on the CPU
    }

    impl Engine {
        pub fn load(path: &Path, accel: Accel, threads: usize) -> Result<Self> {
            let cpath = CString::new(path.to_str().context("model path")?)?;
            let null = std::ptr::null_mut();
            // Filled in step by step, so that Drop frees whatever exists if a step fails.
            let mut e = Engine {
                interp: null,
                model: null,
                opts: null,
                delegate: null,
            };
            unsafe {
                e.model = TfLiteModelCreateFromFile(cpath.as_ptr());
                ensure!(!e.model.is_null(), "loading {}", path.display());
                e.opts = TfLiteInterpreterOptionsCreate();
                if threads > 0 {
                    TfLiteInterpreterOptionsSetNumThreads(e.opts, threads as i32);
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
                    e.delegate = TfLiteGpuDelegateV2Create(&g);
                    ensure!(!e.delegate.is_null(), "creating the GPU delegate");
                    TfLiteInterpreterOptionsAddDelegate(e.opts, e.delegate);
                }
                e.interp = TfLiteInterpreterCreate(e.model, e.opts);
                ensure!(
                    !e.interp.is_null(),
                    "creating the interpreter (GPU delegate rejected the graph?)"
                );
                ensure!(
                    TfLiteInterpreterAllocateTensors(e.interp) == 0,
                    "allocating tensors"
                );
            }
            Ok(e)
        }

        pub fn io_counts(&self) -> (i32, i32) {
            unsafe {
                (
                    TfLiteInterpreterGetInputTensorCount(self.interp),
                    TfLiteInterpreterGetOutputTensorCount(self.interp),
                )
            }
        }

        pub fn input_dims(&self, i: i32) -> Vec<usize> {
            dims(unsafe { TfLiteInterpreterGetInputTensor(self.interp, i) })
        }

        pub fn output_dims(&self, i: i32) -> Vec<usize> {
            dims(unsafe { TfLiteInterpreterGetOutputTensor(self.interp, i) })
        }

        /// Input tensor `i`'s own memory, to be filled before `invoke`.
        pub fn input(&mut self, i: i32, len: usize) -> Result<&mut [f32]> {
            unsafe {
                let t = TfLiteInterpreterGetInputTensor(self.interp, i);
                ensure!(TfLiteTensorByteSize(t) == 4 * len, "input {i} size");
                Ok(std::slice::from_raw_parts_mut(
                    TfLiteTensorData(t).cast::<f32>(),
                    len,
                ))
            }
        }

        /// Output tensor `i`'s own memory, valid until the next `invoke`.
        pub fn output(&mut self, i: i32, len: usize) -> Result<&[f32]> {
            unsafe {
                let t = TfLiteInterpreterGetOutputTensor(self.interp, i);
                ensure!(TfLiteTensorByteSize(t) == 4 * len, "output {i} size");
                Ok(std::slice::from_raw_parts(
                    TfLiteTensorData(t).cast::<f32>(),
                    len,
                ))
            }
        }

        pub fn invoke(&mut self) -> Result<()> {
            ensure!(
                unsafe { TfLiteInterpreterInvoke(self.interp) } == 0,
                "LiteRT invoke failed"
            );
            Ok(())
        }
    }

    impl Drop for Engine {
        /// The interpreter goes first: it uses the other three.
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
}

/// LiteRT 2.3.2: CompiledModel with the GPU accelerator (ML Drift).
#[cfg(feature = "litert2")]
mod engine {
    use super::Accel;
    use anyhow::{ensure, Context, Result};
    use std::ffi::{c_char, c_void, CString};
    use std::path::Path;

    /// An accelerated model with everything it owns; freed on drop.
    pub struct Engine {
        env: *mut c_void,
        model: *mut c_void,
        compiled: *mut c_void,
        inputs: Vec<Buffer>,
        outputs: Vec<Buffer>,
    }

    struct Buffer {
        handle: *mut c_void,
        dims: Vec<usize>,
        locked: *mut f32, // null while unlocked
    }

    /// LiteRtRankedTensorType (litert_model_types.h, litert_layout.h; ABI-stable).
    #[repr(C)]
    struct RankedTensorType {
        element_type: i32,
        bits: u32, // rank: 7 bits, has_strides: 1 bit
        dimensions: [i32; 8],
        strides: [u32; 8],
    }
    const FLOAT32: i32 = 1;
    const HW_CPU: i32 = 1;
    const HW_GPU: i32 = 2;
    const LOCK_READ: i32 = 0;
    const LOCK_WRITE: i32 = 1;

    #[link(name = "LiteRt")]
    extern "C" {
        fn LiteRtCreateEnvironment(n: i32, options: *const c_void, env: *mut *mut c_void) -> i32;
        fn LiteRtDestroyEnvironment(env: *mut c_void);
        fn LiteRtCreateModelFromFile(
            env: *mut c_void,
            path: *const c_char,
            model: *mut *mut c_void,
        ) -> i32;
        fn LiteRtDestroyModel(model: *mut c_void);
        fn LiteRtCreateOptions(options: *mut *mut c_void) -> i32;
        fn LiteRtDestroyOptions(options: *mut c_void);
        fn LiteRtSetOptionsHardwareAccelerators(options: *mut c_void, set: i32) -> i32;
        fn LiteRtCreateOpaqueOptions(
            id: *const c_char,
            payload: *mut c_void,
            destructor: extern "C" fn(*mut c_void),
            opaque: *mut *mut c_void,
        ) -> i32;
        fn LiteRtAddOpaqueOptions(options: *mut c_void, opaque: *mut c_void) -> i32;
        fn LiteRtCreateCompiledModel(
            env: *mut c_void,
            model: *mut c_void,
            options: *mut c_void,
            cm: *mut *mut c_void,
        ) -> i32;
        fn LiteRtDestroyCompiledModel(cm: *mut c_void);
        fn LiteRtCompiledModelIsFullyAccelerated(cm: *mut c_void, full: *mut bool) -> i32;
        fn LiteRtGetModelSignature(
            model: *mut c_void,
            index: usize,
            signature: *mut *mut c_void,
        ) -> i32;
        fn LiteRtGetNumSignatureInputs(signature: *mut c_void, n: *mut usize) -> i32;
        fn LiteRtGetNumSignatureOutputs(signature: *mut c_void, n: *mut usize) -> i32;
        fn LiteRtGetSignatureInputTensorByIndex(
            signature: *mut c_void,
            i: usize,
            tensor: *mut *mut c_void,
        ) -> i32;
        fn LiteRtGetSignatureOutputTensorByIndex(
            signature: *mut c_void,
            i: usize,
            tensor: *mut *mut c_void,
        ) -> i32;
        fn LiteRtGetRankedTensorType(tensor: *mut c_void, ty: *mut RankedTensorType) -> i32;
        fn LiteRtGetCompiledModelInputBufferRequirements(
            cm: *mut c_void,
            sig: usize,
            i: usize,
            req: *mut *mut c_void,
        ) -> i32;
        fn LiteRtGetCompiledModelOutputBufferRequirements(
            cm: *mut c_void,
            sig: usize,
            i: usize,
            req: *mut *mut c_void,
        ) -> i32;
        fn LiteRtCreateManagedTensorBufferFromRequirements(
            env: *mut c_void,
            ty: *const RankedTensorType,
            req: *mut c_void,
            buffer: *mut *mut c_void,
        ) -> i32;
        fn LiteRtDestroyTensorBuffer(buffer: *mut c_void);
        fn LiteRtLockTensorBuffer(buffer: *mut c_void, addr: *mut *mut c_void, mode: i32) -> i32;
        fn LiteRtUnlockTensorBuffer(buffer: *mut c_void) -> i32;
        fn LiteRtRunCompiledModel(
            cm: *mut c_void,
            sig: usize,
            n_in: usize,
            inputs: *mut *mut c_void,
            n_out: usize,
            outputs: *mut *mut c_void,
        ) -> i32;
    }

    extern "C" fn free_payload(p: *mut c_void) {
        drop(unsafe { CString::from_raw(p.cast()) });
    }

    macro_rules! ok {
        ($call:expr, $what:expr) => {
            ensure!(unsafe { $call } == 0, "LiteRT: {}", $what)
        };
    }

    impl Engine {
        /// `threads` is for the CPU accelerator (XNNPACK); 0 leaves it to the runtime.
        pub fn load(path: &Path, accel: Accel, threads: usize) -> Result<Self> {
            let cpath = CString::new(path.to_str().context("model path")?)?;
            let null = std::ptr::null_mut();
            // Filled in step by step, so that Drop frees whatever exists if a step fails.
            let mut e = Engine {
                env: null,
                model: null,
                compiled: null,
                inputs: Vec::new(),
                outputs: Vec::new(),
            };
            ok!(
                LiteRtCreateEnvironment(0, std::ptr::null(), &mut e.env),
                "creating the environment"
            );
            ok!(
                LiteRtCreateModelFromFile(e.env, cpath.as_ptr(), &mut e.model),
                format!("loading {}", path.display())
            );
            let mut opts = null;
            ok!(LiteRtCreateOptions(&mut opts), "creating options");
            let built = (|| {
                let hw = if accel == Accel::Cpu { HW_CPU } else { HW_GPU };
                ok!(
                    LiteRtSetOptionsHardwareAccelerators(opts, hw),
                    "choosing the accelerator"
                );
                // Accelerator options travel as TOML under the accelerator's name, as
                // litert_gpu_options.cc and litert_cpu_options.cc serialize them (the
                // Lrt*Options helpers themselves are not exported by libLiteRt.so).
                // GPU precision: 1 = fp16, 2 = fp32.
                // STEMFLOW_GPU_OPTIONS adds lines to the GPU's, e.g. "priority = 3"
                // (1 low, 2 normal, 3 high), for experiments.
                let extra = std::env::var("STEMFLOW_GPU_OPTIONS").unwrap_or_default();
                let named = match accel {
                    Accel::Cpu if threads == 0 => None,
                    Accel::Cpu => Some((c"xnnpack", format!("num_threads = {threads}\n"))),
                    Accel::Gpu => Some((c"gpu_options", format!("precision = 2\n{extra}\n"))),
                    Accel::GpuFp16 => Some((c"gpu_options", format!("precision = 1\n{extra}\n"))),
                };
                if let Some((name, toml)) = named {
                    let mut opaque = null;
                    ok!(
                        LiteRtCreateOpaqueOptions(
                            name.as_ptr(),
                            CString::new(toml)?.into_raw().cast(),
                            free_payload,
                            &mut opaque
                        ),
                        "creating accelerator options"
                    );
                    ok!(
                        LiteRtAddOpaqueOptions(opts, opaque),
                        "adding accelerator options"
                    );
                }
                ok!(
                    LiteRtCreateCompiledModel(e.env, e.model, opts, &mut e.compiled),
                    "compiling the model (is libLiteRtClGlAccelerator.so on the library path?)"
                );
                Ok(())
            })();
            unsafe { LiteRtDestroyOptions(opts) };
            built?;
            let mut full = false;
            ok!(
                LiteRtCompiledModelIsFullyAccelerated(e.compiled, &mut full),
                "querying acceleration"
            );
            ensure!(
                full || accel == Accel::Cpu,
                "the GPU accelerator did not take the whole graph"
            );

            let mut sig = null;
            ok!(
                LiteRtGetModelSignature(e.model, 0, &mut sig),
                "reading the signature"
            );
            let (mut n_in, mut n_out) = (0, 0);
            ok!(
                LiteRtGetNumSignatureInputs(sig, &mut n_in),
                "counting inputs"
            );
            ok!(
                LiteRtGetNumSignatureOutputs(sig, &mut n_out),
                "counting outputs"
            );
            for (n, output) in [(n_in, false), (n_out, true)] {
                for i in 0..n {
                    let (mut tensor, mut req, mut handle) = (null, null, null);
                    let mut ty: RankedTensorType = unsafe { std::mem::zeroed() };
                    if output {
                        ok!(
                            LiteRtGetSignatureOutputTensorByIndex(sig, i, &mut tensor),
                            "output tensor"
                        );
                        ok!(
                            LiteRtGetCompiledModelOutputBufferRequirements(
                                e.compiled, 0, i, &mut req
                            ),
                            "output buffer"
                        );
                    } else {
                        ok!(
                            LiteRtGetSignatureInputTensorByIndex(sig, i, &mut tensor),
                            "input tensor"
                        );
                        ok!(
                            LiteRtGetCompiledModelInputBufferRequirements(
                                e.compiled, 0, i, &mut req
                            ),
                            "input buffer"
                        );
                    }
                    ok!(LiteRtGetRankedTensorType(tensor, &mut ty), "tensor type");
                    ensure!(ty.element_type == FLOAT32, "tensor {i} is not float32");
                    let rank = (ty.bits & 0x7f) as usize;
                    let dims = ty.dimensions[..rank].iter().map(|&d| d as usize).collect();
                    ok!(
                        LiteRtCreateManagedTensorBufferFromRequirements(
                            e.env,
                            &ty,
                            req,
                            &mut handle
                        ),
                        "creating a buffer"
                    );
                    let b = Buffer {
                        handle,
                        dims,
                        locked: std::ptr::null_mut(),
                    };
                    if output {
                        e.outputs.push(b)
                    } else {
                        e.inputs.push(b)
                    }
                }
            }
            Ok(e)
        }

        pub fn io_counts(&self) -> (i32, i32) {
            (self.inputs.len() as i32, self.outputs.len() as i32)
        }

        pub fn input_dims(&self, i: i32) -> Vec<usize> {
            self.inputs[i as usize].dims.clone()
        }

        pub fn output_dims(&self, i: i32) -> Vec<usize> {
            self.outputs[i as usize].dims.clone()
        }

        fn lock(b: &mut Buffer, len: usize, mode: i32) -> Result<*mut f32> {
            ensure!(b.dims.iter().product::<usize>() == len, "tensor size");
            if b.locked.is_null() {
                let mut p = std::ptr::null_mut();
                ok!(
                    LiteRtLockTensorBuffer(b.handle, &mut p, mode),
                    "locking a buffer"
                );
                b.locked = p.cast();
            }
            Ok(b.locked)
        }

        /// Input tensor `i`'s own memory, to be filled before `invoke`.
        pub fn input(&mut self, i: i32, len: usize) -> Result<&mut [f32]> {
            let p = Self::lock(&mut self.inputs[i as usize], len, LOCK_WRITE)?;
            Ok(unsafe { std::slice::from_raw_parts_mut(p, len) })
        }

        /// Output tensor `i`'s own memory, valid until the next `invoke`.
        pub fn output(&mut self, i: i32, len: usize) -> Result<&[f32]> {
            let p = Self::lock(&mut self.outputs[i as usize], len, LOCK_READ)?;
            Ok(unsafe { std::slice::from_raw_parts(p, len) })
        }

        fn unlock_all(&mut self) {
            for b in self.inputs.iter_mut().chain(self.outputs.iter_mut()) {
                if !b.locked.is_null() {
                    unsafe { LiteRtUnlockTensorBuffer(b.handle) };
                    b.locked = std::ptr::null_mut();
                }
            }
        }

        pub fn invoke(&mut self) -> Result<()> {
            self.unlock_all();
            let mut ins: Vec<_> = self.inputs.iter().map(|b| b.handle).collect();
            let mut outs: Vec<_> = self.outputs.iter().map(|b| b.handle).collect();
            ok!(
                LiteRtRunCompiledModel(
                    self.compiled,
                    0,
                    ins.len(),
                    ins.as_mut_ptr(),
                    outs.len(),
                    outs.as_mut_ptr()
                ),
                "run failed"
            );
            Ok(())
        }
    }

    impl Drop for Engine {
        fn drop(&mut self) {
            self.unlock_all();
            unsafe {
                for b in self.inputs.iter().chain(self.outputs.iter()) {
                    LiteRtDestroyTensorBuffer(b.handle);
                }
                if !self.compiled.is_null() {
                    LiteRtDestroyCompiledModel(self.compiled);
                }
                if !self.model.is_null() {
                    LiteRtDestroyModel(self.model);
                }
                if !self.env.is_null() {
                    LiteRtDestroyEnvironment(self.env);
                }
            }
        }
    }
}

/// One channel's spectrogram -> its two of the `[4, bins, frames]` planes (L re, L im, R re, R im).
fn to_planes(z: &[Complex32], planes: &mut [f32]) {
    let (re, im) = planes.split_at_mut(z.len());
    for (i, c) in z.iter().enumerate() {
        re[i] = c.re;
        im[i] = c.im;
    }
}

/// The inverse STFT of spec.rs (`Stft::ispec` or `HannStft::istft`), for only the
/// samples the caller will use. A live stream keeps `hop + crossfade` samples of each
/// window, so inverting all of a 7.8 s window is mostly wasted: 80 ms per run on a
/// Pixel 10a. Frames are added in the same order with the same arithmetic as in
/// spec.rs, so the samples it does produce are bit-identical.
struct Istft {
    inv: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    nfft: usize,
    hop: usize,
    bins: usize,       // bins per frame in the spectrogram (demucs has no Nyquist bin)
    pad_frames: usize, // all-zero frames assumed on each side (demucs: 2)
    start: usize,      // overlap-add position of output sample 0
    scale: f32,
    guard: bool, // torch.istft's "window envelope is zero" check
}

impl Istft {
    fn new(
        nfft: usize,
        hop: usize,
        bins: usize,
        pad_frames: usize,
        start: usize,
        scale: f32,
        guard: bool,
    ) -> Self {
        let window = (0..nfft)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / nfft as f32).cos())
            .collect();
        let inv = RealFftPlanner::<f32>::new().plan_fft_inverse(nfft);
        Istft {
            inv,
            window,
            nfft,
            hop,
            bins,
            pad_frames,
            start,
            scale,
            guard,
        }
    }

    /// As `Stft::ispec`.
    fn demucs() -> Self {
        let n = crate::spec::NFFT;
        Self::new(
            n,
            n / 4,
            BINS,
            2,
            n / 2 + n / 8 * 3,
            1.0 / (n as f32).sqrt(),
            false,
        )
    }

    /// As `HannStft::istft`.
    fn hann(nfft: usize, hop: usize) -> Self {
        Self::new(
            nfft,
            hop,
            nfft / 2 + 1,
            0,
            nfft / 2,
            1.0 / nfft as f32,
            true,
        )
    }

    /// `planes` is one channel's (re, im) planes, `[bin * frames + frame]` each. Writes
    /// (or, with `add`, adds to) `out[range]`; the rest of `out` is left alone.
    fn run(&self, planes: &[f32], frames: usize, range: Range<usize>, out: &mut [f32], add: bool) {
        let (n, hop) = (self.nfft, self.hop);
        let (re, im) = planes.split_at(self.bins * frames);
        let (lo, hi) = (self.start + range.start, self.start + range.end); // in overlap-add positions
        let mut y = vec![0f32; range.len()];
        let mut env = vec![0f32; range.len()];
        let mut spec = self.inv.make_input_vec();
        let mut frame = self.inv.make_output_vec();
        let first = (lo + 1).saturating_sub(n).div_ceil(hop);
        let last = ((hi - 1) / hop).min(frames + 2 * self.pad_frames - 1);
        for f in first..=last {
            let off = f * hop;
            let (i0, i1) = (lo.saturating_sub(off), (hi - off).min(n));
            for i in i0..i1 {
                env[off + i - lo] += self.window[i] * self.window[i];
            }
            if f < self.pad_frames || f >= frames + self.pad_frames {
                continue;
            }
            let col = f - self.pad_frames;
            for bin in 0..self.bins {
                spec[bin] = Complex32::new(re[bin * frames + col], im[bin * frames + col]);
            }
            spec[0].im = 0.0;
            if self.bins == n / 2 {
                spec[n / 2] = Complex32::new(0.0, 0.0);
            } else {
                spec[self.bins - 1].im = 0.0;
            }
            self.inv.process(&mut spec, &mut frame).expect("ifft");
            for i in i0..i1 {
                y[off + i - lo] += frame[i] * self.scale * self.window[i];
            }
        }
        for ((o, y), e) in out[range].iter_mut().zip(&y).zip(&env) {
            let v = if self.guard && *e <= 1e-11 {
                0.0
            } else {
                y / e
            };
            *o = if add { *o + v } else { v };
        }
    }
}

pub struct LiteRtDemucs {
    engine: Engine,
    // Tensor indices, told apart by rank: the waveforms are 3-D, the spectrograms 4-D.
    in_mix: i32,
    in_mag: i32,
    out_wave: i32,
    out_spec: i32,
    stft: Stft,
    seg: usize,
    frames: usize,
    istft: Istft,
    needed: Range<usize>,
    z: Vec<Complex32>, // scratch kept between runs
}

impl LiteRtDemucs {
    pub fn load(path: &Path, accel: Accel, threads: usize) -> Result<Self> {
        let engine = Engine::load(path, accel, threads)?;
        ensure!(
            engine.io_counts() == (2, 2),
            "expected 2 inputs and 2 outputs"
        );
        let (mut in_mix, mut in_mag, mut out_wave, mut out_spec) = (-1, -1, -1, -1);
        let (mut seg, mut frames) = (0, 0);
        for i in 0..2 {
            let d = engine.input_dims(i);
            match d.as_slice() {
                [1, 2, s] => (in_mix, seg) = (i, *s),
                [1, 4, BINS, f] => (in_mag, frames) = (i, *f),
                _ => bail!("unexpected input shape {d:?}"),
            }
            match engine.output_dims(i).len() {
                3 => out_wave = i,
                4 => out_spec = i,
                n => bail!("unexpected output rank {n}"),
            }
        }
        ensure!(
            in_mix >= 0 && in_mag >= 0 && out_wave >= 0 && out_spec >= 0,
            "model I/O"
        );
        ensure!(
            frames == frames_for(seg),
            "{frames} frames for a {seg}-sample segment"
        );
        let mut m = LiteRtDemucs {
            engine,
            in_mix,
            in_mag,
            out_wave,
            out_spec,
            stft: Stft::new(),
            seg,
            frames,
            istft: Istft::demucs(),
            needed: 0..seg,
            z: vec![Complex32::new(0.0, 0.0); BINS * frames],
        };
        m.separate(&vec![0.0; 2 * seg])?; // first run compiles the GPU kernels
        Ok(m)
    }

    /// Only these samples of each window will be used (a stream's are
    /// `seg - lookahead - hop .. seg - lookahead + crossfade`): `separate` then leaves
    /// the rest of what it returns without the spectrogram branch, i.e. wrong.
    pub fn set_needed(&mut self, range: Range<usize>) {
        assert!(range.start < range.end && range.end <= self.seg);
        self.needed = range;
    }
}

impl Separator for LiteRtDemucs {
    fn window_len(&self) -> usize {
        self.seg
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let (seg, plane) = (self.seg, BINS * self.frames);
        assert_eq!(mix.len(), 2 * seg);
        self.engine
            .input(self.in_mix, 2 * seg)?
            .copy_from_slice(mix);
        let mag = self.engine.input(self.in_mag, 4 * plane)?;
        for ch in 0..2 {
            self.stft.spec(&mix[ch * seg..(ch + 1) * seg], &mut self.z);
            to_planes(&self.z, &mut mag[2 * ch * plane..(2 * ch + 2) * plane]);
        }
        self.engine.invoke()?;
        let mut out = self.engine.output(self.out_wave, 2 * seg)?.to_vec(); // time branch
        let spec = self.engine.output(self.out_spec, 4 * plane)?;
        for ch in 0..2 {
            let planes = &spec[2 * ch * plane..(2 * ch + 2) * plane];
            self.istft.run(
                planes,
                self.frames,
                self.needed.clone(),
                &mut out[ch * seg..(ch + 1) * seg],
                true,
            );
        }
        Ok(out)
    }
}

/// BS-RoFormer on LiteRT, from ../tools/export_bsrofo_litert.py: input is the
/// spectrogram `[1, 4, 1025, frames]` (STFT n_fft 2048, hop 512, Hann), as in model.rs'
/// BsRoformer; output is `[1, 4 * heads, 1025, frames]`, one stem per head (bass, or
/// bass then vocals). The window is fixed by the model file.
pub struct LiteRtRoformer {
    engine: Engine,
    stft: HannStft,
    seg: usize,
    frames: usize,
    heads: usize,
    istft: Istft,
    needed: Range<usize>,
    z: Vec<Complex32>,
}

impl LiteRtRoformer {
    pub fn load(path: &Path, accel: Accel, threads: usize) -> Result<Self> {
        let engine = Engine::load(path, accel, threads)?;
        ensure!(
            engine.io_counts() == (1, 1),
            "expected 1 input and 1 output"
        );
        let stft = HannStft::new(2048, 512);
        let d = engine.input_dims(0);
        let frames = match d.as_slice() {
            [1, 4, b, f] if *b == stft.bins() => *f,
            _ => bail!("unexpected input shape {d:?}"),
        };
        let o = engine.output_dims(0);
        let heads = match o.as_slice() {
            [1, c, b, f] if c % 4 == 0 && *c > 0 && *b == stft.bins() && *f == frames => c / 4,
            _ => bail!("unexpected output shape {o:?}"),
        };
        let seg = (frames - 1) * 512;
        let mut m = LiteRtRoformer {
            engine,
            z: vec![Complex32::new(0.0, 0.0); stft.bins() * frames],
            istft: Istft::hann(stft.nfft, stft.hop),
            needed: 0..seg,
            stft,
            seg,
            frames,
            heads,
        };
        m.separate(&vec![0.0; 2 * seg])?; // first run compiles the GPU kernels
        Ok(m)
    }

    /// As `LiteRtDemucs::set_needed`; the rest of what `separate` returns is zero.
    pub fn set_needed(&mut self, range: Range<usize>) {
        assert!(range.start < range.end && range.end <= self.seg);
        self.needed = range;
    }
}

impl Separator for LiteRtRoformer {
    fn window_len(&self) -> usize {
        self.seg
    }

    fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
        let (seg, plane) = (self.seg, self.z.len());
        assert_eq!(mix.len(), 2 * seg);
        let spec_in = self.engine.input(0, 4 * plane)?;
        for ch in 0..2 {
            self.stft.stft(&mix[ch * seg..(ch + 1) * seg], &mut self.z);
            to_planes(&self.z, &mut spec_in[2 * ch * plane..(2 * ch + 2) * plane]);
        }
        self.engine.invoke()?;
        let spec = self.engine.output(0, 4 * plane * self.heads)?;
        let mut out = vec![0f32; 2 * seg * self.heads];
        for h in 0..self.heads {
            let (s, o) = (&spec[4 * plane * h..], &mut out[2 * seg * h..]);
            for ch in 0..2 {
                let planes = &s[2 * ch * plane..(2 * ch + 2) * plane];
                self.istft.run(
                    planes,
                    self.frames,
                    self.needed.clone(),
                    &mut o[ch * seg..(ch + 1) * seg],
                    false,
                );
            }
        }
        Ok(out)
    }

    fn stems(&self) -> usize {
        self.heads
    }
}

/// Beat This! on LiteRT, from ../tools/export_beat_this_litert.py: a log-mel spectrogram
/// `[1, frames, mels]` in (50 fps; 1500 x 128 as exported), beat and downbeat logits
/// `[1, frames]` out. Meant for the CPU: the GPU is the separator's.
pub struct LiteRtBeatNet {
    engine: Engine,
    frames: usize,
    mels: usize,
}

impl LiteRtBeatNet {
    pub fn load(path: &Path, accel: Accel, threads: usize) -> Result<Self> {
        let engine = Engine::load(path, accel, threads)?;
        ensure!(
            engine.io_counts() == (1, 2),
            "expected 1 input and 2 outputs"
        );
        let d = engine.input_dims(0);
        let (frames, mels) = match d.as_slice() {
            [1, f, m] => (*f, *m),
            _ => bail!("unexpected input shape {d:?}"),
        };
        for i in 0..2 {
            ensure!(engine.output_dims(i) == [1, frames], "output {i} shape");
        }
        Ok(LiteRtBeatNet {
            engine,
            frames,
            mels,
        })
    }

    /// Spectrogram frames per run.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// `frames x mels` spectrogram in (row-major, one frame after another); beat and
    /// downbeat logits out, `frames` each.
    pub fn run(&mut self, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(spect.len() == self.frames * self.mels, "spectrogram size");
        self.engine.input(0, spect.len())?.copy_from_slice(spect);
        self.engine.invoke()?;
        let beat = self.engine.output(0, self.frames)?.to_vec();
        let downbeat = self.engine.output(1, self.frames)?.to_vec();
        Ok((beat, downbeat))
    }
}
