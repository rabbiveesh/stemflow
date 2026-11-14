//! JNI surface of stemflow for the Android app (`org.stemflow.app.Native`).
//!
//! The separator (LiteRT on the GPU), STFT and sliding-window streaming are the same
//! code the desktop and the bench run, pulled in by path. Two independent handles:
//!   - an engine: feed one hop of captured audio, get that hop's bass and band stems;
//!   - a mixer: turn stems into output at the current levels, with the limiter.
//! They are used from different threads (worker and player), so they share nothing.

#![allow(dead_code)]

#[path = "../../bench/src/litert.rs"]
mod litert;
#[path = "../../../src/model.rs"]
mod model;
#[path = "../../../src/spec.rs"]
mod spec;
#[path = "../../../src/stream.rs"]
mod stream;

use anyhow::{ensure, Result};
use jni::objects::{JClass, JFloatArray, JString};
use jni::sys::{jfloat, jint, jlong};
use jni::JNIEnv;
use litert::{Accel, LiteRtDemucs};
use model::{MonoFold, Separator, SAMPLE_RATE};
use std::time::Instant;
use stream::Streamer;

type Engine = Streamer<MonoFold<LiteRtDemucs>>;

/// Highest target level: 2.0 sounds twice as loud as the original (+10 dB).
const MAX_BASS: f32 = 2.0;

// loudness_to_gain and Limiter mirror src/control.rs and src/audio.rs, which can't be
// included here (D-Bus and PulseAudio). They belong in a shared core module once the
// repo is split into a workspace.

/// Perceived loudness (0.5 = half as loud, 2 = twice as loud) to amplitude.
fn loudness_to_gain(p: f32) -> f32 {
    let p = p.clamp(0.0, MAX_BASS);
    if p == 0.0 {
        0.0
    } else {
        p.powf(std::f32::consts::LOG2_10 / 2.0)
    }
}

/// Block peak limiter with one block of lookahead (see src/audio.rs).
struct Limiter {
    g: f32,
}

impl Limiter {
    const CEIL: f32 = 0.97;
    /// ~6 dB/s at 20 ms blocks.
    const RELEASE: f32 = 1.014;

    fn apply(&mut self, out: &mut [f32], next_peak: f32) {
        let peak = out.iter().fold(0f32, |m, x| m.max(x.abs()));
        let need = |p: f32| if p > Self::CEIL { Self::CEIL / p } else { 1.0 };
        let target = (self.g * Self::RELEASE)
            .min(1.0)
            .min(need(peak))
            .min(need(next_peak));
        let n = out.len() / 2;
        if self.g < 1.0 || target < 1.0 {
            for i in 0..n {
                let a = (i as f32 + 1.0) / n as f32;
                let g = self.g + (target - self.g) * a;
                out[2 * i] *= g;
                out[2 * i + 1] *= g;
            }
        }
        self.g = target;
    }
}

struct Mixer {
    g: (f32, f32),
    lim: Limiter,
}

impl Mixer {
    /// `bass`/`band` hold `block` frames to play followed by the frames after them
    /// (the limiter's lookahead); returns the `block` frames to play.
    fn mix(&mut self, bass: &[f32], band: &[f32], block: usize, levels: (f32, f32)) -> Vec<f32> {
        let target = (loudness_to_gain(levels.0), loudness_to_gain(levels.1));
        let mut out = vec![0f32; 2 * block];
        for i in 0..block {
            let a = (i as f32 + 1.0) / block as f32; // ramp: no zipper noise
            let gb = self.g.0 + (target.0 - self.g.0) * a;
            let gr = self.g.1 + (target.1 - self.g.1) * a;
            for ch in 0..2 {
                let k = 2 * i + ch;
                out[k] = gb * bass[k] + gr * band[k];
            }
        }
        let next_peak = bass[2 * block..]
            .iter()
            .zip(&band[2 * block..])
            .fold(0f32, |m, (b, r)| m.max((target.0 * b + target.1 * r).abs()));
        self.lim.apply(&mut out, next_peak);
        self.g = target;
        out
    }
}

fn throw(env: &mut JNIEnv, e: anyhow::Error) {
    let _ = env.throw_new("java/lang/RuntimeException", format!("{e:#}"));
}

/// Load a `.tflite` htdemucs model. `gpu`: 0 = CPU, 1 = GPU fp32, 2 = GPU fp16.
/// Times are in seconds. Blocks for several seconds (model load + GPU kernel build).
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineNew<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    path: JString<'l>,
    gpu: jint,
    hop: jfloat,
    lookahead: jfloat,
    xfade: jfloat,
) -> jlong {
    let r = (|| -> Result<jlong> {
        let path: String = env.get_string(&path)?.into();
        let accel = match gpu {
            0 => Accel::Cpu,
            1 => Accel::Gpu,
            _ => Accel::GpuFp16,
        };
        let sep = MonoFold(LiteRtDemucs::load(std::path::Path::new(&path), accel, 0)?);
        let secs = |s: f32| (s * SAMPLE_RATE as f32).round() as usize;
        let (hop, la, xf) = (secs(hop), secs(lookahead), secs(xfade));
        ensure!(hop > 0 && xf <= la, "crossfade must fit inside the lookahead");
        ensure!(hop + la <= sep.window_len(), "hop + lookahead must fit inside the model window");
        Ok(Box::into_raw(Box::new(Streamer::new(sep, hop, la, xf))) as jlong)
    })();
    r.unwrap_or_else(|e| {
        throw(&mut env, e);
        0
    })
}

/// Frames per hop: `process` takes and returns exactly this many stereo frames.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineHop(_: JNIEnv, _: JClass, h: jlong) -> jint {
    unsafe { &*(h as *const Engine) }.hop as jint
}

/// Frames the output trails the input by (the lookahead).
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineDelay(_: JNIEnv, _: JClass, h: jlong) -> jint {
    unsafe { &*(h as *const Engine) }.delay() as jint
}

/// One hop of interleaved stereo in; that hop's `bass` and `band` (interleaved) out.
/// Returns the compute time in seconds.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineProcess<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
    input: JFloatArray<'l>,
    bass: JFloatArray<'l>,
    band: JFloatArray<'l>,
) -> jfloat {
    let r = (|| -> Result<f32> {
        let eng = unsafe { &mut *(h as *mut Engine) };
        let mut buf = vec![0f32; 2 * eng.hop];
        ensure!(env.get_array_length(&input)? as usize == buf.len(), "input is not one hop");
        env.get_float_array_region(&input, 0, &mut buf)?;
        let t = Instant::now();
        let st = eng.process(&buf)?;
        let dt = t.elapsed().as_secs_f32();
        env.set_float_array_region(&bass, 0, &st.bass)?;
        env.set_float_array_region(&band, 0, &st.band)?;
        Ok(dt)
    })();
    r.unwrap_or_else(|e| {
        throw(&mut env, e);
        0.0
    })
}

#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_mixerNew(_: JNIEnv, _: JClass) -> jlong {
    Box::into_raw(Box::new(Mixer { g: (0.0, 0.0), lim: Limiter { g: 1.0 } })) as jlong
}

/// Mix `block` frames starting at frame `pos` of the stem buffers into `out`, at the
/// given perceived loudness levels (bass 0..2, band 0..1). `ahead` more frames after
/// the block must be present; the limiter looks at them.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_mixerMix<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
    bass: JFloatArray<'l>,
    band: JFloatArray<'l>,
    pos: jint,
    block: jint,
    ahead: jint,
    bass_level: jfloat,
    band_level: jfloat,
    out: JFloatArray<'l>,
) {
    let r = (|| -> Result<()> {
        let m = unsafe { &mut *(h as *mut Mixer) };
        let (block, n) = (block as usize, 2 * (block + ahead) as usize);
        let (mut b, mut r) = (vec![0f32; n], vec![0f32; n]);
        env.get_float_array_region(&bass, 2 * pos, &mut b)?;
        env.get_float_array_region(&band, 2 * pos, &mut r)?;
        let o = m.mix(&b, &r, block, (bass_level, band_level));
        env.set_float_array_region(&out, 0, &o)?;
        Ok(())
    })();
    if let Err(e) = r {
        throw(&mut env, e);
    }
}
