//! JNI surface of stemflow for the Android app (`org.stemflow.app.Native`).
//!
//! The separator (LiteRT on the GPU), STFT and sliding-window streaming are the same
//! code the desktop and the bench run, pulled in by path. Two independent handles:
//!   - an engine: feed one hop of captured audio, get that hop's bass and band stems;
//!   - a mixer: turn stems into output at the current levels, with the limiter.
//! They are used from different threads (worker and player), so they share nothing.

#![allow(dead_code)]

#[path = "../../../src/beats.rs"]
mod beats;
#[path = "../../bench/src/litert.rs"]
mod litert;
#[path = "../../../src/looper.rs"]
mod looper;
#[path = "../../../src/model.rs"]
mod model;
#[path = "../../../src/spec.rs"]
mod spec;
#[path = "../../../src/stream.rs"]
mod stream;
#[path = "../../../src/stretch.rs"]
mod stretch;

use anyhow::{ensure, Result};
use jni::objects::{JClass, JDoubleArray, JFloatArray, JString};
use jni::sys::{jfloat, jfloatArray, jint, jlong};
use jni::JNIEnv;
use litert::{Accel, LiteRtBeatNet, LiteRtDemucs, LiteRtRoformer};
use model::{MonoFold, Separator, SAMPLE_RATE};
use std::time::Instant;
use stream::Streamer;

type Engine = Streamer<MonoFold<Box<dyn Separator>>>;

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
    g: (f32, f32, f32),
    lim: Limiter,
}

impl Mixer {
    /// `bass`/`band`/`second` hold `block` frames to play followed by the frames after
    /// them (the limiter's lookahead); `second` is empty for a one-stem model. Returns
    /// the `block` frames to play.
    fn mix(
        &mut self,
        bass: &[f32],
        band: &[f32],
        second: &[f32],
        block: usize,
        levels: (f32, f32, f32),
    ) -> Vec<f32> {
        let target = (
            loudness_to_gain(levels.0),
            loudness_to_gain(levels.1),
            loudness_to_gain(levels.2),
        );
        let sec = |k: usize| second.get(k).copied().unwrap_or(0.0);
        let mut out = vec![0f32; 2 * block];
        for i in 0..block {
            let a = (i as f32 + 1.0) / block as f32; // ramp: no zipper noise
            let gb = self.g.0 + (target.0 - self.g.0) * a;
            let gr = self.g.1 + (target.1 - self.g.1) * a;
            let gs = self.g.2 + (target.2 - self.g.2) * a;
            for ch in 0..2 {
                let k = 2 * i + ch;
                out[k] = gb * bass[k] + gr * band[k] + gs * sec(k);
            }
        }
        let next_peak = (2 * block..bass.len()).fold(0f32, |m, k| {
            m.max((target.0 * bass[k] + target.1 * band[k] + target.2 * sec(k)).abs())
        });
        self.lim.apply(&mut out, next_peak);
        self.g = target;
        out
    }
}

fn throw(env: &mut JNIEnv, e: anyhow::Error) {
    let _ = env.throw_new("java/lang/RuntimeException", format!("{e:#}"));
}

/// Load a `.tflite` model: BS-RoFormer if the file name starts with `bsrofo` (as the
/// export scripts name them), htdemucs otherwise. `gpu`: 0 = CPU, 1 = GPU fp32, 2 = GPU fp16.
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
        let path = std::path::Path::new(&path);
        let roformer = path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("bsrofo"));
        let sep: Box<dyn Separator> = if roformer {
            Box::new(LiteRtRoformer::load(path, accel, 0)?)
        } else {
            Box::new(LiteRtDemucs::load(path, accel, 0)?)
        };
        let sep = MonoFold(sep);
        let secs = |s: f32| (s * SAMPLE_RATE as f32).round() as usize;
        let (hop, la, xf) = (secs(hop), secs(lookahead), secs(xfade));
        ensure!(
            hop > 0 && xf <= la,
            "crossfade must fit inside the lookahead"
        );
        ensure!(
            hop + la <= sep.window_len(),
            "hop + lookahead must fit inside the model window"
        );
        Ok(Box::into_raw(Box::new(Streamer::new(sep, hop, la, xf))) as jlong)
    })();
    r.unwrap_or_else(|e| {
        throw(&mut env, e);
        0
    })
}

/// Release an engine (model, GPU delegate and their ~1 GB). `h` is dead afterwards.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineFree(_: JNIEnv, _: JClass, h: jlong) {
    if h != 0 {
        drop(unsafe { Box::from_raw(h as *mut Engine) });
    }
}

/// Frames per hop: `process` takes and returns exactly this many stereo frames.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineHop(
    _: JNIEnv,
    _: JClass,
    h: jlong,
) -> jint {
    unsafe { &*(h as *const Engine) }.hop as jint
}

/// Frames the output trails the input by (the lookahead).
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineDelay(
    _: JNIEnv,
    _: JClass,
    h: jlong,
) -> jint {
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
    second: JFloatArray<'l>,
) -> jfloat {
    let r = (|| -> Result<f32> {
        let eng = unsafe { &mut *(h as *mut Engine) };
        let mut buf = vec![0f32; 2 * eng.hop];
        ensure!(
            env.get_array_length(&input)? as usize == buf.len(),
            "input is not one hop"
        );
        env.get_float_array_region(&input, 0, &mut buf)?;
        let t = Instant::now();
        let st = eng.process(&buf)?;
        let dt = t.elapsed().as_secs_f32();
        env.set_float_array_region(&bass, 0, &st.bass)?;
        env.set_float_array_region(&band, 0, &st.band)?;
        // a two-stem model's second stem (vocals); `second` is empty otherwise
        if let Some(e) = st.extra.first() {
            if env.get_array_length(&second)? as usize >= e.len() {
                env.set_float_array_region(&second, 0, e)?;
            }
        }
        Ok(dt)
    })();
    r.unwrap_or_else(|e| {
        throw(&mut env, e);
        0.0
    })
}

#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_mixerNew(_: JNIEnv, _: JClass) -> jlong {
    Box::into_raw(Box::new(Mixer {
        g: (0.0, 0.0, 0.0),
        lim: Limiter { g: 1.0 },
    })) as jlong
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
    second: JFloatArray<'l>,
    pos: jint,
    block: jint,
    ahead: jint,
    bass_level: jfloat,
    band_level: jfloat,
    second_level: jfloat,
    out: JFloatArray<'l>,
) {
    let r = (|| -> Result<()> {
        let m = unsafe { &mut *(h as *mut Mixer) };
        let (block, n) = (block as usize, 2 * (block + ahead) as usize);
        let (mut b, mut r) = (vec![0f32; n], vec![0f32; n]);
        env.get_float_array_region(&bass, 2 * pos, &mut b)?;
        env.get_float_array_region(&band, 2 * pos, &mut r)?;
        let mut s2 = Vec::new();
        if env.get_array_length(&second)? as usize >= 2 * pos as usize + n {
            s2 = vec![0f32; n];
            env.get_float_array_region(&second, 2 * pos, &mut s2)?;
        }
        let o = m.mix(&b, &r, &s2, block, (bass_level, band_level, second_level));
        env.set_float_array_region(&out, 0, &o)?;
        Ok(())
    })();
    if let Err(e) = r {
        throw(&mut env, e);
    }
}

// ---------------------------------------------------------------- looping (src/looper.rs)
//
// The player pushes every stem block it receives and, while a loop plays, reads its
// blocks from the looper instead of the live stems. A background thread runs `find`
// (Beat This! once over the last minute); the screen edits the loop and draws it.

/// LiteRT's Beat This! as the looper's model.
struct LiteRtBeats(LiteRtBeatNet);

// The interpreter is only used from one thread at a time (behind the looper's lock).
unsafe impl Send for LiteRtBeats {}

impl beats::BeatModel for LiteRtBeats {
    fn run(&mut self, spect: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        self.0.run(spect)
    }
}

fn looper<'a>(h: jlong) -> &'a looper::Looper {
    unsafe { &*(h as *const looper::Looper) }
}

/// `model`: a Beat This! LiteRT export (`beat_this_small0_1500.tflite`), run on the
/// CPU (XNNPACK) with `threads` threads; the GPU is the separator's.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperNew<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    model: JString<'l>,
    threads: jint,
) -> jlong {
    let r = (|| -> Result<jlong> {
        let model: String = env.get_string(&model)?.into();
        let net = LiteRtBeatNet::load(
            std::path::Path::new(&model),
            Accel::Cpu,
            threads.max(1) as usize,
        )?;
        ensure!(
            net.frames() == beats::FRAMES,
            "beat model takes {} frames, not {}",
            net.frames(),
            beats::FRAMES
        );
        Ok(Box::into_raw(Box::new(looper::Looper::new(Box::new(LiteRtBeats(net))))) as jlong)
    })();
    r.unwrap_or_else(|e| {
        throw(&mut env, e);
        0
    })
}

#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperFree(_: JNIEnv, _: JClass, h: jlong) {
    if h != 0 {
        drop(unsafe { Box::from_raw(h as *mut looper::Looper) });
    }
}

/// One block of stems as received (interleaved stereo), in play order.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperPush<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
    bass: JFloatArray<'l>,
    band: JFloatArray<'l>,
    second: JFloatArray<'l>,
) {
    let r = (|| -> Result<()> {
        let n = env.get_array_length(&bass)? as usize;
        let (mut b, mut r) = (vec![0f32; n], vec![0f32; n]);
        env.get_float_array_region(&bass, 0, &mut b)?;
        env.get_float_array_region(&band, 0, &mut r)?;
        let mut s2 = Vec::new();
        if env.get_array_length(&second)? as usize >= n {
            s2 = vec![0f32; n];
            env.get_float_array_region(&second, 0, &mut s2)?;
        }
        looper(h).push(&b, &r, &s2);
        Ok(())
    })();
    if let Err(e) = r {
        throw(&mut env, e);
    }
}

/// Loop the `bars` bars ending at the bar line nearest stem frame `press`. Blocks for
/// the analysis (a few seconds on the CPU); throws if no beat was found.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperFind(
    mut env: JNIEnv,
    _: JClass,
    h: jlong,
    press: jlong,
    bars: jint,
    meter: jint,
    feel: jfloat,
) {
    let s = beats::Settings {
        meter: meter.max(0) as usize,
        feel,
        ..Default::default()
    };
    if let Err(e) = looper(h).find(press.max(0) as u64, bars.max(1) as usize, s) {
        throw(&mut env, e);
    }
}

/// While a loop (or the tail after it) plays: `n` frames into `bass`/`band` (stereo)
/// and `click` (mono), then advance by `advance`. Returns the buffer frame the block
/// starts at, or -1 (arrays untouched) when the live stream should play.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperRead<'l>(
    mut env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
    bass: JFloatArray<'l>,
    band: JFloatArray<'l>,
    second: JFloatArray<'l>,
    click: JFloatArray<'l>,
    n: jint,
    advance: jint,
) -> jlong {
    let r = (|| -> Result<i64> {
        let n = n as usize;
        let (mut b, mut r, mut s2, mut c) = (
            vec![0f32; 2 * n],
            vec![0f32; 2 * n],
            vec![0f32; 2 * n],
            vec![0f32; n],
        );
        let Some(at) = looper(h).read(&mut b, &mut r, &mut s2, &mut c, n, advance as usize) else {
            return Ok(-1);
        };
        env.set_float_array_region(&bass, 0, &b)?;
        env.set_float_array_region(&band, 0, &r)?;
        if env.get_array_length(&second)? as usize >= 2 * n {
            env.set_float_array_region(&second, 0, &s2)?;
        }
        env.set_float_array_region(&click, 0, &c)?;
        Ok(at)
    })();
    r.unwrap_or_else(|e| {
        throw(&mut env, e);
        -1
    })
}

/// Edits from the screen. `op`: 0 back to live (playing on from the loop), 1 play from
/// the top, 2 shift by `a` beats,
/// 3 set the ends to beats `a`..`b`, 4 `a` bars, 5 beat `a` is bar 1, 6 click on (`a`
/// = 1) or off, 7 re-decode with meter `a` (0 = auto) and feel `f`, 9 speed `f`,
/// transposition `a` cents, keep pitch `b`.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperEdit(
    _: JNIEnv,
    _: JClass,
    h: jlong,
    op: jint,
    a: jint,
    b: jint,
    f: jfloat,
) {
    let l = looper(h);
    match op {
        0 => l.go_live(),
        8 => l.stop(),
        1 => l.play(),
        2 => l.shift(a),
        3 => l.set_edges(a.max(0) as usize, b.max(0) as usize),
        4 => l.set_bars(a.max(1) as usize),
        5 => l.bar_one(a.max(0) as usize),
        6 => l.set_click(a != 0),
        7 => l.redecode(a.max(0) as usize, f),
        // speed f, transposition a/100 semitones, keep pitch b != 0
        9 => l.set_speed(f, a as f32 / 100.0, b != 0),
        _ => {}
    }
}

/// The loop for drawing: [playing, finding, bpm, meter, bars, from, to, start, end,
/// playhead, a, b, beats in view, bins, then (frame, bar line?, index) per beat, then
/// the peaks]. Frames are relative to `from` (f32 holds them exactly). Empty while
/// there is no loop.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperView<'l>(
    env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
    bins: jint,
) -> jfloatArray {
    let l = looper(h);
    let mut v: Vec<f32> = Vec::new();
    if let Some(w) = l.view(bins.max(1) as usize) {
        let rel = |f: f64| (f - w.from) as f32;
        v.extend([
            w.playing as u8 as f32,
            w.finding as u8 as f32,
            w.bpm,
            w.meter as f32,
            w.bars as f32,
            w.from as f32,
            rel(w.to),
            rel(w.start),
            rel(w.end),
            rel(w.playhead),
            w.a as f32,
            w.b as f32,
            w.beats.len() as f32,
            w.peaks.len() as f32,
        ]);
        for (f, bar, i) in &w.beats {
            v.extend([rel(*f), *bar as u8 as f32, *i as f32]);
        }
        v.extend(&w.peaks);
    } else if l.finding() {
        v.extend([0.0, 1.0]);
    }
    match env.new_float_array(v.len() as i32) {
        Ok(a) => {
            let _ = env.set_float_array_region(&a, 0, &v);
            a.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// After a model switch: re-separate the loop with the new model (`engine`), so it
/// plays in the new stems. Blocks for the runs (seconds); false without a loop.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperRestem(
    mut env: JNIEnv,
    _: JClass,
    h: jlong,
    engine: jlong,
) -> jni::sys::jboolean {
    let eng = unsafe { &mut *(engine as *mut Engine) };
    match looper::restem(looper(h), eng) {
        Ok(b) => b as jni::sys::jboolean,
        Err(e) => {
            throw(&mut env, e);
            0
        }
    }
}

/// The loop for a bookmark: [start, end, bar 1, beats per bar, bars, feel] in buffer
/// frames (doubles: frame numbers outgrow f32 after a few minutes); empty without one.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperTimes<'l>(
    env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
) -> jni::sys::jdoubleArray {
    let v: Vec<f64> = looper(h)
        .times()
        .map(|(s, e, b, m, n, f)| vec![s, e, b, m as f64, n as f64, f as f64])
        .unwrap_or_default();
    match env.new_double_array(v.len() as i32) {
        Ok(a) => {
            let _ = env.set_double_array_region(&a, 0, &v);
            a.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// A recalled bookmark: the loop's ends and bar 1 onto the beats nearest these frames.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperSetTimes(
    _: JNIEnv,
    _: JClass,
    h: jlong,
    start: jni::sys::jdouble,
    end: jni::sys::jdouble,
    bar1: jni::sys::jdouble,
) {
    looper(h).set_times(start, end, bar1);
}

/// The beats around the loop for a bookmark: [a, b, bar line, meter, bpm, beat
/// frames…] (indices into the beats), or empty without a loop.
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperGrid<'l>(
    env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
) -> jni::sys::jdoubleArray {
    let v: Vec<f64> = looper(h)
        .grid_around(2)
        .map(|(beats, a, b, one, m, bpm)| {
            let mut v = vec![a as f64, b as f64, one as f64, m as f64, bpm as f64];
            v.extend(beats);
            v
        })
        .unwrap_or_default();
    match env.new_double_array(v.len() as i32) {
        Ok(a) => {
            let _ = env.set_double_array_region(&a, 0, &v);
            a.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// A recalled bookmark's own grid (beat frames), looped from beat `a` to `b` with
/// a bar line on beat `one`: no analysis.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_stemflow_app_Native_looperLay(
    mut env: JNIEnv,
    _: JClass,
    h: jlong,
    beats: JDoubleArray,
    a: jint,
    b: jint,
    one: jint,
    meter: jint,
    bpm: jfloat,
    feel: jfloat,
) {
    let n = env.get_array_length(&beats).unwrap_or(0) as usize;
    let mut v = vec![0f64; n];
    let _ = env.get_double_array_region(&beats, 0, &mut v);
    let r = looper(h).lay(
        v,
        a.max(0) as usize,
        b.max(0) as usize,
        one.max(0) as usize,
        meter.max(1) as usize,
        bpm,
        feel,
    );
    if let Err(e) = r {
        throw(&mut env, e);
    }
}

/// The newest minute, mixed back to stereo (debugging: `--es dump 1`).
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperRecent<'l>(
    env: JNIEnv<'l>,
    _: JClass<'l>,
    h: jlong,
) -> jfloatArray {
    let v = looper(h).recent_mix();
    match env.new_float_array(v.len() as i32) {
        Ok(a) => {
            let _ = env.set_float_array_region(&a, 0, &v);
            a.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frames pushed so far (the newest frame the buffer has).
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_looperTotal(
    _: JNIEnv,
    _: JClass,
    h: jlong,
) -> jlong {
    looper(h).total() as jlong
}

/// How many stems the engine's model gives per run (1, or 2 for bass + vocals).
#[no_mangle]
pub extern "system" fn Java_org_stemflow_app_Native_engineStems(
    _: JNIEnv,
    _: JClass,
    h: jlong,
) -> jint {
    unsafe { &*(h as *const Engine) }.stems() as jint
}
