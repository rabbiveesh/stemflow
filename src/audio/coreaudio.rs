//! macOS I/O through Core Audio process taps (macOS 14.2+).
//!
//! A tap on the target app's processes, muted while we read it, is the input of a
//! private aggregate device whose output is the real output device, so one IOProc
//! reads the app and writes our mix on the same clock. The IOProc only moves
//! samples through lock-free rings; resampling between the device rate and
//! 44.1 kHz happens on our own threads. Private taps and aggregate devices belong
//! to the process that made them, so on Ctrl-C (or a crash) the app gets its sound
//! back without any clean-up.
//!
//! Recording another app's audio needs the "System Audio Recording" permission,
//! which macOS gives to the app stemflow was started from (Terminal, iTerm, ...).
//! Without it the tap opens fine and hears silence; `regrab` notices that.

use super::{Capture, Io, Playback};
use crate::model::SAMPLE_RATE;
use anyhow::{anyhow, bail, Result};
use audioadapter_buffers::direct::InterleavedSlice;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::AnyThread;
use objc2_core_audio::*;
use objc2_core_audio_types::{
    kAudioFormatFlagIsFloat, kAudioFormatFlagIsNonInterleaved, AudioBuffer, AudioBufferList,
    AudioStreamBasicDescription, AudioTimeStamp,
};
use objc2_core_foundation::CFDictionary;
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSUUID};
use rubato::{Fft, FixedSync, Resampler};
use std::ffi::{c_void, CStr};
use std::mem::{size_of, MaybeUninit};
use std::ptr::{null, NonNull};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const SYSTEM: AudioObjectID = kAudioObjectSystemObject as AudioObjectID;
const GLOBAL: u32 = kAudioObjectPropertyScopeGlobal;
const OUTPUT: u32 = kAudioObjectPropertyScopeOutput;

fn check(st: i32, what: &str) -> Result<()> {
    if st == 0 {
        return Ok(());
    }
    // Most Core Audio errors are four-character codes.
    let b = st.to_be_bytes();
    let code = if b.iter().all(|c| c.is_ascii_graphic()) {
        format!("'{}'", String::from_utf8_lossy(&b))
    } else {
        st.to_string()
    };
    bail!("{what}: OSStatus {code}")
}

fn addr(sel: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: sel,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// A property of variable size, in 8-byte-aligned storage (it may hold structs).
fn get_bytes(obj: AudioObjectID, sel: u32, scope: u32) -> Result<(Vec<u64>, usize)> {
    let a = addr(sel, scope);
    let mut size = 0u32;
    check(
        unsafe {
            AudioObjectGetPropertyDataSize(
                obj,
                NonNull::from(&a),
                0,
                null(),
                NonNull::from(&mut size),
            )
        },
        "AudioObjectGetPropertyDataSize",
    )?;
    let mut buf = vec![0u64; (size as usize).div_ceil(8).max(1)];
    check(
        unsafe {
            AudioObjectGetPropertyData(
                obj,
                NonNull::from(&a),
                0,
                null(),
                NonNull::from(&mut size),
                NonNull::new(buf.as_mut_ptr()).unwrap().cast(),
            )
        },
        "AudioObjectGetPropertyData",
    )?;
    Ok((buf, size as usize))
}

/// A fixed-size property (`T` must be plain data).
fn get<T: Copy>(obj: AudioObjectID, sel: u32, scope: u32) -> Result<T> {
    let a = addr(sel, scope);
    let mut v = MaybeUninit::<T>::zeroed();
    let mut size = size_of::<T>() as u32;
    check(
        unsafe {
            AudioObjectGetPropertyData(
                obj,
                NonNull::from(&a),
                0,
                null(),
                NonNull::from(&mut size),
                NonNull::new(v.as_mut_ptr()).unwrap().cast(),
            )
        },
        "AudioObjectGetPropertyData",
    )?;
    Ok(unsafe { v.assume_init() })
}

fn get_ids(obj: AudioObjectID, sel: u32) -> Result<Vec<AudioObjectID>> {
    let (buf, size) = get_bytes(obj, sel, GLOBAL)?;
    let p = buf.as_ptr() as *const AudioObjectID;
    Ok(unsafe { std::slice::from_raw_parts(p, size / size_of::<AudioObjectID>()) }.to_vec())
}

/// A CFString property (returned retained, so we own it).
fn get_string(obj: AudioObjectID, sel: u32) -> Result<String> {
    let p: *mut NSString = get(obj, sel, GLOBAL)?;
    Ok(unsafe { Retained::from_raw(p) }.map_or_else(String::new, |s| s.to_string()))
}

fn ns(s: &CStr) -> Retained<NSString> {
    NSString::from_str(s.to_str().unwrap())
}

struct Process {
    id: AudioObjectID,
    pid: i32,
    name: String,
}

fn exe_name(pid: i32) -> String {
    let mut buf = [0u8; 4096];
    let n = unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
    if n <= 0 {
        return String::new();
    }
    let path = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
    path.rsplit('/').next().unwrap_or_default().to_string()
}

/// Core Audio's client processes whose bundle id or executable name contains
/// `needle` (case-insensitive), except us. Spotify plays from a helper process
/// (`com.spotify.client.helper`), which this catches too.
fn find_processes(needle: &str) -> Result<Vec<Process>> {
    let needle = needle.to_lowercase();
    let me = std::process::id() as i32;
    let mut out = Vec::new();
    for id in get_ids(SYSTEM, kAudioHardwarePropertyProcessObjectList)? {
        let pid: i32 = get(id, kAudioProcessPropertyPID, GLOBAL).unwrap_or(-1);
        if pid == me {
            continue;
        }
        let bundle = get_string(id, kAudioProcessPropertyBundleID).unwrap_or_default();
        let name = if bundle.is_empty() {
            exe_name(pid)
        } else {
            bundle
        };
        if name.to_lowercase().contains(&needle) || exe_name(pid).to_lowercase().contains(&needle) {
            out.push(Process { id, pid, name });
        }
    }
    out.sort_by_key(|p| p.id);
    Ok(out)
}

fn is_playing(p: &Process) -> bool {
    get::<u32>(p.id, kAudioProcessPropertyIsRunningOutput, GLOBAL).is_ok_and(|v| v != 0)
}

/// Channels per stream in a scope, from kAudioDevicePropertyStreamConfiguration.
fn stream_channels(dev: AudioObjectID, scope: u32) -> Result<Vec<u32>> {
    let (buf, _) = get_bytes(dev, kAudioDevicePropertyStreamConfiguration, scope)?;
    let list = unsafe { &*(buf.as_ptr() as *const AudioBufferList) };
    let bufs =
        unsafe { std::slice::from_raw_parts(list.mBuffers.as_ptr(), list.mNumberBuffers as usize) };
    Ok(bufs.iter().map(|b| b.mNumberChannels).collect())
}

struct Device {
    uid: String,
    name: String,
}

fn device(id: AudioObjectID) -> Result<Device> {
    Ok(Device {
        uid: get_string(id, kAudioDevicePropertyDeviceUID)?,
        name: get_string(id, kAudioObjectPropertyName).unwrap_or_default(),
    })
}

/// The default output, or the output device whose UID is `want` or whose name contains it.
fn output_device(want: Option<&str>) -> Result<Device> {
    let Some(want) = want else {
        return device(get(
            SYSTEM,
            kAudioHardwarePropertyDefaultOutputDevice,
            GLOBAL,
        )?);
    };
    let mut names = Vec::new();
    for id in get_ids(SYSTEM, kAudioHardwarePropertyDevices)? {
        let outs: u32 = stream_channels(id, OUTPUT).unwrap_or_default().iter().sum();
        let Ok(d) = device(id) else { continue };
        if outs == 0 {
            continue;
        }
        if d.uid == want || d.name.to_lowercase().contains(&want.to_lowercase()) {
            return Ok(d);
        }
        names.push(format!("{} (uid {})", d.name, d.uid));
    }
    bail!("no output device `{want}`; there are: {}", names.join(", "))
}

fn tap_description(procs: &[Process]) -> Retained<CATapDescription> {
    let ids: Vec<Retained<NSNumber>> = procs.iter().map(|p| NSNumber::new_u32(p.id)).collect();
    let ids = NSArray::from_retained_slice(&ids);
    unsafe {
        let d = CATapDescription::initStereoMixdownOfProcesses(CATapDescription::alloc(), &ids);
        d.setName(&NSString::from_str("stemflow"));
        d.setPrivate(true);
        // Silent on the speakers only while we read it: if we die, it plays again.
        d.setMuteBehavior(CATapMuteBehavior::MutedWhenTapped);
        d
    }
}

/// One channel inside a buffer list: `ptr[i * stride]` is frame i.
#[derive(Clone, Copy)]
struct Chan {
    ptr: *mut f32,
    stride: usize,
}

/// The two stereo channels of a buffer list: from its first two channels (`first`)
/// or its last two. A single mono channel is used for both.
fn pick(bufs: &[AudioBuffer], first: bool) -> Option<(Chan, Chan, usize)> {
    let ch = |b: &AudioBuffer, c: usize| Chan {
        ptr: unsafe { (b.mData as *mut f32).add(c) },
        stride: b.mNumberChannels as usize,
    };
    let frames =
        |b: &AudioBuffer| b.mDataByteSize as usize / 4 / (b.mNumberChannels as usize).max(1);
    let usable = |b: &&AudioBuffer| !b.mData.is_null() && b.mNumberChannels > 0;
    let b = if first {
        bufs.iter().find(usable)?
    } else {
        bufs.iter().rev().find(usable)?
    };
    let n = b.mNumberChannels as usize;
    if n >= 2 {
        let c = if first { 0 } else { n - 2 };
        return Some((ch(b, c), ch(b, c + 1), frames(b)));
    }
    // Mono buffers: pair this one with its neighbour if that's mono too.
    let i = bufs.iter().position(|x| std::ptr::eq(x, b)).unwrap();
    let j = if first { i + 1 } else { i.wrapping_sub(1) };
    match bufs.get(j).filter(|x| usable(x) && x.mNumberChannels == 1) {
        Some(o) if first => Some((ch(b, 0), ch(o, 0), frames(b).min(frames(o)))),
        Some(o) => Some((ch(o, 0), ch(b, 0), frames(b).min(frames(o)))),
        None => Some((ch(b, 0), ch(b, 0), frames(b))),
    }
}

unsafe fn buffers<'a>(l: NonNull<AudioBufferList>) -> &'a [AudioBuffer] {
    let l = l.as_ref();
    std::slice::from_raw_parts(l.mBuffers.as_ptr(), l.mNumberBuffers as usize)
}

/// Counters the real-time thread leaves for the others.
#[derive(Default)]
struct Stats {
    /// Largest |sample| captured since the last look (f32 bits; positive floats
    /// order like their bits).
    peak: AtomicU32,
    /// Captured frames dropped because the reader fell 2 s behind.
    dropped: AtomicU64,
    /// Callbacks that found too little to play after playback had started.
    underruns: AtomicU64,
    /// IOProc calls so far, and the input buffer list's shape on the last one
    /// (buffers << 16 | channels of the last buffer), for when nothing arrives.
    callbacks: AtomicU64,
    input_shape: AtomicU32,
}

/// What the IOProc owns.
struct Rt {
    cap: rtrb::Producer<f32>,
    play: rtrb::Consumer<f32>,
    playing: bool,
    stats: Arc<Stats>,
}

unsafe extern "C-unwind" fn io_proc(
    _dev: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    _in_time: NonNull<AudioTimeStamp>,
    output: NonNull<AudioBufferList>,
    _out_time: NonNull<AudioTimeStamp>,
    data: *mut c_void,
) -> i32 {
    let rt = &mut *(data as *mut Rt);
    rt.stats.callbacks.fetch_add(1, Ordering::Relaxed);
    let ins = buffers(input);
    let shape = (ins.len() as u32) << 16 | ins.last().map_or(0, |b| b.mNumberChannels);
    rt.stats.input_shape.store(shape, Ordering::Relaxed);
    // The tap's streams come after any of the output device's own inputs.
    if let Some((l, r, n)) = pick(buffers(input), false) {
        let mut peak = 0f32;
        if rt.cap.slots() >= 2 * n {
            if let Ok(chunk) = rt.cap.write_chunk_uninit(2 * n) {
                chunk.fill_from_iter((0..n).flat_map(|i| {
                    let (a, b) = (*l.ptr.add(i * l.stride), *r.ptr.add(i * r.stride));
                    peak = peak.max(a.abs()).max(b.abs());
                    [a, b]
                }));
            }
        } else {
            rt.stats.dropped.fetch_add(n as u64, Ordering::Relaxed);
        }
        rt.stats.peak.fetch_max(peak.to_bits(), Ordering::Relaxed);
    }
    let outs = buffers(output);
    for b in outs {
        if !b.mData.is_null() {
            std::ptr::write_bytes(b.mData as *mut u8, 0, b.mDataByteSize as usize);
        }
    }
    if let Some((l, r, n)) = pick(outs, true) {
        let have = rt.play.slots() / 2;
        if have < n && rt.playing {
            rt.stats.underruns.fetch_add(1, Ordering::Relaxed);
        }
        let m = have.min(n);
        if m > 0 {
            rt.playing = true;
            if let Ok(chunk) = rt.play.read_chunk(2 * m) {
                let (a, b) = chunk.as_slices();
                for (i, v) in a.iter().chain(b).enumerate() {
                    let c = if i % 2 == 0 { l } else { r };
                    *c.ptr.add((i / 2) * c.stride) = *v;
                }
                chunk.commit_all();
            }
        }
    }
    0
}

/// Synchronous resampling of interleaved stereo, any chunk size in.
struct Rs {
    r: Fft<f32>,
    pending: Vec<f32>,
    out: Vec<f32>,
}

impl Rs {
    fn new(from: u32, to: u32) -> Result<Option<Self>> {
        if from == to {
            return Ok(None);
        }
        let chunk = from as usize / 50;
        let r = Fft::<f32>::new(from as usize, to as usize, chunk, 1, 2, FixedSync::Input)
            .map_err(|e| anyhow!("resampler {from} -> {to} Hz: {e:?}"))?;
        Ok(Some(Rs {
            r,
            pending: Vec::new(),
            out: Vec::new(),
        }))
    }

    fn push(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        self.pending.extend_from_slice(x);
        let mut res = Vec::new();
        loop {
            let need = self.r.input_frames_next();
            if self.pending.len() < 2 * need {
                return Ok(res);
            }
            let n_out = self.r.output_frames_next();
            self.out.resize(2 * n_out, 0.0);
            let inb = InterleavedSlice::new(&self.pending[..2 * need], 2, need)
                .map_err(|e| anyhow!("{e:?}"))?;
            let mut outb =
                InterleavedSlice::new_mut(&mut self.out, 2, n_out).map_err(|e| anyhow!("{e:?}"))?;
            let (used, wrote) = self
                .r
                .process_into_buffer(&inb, &mut outb, None)
                .map_err(|e| anyhow!("resampling: {e:?}"))?;
            self.pending.drain(..2 * used);
            res.extend_from_slice(&self.out[..2 * wrote]);
        }
    }
}

struct Cap {
    rx: rtrb::Consumer<f32>,
    rs: Option<Rs>,
    /// Frames per read at the device rate (20 ms).
    chunk: usize,
    stats: Arc<Stats>,
    /// Said once that nothing arrives.
    told: bool,
}

impl Capture for Cap {
    fn read(&mut self) -> Result<Vec<f32>> {
        loop {
            let t = std::time::Instant::now();
            while self.rx.slots() < 2 * self.chunk {
                std::thread::sleep(Duration::from_millis(5));
                if !self.told && t.elapsed() > Duration::from_secs(3) {
                    self.told = true;
                    let shape = self.stats.input_shape.load(Ordering::Relaxed);
                    eprintln!(
                        "stemflow: no audio from the tap after 3 s ({} device callbacks; last \
                         input: {} buffer(s), {} channel(s) in the last)",
                        self.stats.callbacks.load(Ordering::Relaxed),
                        shape >> 16,
                        shape & 0xffff
                    );
                }
            }
            let c = self.rx.read_chunk(2 * self.chunk)?;
            let (a, b) = c.as_slices();
            let v = [a, b].concat();
            c.commit_all();
            let v = match &mut self.rs {
                Some(rs) => rs.push(&v)?,
                None => v,
            };
            if !v.is_empty() {
                return Ok(v);
            }
        }
    }
}

struct Play {
    tx: rtrb::Producer<f32>,
    rs: Option<Rs>,
}

impl Playback for Play {
    fn write(&mut self, x: &[f32]) -> Result<()> {
        let y = match &mut self.rs {
            Some(rs) => rs.push(x)?,
            None => x.to_vec(),
        };
        let mut off = 0;
        while off < y.len() {
            let n = self.tx.slots().min(y.len() - off) & !1;
            if n == 0 {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            let c = self.tx.write_chunk_uninit(n)?;
            off += c.fill_from_iter(y[off..off + n].iter().copied());
        }
        Ok(())
    }
}

/// The tap, the aggregate device and its IOProc, torn down in reverse on drop.
struct Route {
    tap: AudioObjectID,
    tap_uuid: String,
    agg: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    rt: *mut Rt,
    procs: Vec<AudioObjectID>,
}

// The raw parts are only touched again in `drop` and `retarget`, from one thread.
unsafe impl Send for Route {}

impl Route {
    /// Point the tap at a new set of processes (the app restarted or opened another).
    fn retarget(&mut self, procs: &[Process]) -> Result<()> {
        let d = tap_description(procs);
        let uuid = NSUUID::from_string(&NSString::from_str(&self.tap_uuid))
            .ok_or_else(|| anyhow!("bad tap uuid"))?;
        unsafe { d.setUUID(&uuid) };
        let a = addr(kAudioTapPropertyDescription, GLOBAL);
        let p = Retained::as_ptr(&d);
        check(
            unsafe {
                AudioObjectSetPropertyData(
                    self.tap,
                    NonNull::from(&a),
                    0,
                    null(),
                    size_of::<*const CATapDescription>() as u32,
                    NonNull::from(&p).cast(),
                )
            },
            "updating the tap",
        )?;
        self.procs = procs.iter().map(|p| p.id).collect();
        Ok(())
    }
}

impl Drop for Route {
    fn drop(&mut self) {
        unsafe {
            AudioDeviceStop(self.agg, self.proc_id);
            AudioDeviceDestroyIOProcID(self.agg, self.proc_id);
            AudioHardwareDestroyAggregateDevice(self.agg);
            AudioHardwareDestroyProcessTap(self.tap);
            drop(Box::from_raw(self.rt));
        }
    }
}

fn describe(procs: &[Process]) -> String {
    let names: Vec<String> = procs
        .iter()
        .map(|p| format!("{} [{}]", p.name, p.pid))
        .collect();
    names.join(", ")
}

pub fn open(grab: &str, output: Option<String>, _name: &str, play_frames: usize) -> Result<Io> {
    let out = output_device(output.as_deref())?;
    let mut procs = find_processes(grab)?;
    if procs.is_empty() {
        eprintln!("stemflow: waiting for `{grab}` to open audio (start it, or press play)");
        while procs.is_empty() {
            std::thread::sleep(Duration::from_secs(1));
            procs = find_processes(grab)?;
        }
    }

    let desc = tap_description(&procs);
    let tap_uuid = unsafe { desc.UUID() }.UUIDString().to_string();
    let mut tap: AudioObjectID = 0;
    check(
        unsafe { AudioHardwareCreateProcessTap(Some(&desc), &mut tap) },
        "creating the process tap (needs macOS 14.2+)",
    )?;
    let tap_fmt: AudioStreamBasicDescription = get(tap, kAudioTapPropertyFormat, GLOBAL)?;
    if tap_fmt.mFormatFlags & kAudioFormatFlagIsFloat == 0 {
        unsafe { AudioHardwareDestroyProcessTap(tap) };
        bail!("the tap's format isn't float ({:?})", tap_fmt.mFormatFlags);
    }

    // One private device: the real output (it sets the clock), plus the tap as input.
    let sub = NSDictionary::<NSString, AnyObject>::from_slices(
        &[&*ns(kAudioSubDeviceUIDKey)],
        &[&*NSString::from_str(&out.uid) as &AnyObject],
    );
    let tap_entry = NSDictionary::<NSString, AnyObject>::from_slices(
        &[
            &*ns(kAudioSubTapUIDKey),
            &*ns(kAudioSubTapDriftCompensationKey),
        ],
        &[
            &*NSString::from_str(&tap_uuid) as &AnyObject,
            &*NSNumber::new_bool(true),
        ],
    );
    let agg_uid = format!("org.stemflow.route.{}", tap_uuid);
    let (yes, no) = (NSNumber::new_bool(true), NSNumber::new_bool(false));
    let subs = NSArray::from_retained_slice(&[sub]);
    let taps = NSArray::from_retained_slice(&[tap_entry]);
    let (name, uid, main) = (
        NSString::from_str("stemflow"),
        NSString::from_str(&agg_uid),
        NSString::from_str(&out.uid),
    );
    let keys = [
        ns(kAudioAggregateDeviceNameKey),
        ns(kAudioAggregateDeviceUIDKey),
        ns(kAudioAggregateDeviceMainSubDeviceKey),
        ns(kAudioAggregateDeviceIsPrivateKey),
        ns(kAudioAggregateDeviceIsStackedKey),
        ns(kAudioAggregateDeviceTapAutoStartKey),
        ns(kAudioAggregateDeviceSubDeviceListKey),
        ns(kAudioAggregateDeviceTapListKey),
    ];
    let vals: [&AnyObject; 8] = [&name, &uid, &main, &yes, &no, &yes, &subs, &taps];
    let keys: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let dict = NSDictionary::<NSString, AnyObject>::from_slices(&keys, &vals);
    // NSDictionary and CFDictionary are toll-free bridged.
    let cf = unsafe { &*(Retained::as_ptr(&dict) as *const CFDictionary) };
    let mut agg: AudioObjectID = 0;
    if let Err(e) = check(
        unsafe { AudioHardwareCreateAggregateDevice(cf, NonNull::from(&mut agg)) },
        "creating the aggregate device",
    ) {
        unsafe { AudioHardwareDestroyProcessTap(tap) };
        return Err(e);
    }

    let rate: f64 = get(agg, kAudioDevicePropertyNominalSampleRate, GLOBAL)?;
    let rate = rate.round() as u32;
    if tap_fmt.mSampleRate.round() as u32 != rate {
        eprintln!(
            "stemflow: note: tap at {} Hz, device at {rate} Hz",
            tap_fmt.mSampleRate
        );
    }
    if tap_fmt.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0 {
        eprintln!("stemflow: note: the tap is non-interleaved");
    }
    let dev_play = (play_frames as u64 * rate as u64 / SAMPLE_RATE as u64) as usize;
    let (cap_tx, cap_rx) = rtrb::RingBuffer::<f32>::new(2 * 2 * rate as usize); // 2 s
    let (play_tx, play_rx) = rtrb::RingBuffer::<f32>::new(2 * dev_play.max(64));
    let stats = Arc::new(Stats::default());
    let rt = Box::into_raw(Box::new(Rt {
        cap: cap_tx,
        play: play_rx,
        playing: false,
        stats: stats.clone(),
    }));
    let mut proc_id: AudioDeviceIOProcID = None;
    let started = check(
        unsafe {
            AudioDeviceCreateIOProcID(agg, Some(io_proc), rt.cast(), NonNull::from(&mut proc_id))
        },
        "creating the IOProc",
    )
    .and_then(|_| {
        check(
            unsafe { AudioDeviceStart(agg, proc_id) },
            "starting the device",
        )
    });
    let mut route = Route {
        tap,
        tap_uuid,
        agg,
        proc_id,
        rt,
        procs: procs.iter().map(|p| p.id).collect(),
    };
    started?; // dropping `route` tears down what was made

    let stats_cap = stats.clone();
    let shown = describe(&procs);
    let needle = grab.to_string();
    let (mut quiet, mut warned, mut dropped, mut under) = (0, false, 0u64, 0u64);
    let regrab = move || {
        let now = find_processes(&needle).unwrap_or_default();
        let ids: Vec<AudioObjectID> = now.iter().map(|p| p.id).collect();
        if !ids.is_empty() && ids != route.procs {
            match route.retarget(&now) {
                Ok(()) => eprintln!("stemflow: now tapping {}", describe(&now)),
                Err(e) => eprintln!("stemflow: couldn't follow `{needle}`: {e:#}"),
            }
        }
        // Silence while the app says it's playing: no permission to record it.
        let peak = f32::from_bits(stats.peak.swap(0, Ordering::Relaxed));
        if peak == 0.0 && now.iter().any(is_playing) {
            quiet += 1;
        } else {
            quiet = 0;
        }
        if quiet == 3 && !warned {
            warned = true;
            eprintln!(
                "stemflow: `{needle}` is playing but the tap hears only silence. Allow \
                 recording for the app you started stemflow from: System Settings > Privacy & \
                 Security > Screen & System Audio Recording (the \"System Audio Recording \
                 Only\" list), then restart stemflow."
            );
        }
        let d = stats.dropped.load(Ordering::Relaxed);
        if d > dropped {
            eprintln!(
                "warning: dropped {} captured frames (reader fell behind)",
                d - dropped
            );
            dropped = d;
        }
        let u = stats.underruns.load(Ordering::Relaxed);
        if u > under + 10 {
            eprintln!("warning: {} output underruns", u - under);
            under = u;
        }
    };

    Ok(Io {
        capture: Box::new(Cap {
            rx: cap_rx,
            rs: Rs::new(rate, SAMPLE_RATE)?,
            chunk: rate as usize / 50,
            stats: stats_cap,
            told: false,
        }),
        playback: Box::new(Play {
            tx: play_tx,
            rs: Rs::new(SAMPLE_RATE, rate)?,
        }),
        regrab: Box::new(regrab),
        // The tap and the device are private to this process and go with it.
        cleanup: Box::new(|| {}),
        route: format!("tap on {shown}"),
        output: format!("{} ({rate} Hz)", out.name),
        hint: None,
    })
}
