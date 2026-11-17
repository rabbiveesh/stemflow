//! Time-stretch and pitch-shift (Signalsmith Stretch, MIT; third_party/signalsmith).
//!
//! A thin safe wrapper over the C interface in third_party/signalsmith/wrapper.cpp,
//! declared by hand (no bindgen), so it builds anywhere a C++ compiler does. The
//! build script compiles the C++ (desktop: build.rs; Android: android/core/build.rs).

// Some of the interface is only used by the Android core.
#![allow(dead_code)]

use std::os::raw::{c_float, c_int};

#[repr(C)]
struct Handle {
    _private: [u8; 0],
}

extern "C" {
    fn signalsmith_stretch_create_preset_default(channels: c_int, rate: c_float) -> *mut Handle;
    fn signalsmith_stretch_create_preset_cheaper(channels: c_int, rate: c_float) -> *mut Handle;
    fn signalsmith_stretch_destroy(h: *mut Handle);
    fn signalsmith_stretch_reset(h: *mut Handle);
    fn signalsmith_stretch_input_latency(h: *mut Handle) -> usize;
    fn signalsmith_stretch_output_latency(h: *mut Handle) -> usize;
    fn signalsmith_stretch_seek(h: *mut Handle, input: *mut f32, frames: usize, rate: f64);
    fn signalsmith_stretch_set_transpose_factor(
        h: *mut Handle,
        factor: c_float,
        tonality_limit: c_float,
    );
    fn signalsmith_stretch_process(
        h: *mut Handle,
        input: *mut f32,
        in_frames: usize,
        output: *mut f32,
        out_frames: usize,
    );
}

/// One stretcher over `channels` interleaved channels.
pub struct Stretch {
    h: *mut Handle,
    channels: usize,
}

// The handle is only used through &mut self.
unsafe impl Send for Stretch {}

impl Stretch {
    /// `cheaper`: Signalsmith's lighter preset (shorter blocks, less CPU).
    pub fn new(channels: usize, rate: f32, cheaper: bool) -> Self {
        let h = unsafe {
            if cheaper {
                signalsmith_stretch_create_preset_cheaper(channels as c_int, rate)
            } else {
                signalsmith_stretch_create_preset_default(channels as c_int, rate)
            }
        };
        assert!(!h.is_null(), "signalsmith_stretch_create failed");
        Stretch { h, channels }
    }

    pub fn reset(&mut self) {
        unsafe { signalsmith_stretch_reset(self.h) }
    }

    /// Input frames the output lags behind (at the input's rate).
    pub fn input_latency(&self) -> usize {
        unsafe { signalsmith_stretch_input_latency(self.h) }
    }

    /// Output frames the output lags behind.
    pub fn output_latency(&self) -> usize {
        unsafe { signalsmith_stretch_output_latency(self.h) }
    }

    /// Frequency multiplier (2 = an octave up); 0 = no tonality limit.
    pub fn set_transpose(&mut self, factor: f32) {
        unsafe { signalsmith_stretch_set_transpose_factor(self.h, factor, 0.0) }
    }

    /// Pre-roll: input before the stream's position, so output starts without a fade-in.
    pub fn seek(&mut self, input: &[f32], rate: f64) {
        let n = input.len() / self.channels;
        unsafe { signalsmith_stretch_seek(self.h, input.as_ptr() as *mut f32, n, rate) }
    }

    /// Interleaved `input` frames in, `output` frames out; their ratio is the speed.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let (ni, no) = (input.len() / self.channels, output.len() / self.channels);
        unsafe {
            signalsmith_stretch_process(
                self.h,
                input.as_ptr() as *mut f32,
                ni,
                output.as_mut_ptr(),
                no,
            )
        }
    }
}

impl Drop for Stretch {
    fn drop(&mut self) {
        unsafe { signalsmith_stretch_destroy(self.h) }
    }
}
