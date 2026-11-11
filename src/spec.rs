//! STFT / iSTFT that bit-for-bit mirror demucs' `HTDemucs._spec` / `_ispec`
//! (torch.stft/istft with n_fft=4096, hop=1024, periodic Hann, normalized=True,
//! center=True, reflect padding, plus demucs' own extra 3/4-hop reflect padding).

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const NFFT: usize = 4096;
pub const HOP: usize = NFFT / 4;
pub const BINS: usize = NFFT / 2; // demucs drops the Nyquist bin
const PAD: usize = HOP / 2 * 3;

pub struct Stft {
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    norm: f32,
}

/// torch-style reflect padding (edge sample not repeated).
fn reflect_pad(x: &[f32], left: usize, right: usize) -> Vec<f32> {
    let n = x.len();
    assert!(left < n && right < n, "reflect pad larger than signal");
    let mut out = Vec::with_capacity(n + left + right);
    out.extend((1..=left).rev().map(|i| x[i]));
    out.extend_from_slice(x);
    out.extend((0..right).map(|j| x[n - 2 - j]));
    out
}

pub fn frames_for(len: usize) -> usize {
    len.div_ceil(HOP)
}

impl Stft {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let window = (0..NFFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / NFFT as f32).cos())
            .collect();
        Stft {
            fwd: planner.plan_fft_forward(NFFT),
            inv: planner.plan_fft_inverse(NFFT),
            window,
            norm: 1.0 / (NFFT as f32).sqrt(),
        }
    }

    /// Spectrogram of one channel, written as `out[bin * frames + frame]`.
    pub fn spec(&self, x: &[f32], out: &mut [Complex32]) {
        let le = frames_for(x.len());
        assert_eq!(out.len(), BINS * le);
        // demucs pad, then torch center pad.
        let xp = reflect_pad(x, PAD, PAD + le * HOP - x.len());
        let xc = reflect_pad(&xp, NFFT / 2, NFFT / 2);
        let mut buf = self.fwd.make_input_vec();
        let mut spec = self.fwd.make_output_vec();
        for f in 0..le {
            let start = (f + 2) * HOP; // demucs keeps frames [2, 2 + le)
            for (i, b) in buf.iter_mut().enumerate() {
                *b = xc[start + i] * self.window[i];
            }
            self.fwd.process(&mut buf, &mut spec).expect("fft");
            for bin in 0..BINS {
                out[bin * le + f] = spec[bin] * self.norm;
            }
        }
    }

    /// Inverse of `spec` for one channel: `z[bin * le + frame]` -> `length` samples.
    pub fn ispec(&self, z: &[Complex32], length: usize, out: &mut [f32]) {
        let le = frames_for(length);
        assert_eq!(z.len(), BINS * le);
        assert_eq!(out.len(), length);
        let nframes = le + 4; // demucs zero-pads 2 frames each side
        let total = NFFT + HOP * (nframes - 1);
        let mut y = vec![0f32; total];
        let mut env = vec![0f32; total];
        let mut spec = self.inv.make_input_vec();
        let mut frame = self.inv.make_output_vec();
        for f in 0..nframes {
            let off = f * HOP;
            for i in 0..NFFT {
                env[off + i] += self.window[i] * self.window[i];
            }
            if f < 2 || f >= le + 2 {
                continue;
            }
            for bin in 0..BINS {
                spec[bin] = z[bin * le + (f - 2)];
            }
            spec[0].im = 0.0;
            spec[BINS] = Complex32::new(0.0, 0.0);
            self.inv.process(&mut spec, &mut frame).expect("ifft");
            for i in 0..NFFT {
                y[off + i] += frame[i] * self.norm * self.window[i];
            }
        }
        let start = NFFT / 2 + PAD;
        for (i, o) in out.iter_mut().enumerate() {
            *o = y[start + i] / env[start + i];
        }
    }
}

// Correctness vs PyTorch is checked by `stemflow verify` (demucs' spec/ispec pair is
// deliberately not a perfect inverse, so a roundtrip unit test would be meaningless).

/// Plain `torch.stft` / `torch.istft` with `center=True` (reflect pad),
/// `normalized=True` and *no window* (rectangular), as SCNet calls them.
/// Keeps all `NFFT / 2 + 1` bins. Layout is `[bin * frames + frame]`.
pub struct PlainStft {
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    norm: f32,
}

pub const FULL_BINS: usize = NFFT / 2 + 1;

impl PlainStft {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        PlainStft {
            fwd: planner.plan_fft_forward(NFFT),
            inv: planner.plan_fft_inverse(NFFT),
            norm: 1.0 / (NFFT as f32).sqrt(),
        }
    }

    pub fn frames_for(len: usize) -> usize {
        1 + len / HOP
    }

    pub fn stft(&self, x: &[f32], out: &mut [Complex32]) {
        let frames = Self::frames_for(x.len());
        assert_eq!(out.len(), FULL_BINS * frames);
        let xc = reflect_pad(x, NFFT / 2, NFFT / 2);
        let mut buf = self.fwd.make_input_vec();
        let mut spec = self.fwd.make_output_vec();
        for f in 0..frames {
            buf.copy_from_slice(&xc[f * HOP..f * HOP + NFFT]);
            self.fwd.process(&mut buf, &mut spec).expect("fft");
            for bin in 0..FULL_BINS {
                out[bin * frames + f] = spec[bin] * self.norm;
            }
        }
    }

    /// `torch.istft(..., length=None)`: returns `HOP * (frames - 1)` samples.
    pub fn istft(&self, z: &[Complex32], frames: usize, out: &mut [f32]) {
        assert_eq!(z.len(), FULL_BINS * frames);
        let len = HOP * (frames - 1);
        assert_eq!(out.len(), len);
        let total = NFFT + HOP * (frames - 1);
        let mut y = vec![0f32; total];
        let mut env = vec![0f32; total];
        let mut spec = self.inv.make_input_vec();
        let mut frame = self.inv.make_output_vec();
        for f in 0..frames {
            for bin in 0..FULL_BINS {
                spec[bin] = z[bin * frames + f];
            }
            spec[0].im = 0.0;
            spec[FULL_BINS - 1].im = 0.0;
            self.inv.process(&mut spec, &mut frame).expect("ifft");
            let off = f * HOP;
            for i in 0..NFFT {
                y[off + i] += frame[i] * self.norm;
                env[off + i] += 1.0;
            }
        }
        let start = NFFT / 2;
        for (i, o) in out.iter_mut().enumerate() {
            *o = y[start + i] / env[start + i];
        }
    }
}
