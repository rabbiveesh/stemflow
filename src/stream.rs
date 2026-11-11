//! Sliding-window streaming on top of a fixed-window separator.
//!
//! Every `hop` samples we run the model on the most recent full window (all past
//! context, so the model never sees zero padding) and keep only a slice that sits
//! `lookahead` samples before the window's end, so each emitted sample had that much
//! future context. Consecutive slices are joined with a short linear crossfade.
//!
//!   window:  [ ........ past context ........ | out (hop) | lookahead ]
//!                                                         ^ xfade tail spills here
//!
//! Output is `lookahead` samples behind the input stream (`delay()`); in wall-clock
//! terms a sample waits up to `hop` more for its block to be computed, so live
//! latency ≈ lookahead + hop + compute time + device buffering.

use crate::model::Separator;
use anyhow::Result;

pub struct Mix {
    pub bass_gain: f32,
    /// Gain for everything-but-the-bass (mix minus bass). 0 = bass solo.
    pub rest_gain: f32,
}

pub struct Streamer<S: Separator> {
    sep: S,
    seg: usize,
    pub hop: usize,
    pub lookahead: usize,
    xfade: usize,
    win: Vec<f32>,  // planar [2, seg]
    tail: Vec<f32>, // planar [2, xfade]: previous step's output just past its slice
    have_tail: bool,
    pub mix: Mix,
}

impl<S: Separator> Streamer<S> {
    pub fn new(sep: S, hop: usize, lookahead: usize, xfade: usize, mix: Mix) -> Self {
        let seg = sep.window_len();
        assert!(
            hop > 0 && xfade <= lookahead,
            "crossfade must fit inside the lookahead"
        );
        assert!(
            hop + lookahead <= seg,
            "hop + lookahead must fit inside the model window ({seg})"
        );
        Streamer {
            sep,
            seg,
            hop,
            lookahead,
            xfade,
            win: vec![0.0; 2 * seg],
            tail: vec![0.0; 2 * xfade],
            have_tail: false,
            mix,
        }
    }

    /// Sample offset between input and output streams.
    pub fn delay(&self) -> usize {
        self.lookahead
    }

    /// Worst-case wall-clock latency in samples (excluding compute and device buffers).
    pub fn latency(&self) -> usize {
        self.hop + self.lookahead
    }

    /// Feed exactly `hop` interleaved stereo frames, get `hop` interleaved frames back
    /// (delayed by `delay()`).
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let (seg, hop, xf) = (self.seg, self.hop, self.xfade);
        assert_eq!(input.len(), 2 * hop);
        for ch in 0..2 {
            let w = &mut self.win[ch * seg..(ch + 1) * seg];
            w.copy_within(hop.., 0);
            for (i, s) in w[seg - hop..].iter_mut().enumerate() {
                *s = input[2 * i + ch];
            }
        }
        let bass = self.sep.separate(&self.win)?;

        let start = seg - self.lookahead - hop;
        let (bg, rg) = (self.mix.bass_gain, self.mix.rest_gain);
        let mut out = vec![0f32; 2 * hop];
        for ch in 0..2 {
            let b = &bass[ch * seg..(ch + 1) * seg];
            let m = &self.win[ch * seg..(ch + 1) * seg];
            let y = |i: usize| bg * b[i] + rg * (m[i] - b[i]);
            for i in 0..hop {
                let mut v = y(start + i);
                if self.have_tail && i < xf {
                    let a = (i as f32 + 0.5) / xf as f32;
                    v = self.tail[ch * xf + i] * (1.0 - a) + v * a;
                }
                out[2 * i + ch] = v;
            }
            for i in 0..xf {
                self.tail[ch * xf + i] = y(start + hop + i);
            }
        }
        self.have_tail = true;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity "separator": the stream must reproduce its input, delayed by latency().
    struct Ident(usize);
    impl Separator for Ident {
        fn window_len(&self) -> usize {
            self.0
        }
        fn separate(&mut self, mix: &[f32]) -> Result<Vec<f32>> {
            Ok(mix.to_vec())
        }
    }

    #[test]
    fn identity_is_delayed_copy() {
        let (hop, la) = (100, 37);
        let mut s = Streamer::new(
            Ident(1000),
            hop,
            la,
            20,
            Mix {
                bass_gain: 1.0,
                rest_gain: 0.0,
            },
        );
        let n = 20 * hop;
        let x: Vec<f32> = (0..2 * n).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut y = Vec::new();
        for c in x.chunks(2 * hop) {
            y.extend(s.process(c).unwrap());
        }
        let d = 2 * s.delay();
        for i in 0..(2 * n - d) {
            assert!((y[i + d] - x[i]).abs() < 1e-6, "mismatch at {i}");
        }
    }
}
