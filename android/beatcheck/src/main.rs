//! Check src/beats.rs against PyTorch and time it.
//!
//!   stemflow-beatcheck <libonnxruntime.so> <model.onnx> <clip.wav> [ref_dir] [threads]
//!
//! With `ref_dir` (from tools/export_beat_this.py + a dump to raw f32: audio22k.f32,
//! spect.f32, beat.f32, downbeat.f32) it compares the mel spectrogram and the logits;
//! then it plays the clip through the live Tracker (100 ms pushes, analysis every 3 s)
//! and prints each grid.

#[path = "../../../src/beats.rs"]
mod beats;

use anyhow::Result;
use beats::BeatModel;
use std::path::Path;
use std::time::Instant;

fn raw(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let env = beats::ort_env(Path::new(&a[1]))?;
    let threads = a.get(5).map_or(4, |t| t.parse().unwrap());
    let t = Instant::now();
    let mut net = beats::BeatNet::load(env, Path::new(&a[2]), threads, false)?;
    println!("model loaded in {:.2} s", t.elapsed().as_secs_f32());
    let mel = beats::Mel::new();

    if let Some(dir) = a.get(4).filter(|d| !d.is_empty()) {
        let dir = Path::new(dir);
        let x = raw(&dir.join("audio22k.f32"))?;
        let t = Instant::now();
        let s = mel.spect(&x[..beats::WINDOW]);
        println!("mel {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        let r = raw(&dir.join("spect.f32"))?;
        // the last few frames see different padding (torch had the audio after them)
        let k = (beats::FRAMES - 4) * 128;
        println!("mel vs torch: max |d| {:.2e} (values up to {:.1})", maxdiff(&s[..k], &r[..k]),
                 r.iter().fold(0f32, |m, v| m.max(*v)));
        let t = Instant::now();
        let (b, d) = net.run(&r)?;
        println!("model on torch's mel: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        for _ in 0..3 {
            let t = Instant::now();
            net.run(&r)?;
            println!("  again: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        let (rb, rd) = (raw(&dir.join("beat.f32"))?, raw(&dir.join("downbeat.f32"))?);
        println!("logits vs torch: beat {:.2e}, downbeat {:.2e}", maxdiff(&b, &rb), maxdiff(&d, &rd));
        let (b2, _) = net.run(&s)?;
        for k in [100usize, 500, 1000, 1400, 1480, 1495] {
            let i = (0..k).max_by(|&i, &j| (b2[i] - b[i]).abs().total_cmp(&(b2[j] - b[j]).abs())).unwrap();
            println!("logits, our mel vs torch's mel, frames < {k}: max {:.2e} at {i}", (b2[i] - b[i]).abs());
        }
        let dec = beats::decode(&rb, &rd, &beats::Settings::default());
        println!("decode(torch logits): {:.1} bpm, meter {}, first bar {}, {} beats, first {:?}",
                 dec.bpm, dec.meter, dec.first_bar, dec.beats.len(), &dec.beats[..4.min(dec.beats.len())]);
    }

    // live: the clip through the Tracker
    let mut r = hound::WavReader::open(&a[3])?;
    let spec = r.spec();
    anyhow::ensure!(spec.sample_rate == 44100, "need 44.1 kHz");
    let xs: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let s = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|v| v.map(|v| v as f32 / s)).collect::<Result<_, _>>()?
        }
    };
    let stereo: Vec<f32> = if spec.channels == 1 { xs.iter().flat_map(|&v| [v, v]).collect() } else { xs };
    let tr = beats::Tracker::new(Box::new(net));
    let chunk = 4410 * 2;
    let mut since = 0;
    for (i, c) in stereo.chunks(chunk).enumerate() {
        tr.push(c);
        since += c.len() / 2;
        if since >= 3 * 44100 {
            since = 0;
            if tr.analyze()? {
                let g = tr.grid().unwrap();
                let now = (i + 1) as f64 * 0.1;
                let last = g.beats.last().copied().unwrap_or(0.0) / 44100.0;
                let info = tr.info();
                println!("t={now:5.1}s  {:5.1} bpm  {}/bar  {:3} beats  last beat {:.2}s  analysis {:.0} ms",
                         info.bpm, info.meter, info.beats, last, info.analysis_s * 1e3);
            }
        }
    }
    if let Some(g) = tr.grid() {
        let bars: Vec<String> = g.beats.iter().zip(&g.bar).filter(|(_, &d)| d).take(8)
            .map(|(b, _)| format!("{:.2}", b / 44100.0)).collect();
        println!("bar lines: {}", bars.join(" "));
    }
    Ok(())
}
