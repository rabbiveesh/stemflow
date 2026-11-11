# stemflow

Live bass isolation for whatever your Linux desktop is playing. Route Spotify (or
any app) through stemflow and hear just the bass line, a couple of seconds behind,
so you can learn it by ear.

It runs the bass specialist from [Demucs](https://github.com/facebookresearch/demucs)
`htdemucs_ft` through ONNX Runtime from Rust, with the STFT/iSTFT reimplemented to
match PyTorch's output (checked in CI).

```
app ──▶ [stemflow null sink] ──monitor──▶ stemflow ──▶ bass ──▶ your real output
                                           │
                       every hop: run the model on the last 7.8 s,
                       keep a slice `lookahead` before the end,
                       crossfade it onto the previous slice
```

## Requirements

- Linux with PipeWire (`pipewire-pulse`) or PulseAudio, plus `libpulse-dev` to build
- Rust (stable)
- [`uv`](https://github.com/astral-sh/uv) and `gh` for the one-time setup below
- An NVIDIA GPU for live use. On CPU one model run takes ~2.5 s, so live mode
  needs `--hop 3` or more.

## Setup

```bash
# ONNX Runtime from Microsoft's GitHub releases (+ CUDA 12 / cuDNN 9 libs for cuda)
tools/fetch-ort.sh cuda        # or: tools/fetch-ort.sh cpu

# One-time model export (CPU-only torch, ~1 GB venv)
tools/setup-export-env.sh
(cd tools && .venv/bin/python -I export_htdemucs.py ../models)

cargo build --release --features cuda
./target/release/stemflow verify --device cuda   # Rust pipeline vs PyTorch
```

## Use

```bash
./target/release/stemflow live --device cuda
```

Press play in Spotify. stemflow moves Spotify's stream into its sink, and Ctrl-C
removes the sink, which puts Spotify back on your normal output.

| flag | default | |
|---|---|---|
| `--grab NAME` | `spotify` | which app's streams to capture (substring of the app name) |
| `--hop S` | `1.0` | seconds of new audio per model run; compute must stay under this |
| `--lookahead S` | `1.5` | future context per sample; more = cleaner but laggier |
| `--rest-gain G` | `0` | mix the rest of the band back in quietly (e.g. `0.15`) to keep your place |
| `--output SINK` | default sink | where the bass plays |

Latency is roughly lookahead + hop + 0.3 s of buffer, so about 2.8 s by default.

To try settings offline and see whether they'd keep up live:

```bash
ffmpeg -i track.flac -ar 44100 track.wav
./target/release/stemflow file --device cuda track.wav bass.wav
```

On an RTX 4060 Laptop GPU, one 7.8 s window takes 0.33 s at the median and 0.51 s at
worst, so a 1 s hop has plenty of headroom.

## Notes

- **Snap Spotify on Ubuntu 22.04 with PulseAudio:** recent snaps talk to PipeWire
  directly and fail with "can't play current song" when PipeWire isn't the audio
  server. Either switch to `pipewire-pulse`, or run `sudo snap disconnect spotify:pipewire`.
- Only the separated audio is played back; nothing is recorded or saved in live mode.
  Saving streamed audio would go against most streaming services' terms.
- The model is swappable behind the `Separator` trait in `src/model.rs`.

## Other models

`--sep scnet` runs SCNet XL IHF (from ZFTurbo's Music-Source-Separation-Training,
Multisong bass SDR 11.94 vs 11.96 for htdemucs_ft). It takes 1.44 s per 11 s window
on an RTX 4060, so it needs `--hop 2` or more. Export it with:

```bash
git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training tools/third_party/msst
(cd tools && .venv/bin/python -I export_scnet.py ../models)
./target/release/stemflow verify --sep scnet --device cuda
```

`--sep ensemble` averages both models. In fp32 the two need about 3 GB + 5 GB of GPU
memory, so it doesn't fit on an 8 GB card yet. Since streaming is linear,
`tools/make_ab.py` builds the exact ensemble output offline from the two single-model
runs, mixed into a blind A/B/C listening set.

## License

MIT. Demucs and its pretrained weights are by Meta AI Research, under MIT. This repo
doesn't redistribute the weights; the export script downloads them.
