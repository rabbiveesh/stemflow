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

# One-time model export (CPU-only torch, ~1 GB venv). The default model is
# BS-RoFormer SW: its weights (enerjazzer/BS-ROFO-SW-Fixed on Hugging Face,
# config.yaml + model.ckpt) go in models/ext/bsrofo/, and the model class comes
# from ZFTurbo's Music-Source-Separation-Training.
tools/setup-export-env.sh
git clone --depth 1 https://github.com/ZFTurbo/Music-Source-Separation-Training tools/third_party/msst
(cd tools && .venv/bin/python -I export_bsrofo.py ../models/ext/bsrofo ../models)

cargo build --release --features cuda
./target/release/stemflow verify --sep bsrofo --device cuda   # Rust pipeline vs PyTorch
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
| `--balance B` | last used, else `-1` | one line: `-1` bass solo, `0` full mix, `1` no bass (play-along) |
| `--bass P` / `--band P` | | set each level separately, in % of how loud it sounds (50 = half as loud) |
| `--window S` | `3` | BS-RoFormer context; longer needs more GPU memory, not more latency |
| `--output SINK` | default sink | where the bass plays |
| `--stereo-in` | off | feed the model real stereo instead of a mono fold (see below) |
| `--target vocals` | `bass` | karaoke: solo / full / remove act on the vocals instead (`bsrofo` or `demucs`) |

Latency is roughly lookahead + hop + 0.3 s of buffer, so about 2.8 s by default.
Level changes are applied as audio leaves for the speakers, so they're heard within
about 0.1 s. The last levels are remembered in `~/.local/state/stemflow/levels`.

### Karaoke

`--target vocals` swaps BS-RoFormer SW's bass head for its vocals head. Everything
else stays the same: the balance runs from vocals solo to no vocals, and the
D-Bus `Bass` property and the "nobass" hotkey now mean the vocals (the `Target`
property says which stem it is). It costs the same as bass (0.62 s per 1 s hop on
an RTX 4060 laptop). Export the head once:

```bash
(cd tools && .venv/bin/python -I export_bsrofo.py ../models/ext/bsrofo ../models vocals)
./target/release/stemflow verify --sep bsrofo --device cuda --target vocals
./target/release/stemflow live --device cuda --target vocals --balance 1
```

## GNOME Shell extension (GNOME 42)

```bash
gnome/install.sh     # systemd user unit + extension; then reload the shell
```

A top-bar menu to start/stop stemflow and move the mix, plus hotkeys:

| key | |
|---|---|
| `Super+[` / `Super+]` | toward bass solo / toward play-along |
| `Super+\` | bass solo, or back to where you were |
| `Super+Shift+\` | no bass (play-along), or back |
| `Super+Alt+B` | start / stop |

Scrolling on the icon moves the mix too. "Separate sliders" in the menu gives the
bass and the band a slider each. stemflow also serves `org.stemflow.Control` on the
session bus (`busctl --user introspect org.stemflow.Control /org/stemflow/Control`),
and the unit takes extra flags from `~/.config/stemflow/env`
(`STEMFLOW_ARGS="--grab firefox"`).

To try settings offline and see whether they'd keep up live:

```bash
ffmpeg -i track.flac -ar 44100 track.wav
./target/release/stemflow file --device cuda track.wav bass.wav
```

On an RTX 4060 Laptop GPU, BS-RoFormer takes 0.62 s per 3 s window, which keeps up
with a 1 s hop. Windows of 5 s and up run out of the 8 GB in fp32.

## Notes

- **Snap Spotify on Ubuntu 22.04 with PulseAudio:** recent snaps talk to PipeWire
  directly and fail with "can't play current song" when PipeWire isn't the audio
  server. Either switch to `pipewire-pulse`, or run `sudo snap disconnect spotify:pipewire`.
- Only the separated audio is played back; nothing is recorded or saved in live mode.
  Saving streamed audio would go against most streaming services' terms.
- **Input is folded to mono by default.** The models were trained on pop mixes with the
  bass in the centre, and they miss bass panned to one side. On wide 1960s stereo
  (the Oscar Peterson Trio's *Night Train*), htdemucs_ft returned silence for a whole
  30 s clip; with mono input it found the bass in every second.
- Bowed (arco) bass is mostly missed by both models; there's very little of it in
  their training data.
- The model is swappable behind the `Separator` trait in `src/model.rs`.

## Other models

BS-RoFormer SW won blind listening rounds against all of these. `--sep demucs`
runs htdemucs_ft (export with `export_htdemucs.py`; 0.33 s per 7.8 s window).

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
