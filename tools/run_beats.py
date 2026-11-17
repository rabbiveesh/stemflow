"""Compare beat trackers on the A/B clips and render click tracks to listen to.

For each clip and tracker, writes <out>/<clip>-<tracker>.json (beat and downbeat
times in seconds; for Beat This!, also the per-frame beat/downbeat logits at 50 fps
so a page can redo the peak-picking with a meter hint or other settings) and <out>/<clip>-<tracker>.flac: the mix at -6 dB with a click on
every beat (high click on downbeats where the tracker gives them).

Trackers:
  bt       Beat This! final0 (CPJKU, ISMIR 2024): beats + downbeats, no meter assumed
  btsmall  Beat This! small0: the same, a tenth of the size
  librosa  librosa.beat.beat_track: classic onset/tempo baseline, beats only

Usage: .venv-gpu/bin/python -I run_beats.py <out_dir> clip1.wav [clip2.wav ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

out_dir = Path(sys.argv[1])
inputs = [Path(p) for p in sys.argv[2:]]
out_dir.mkdir(parents=True, exist_ok=True)


def click(sr, freq, dur=0.04):
    t = np.arange(int(sr * dur)) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t * 90)).astype(np.float32)


def render(mix, sr, beats, downbeats):
    y = 0.5 * mix.mean(axis=1)
    lo, hi = click(sr, 1000), click(sr, 1700)
    down = {round(d, 2) for d in downbeats}
    for b in beats:
        c = hi if round(b, 2) in down else lo
        i = int(b * sr)
        n = min(len(c), len(y) - i)
        if n > 0:
            y[i:i + n] += 0.45 * c[:n]
    return np.stack([y, y], 1) * (0.95 / max(1e-9, np.abs(y).max()))


def tempo(beats):
    d = np.diff(beats)
    return float(60 / np.median(d)) if len(d) else 0.0


def meter(beats, downbeats):
    """Beats per bar, from how many beats fall between consecutive downbeats."""
    if len(downbeats) < 2:
        return []
    b = np.asarray(beats)
    return [int(((b >= a - 0.03) & (b < z - 0.03)).sum()) for a, z in zip(downbeats, downbeats[1:])]


trackers = {}
try:
    import torch
    from beat_this.inference import File2Beats

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from beat_this.inference import Audio2Frames
    from beat_this.preprocessing import load_audio

    for name, ck in (("bt", "final0"), ("btsmall", "small0")):
        f2b = File2Beats(checkpoint_path=ck, device=dev, dbn=False)
        a2f = Audio2Frames(checkpoint_path=ck, device=dev)

        def run(p, f=f2b, a=a2f):
            beats, downbeats = f(str(p))
            sig, sr = load_audio(str(p))
            bl, dl = a(sig, sr)
            r = lambda x: [round(float(v), 2) for v in x.float().cpu().numpy().ravel()]  # noqa: E731
            return list(beats), list(downbeats), dict(fps=50, beat_logits=r(bl), downbeat_logits=r(dl))

        trackers[name] = run
except ImportError as e:
    print("beat_this not installed, skipping:", e)

import librosa  # noqa: E402


def lib(p):
    y, sr = librosa.load(str(p), sr=22050, mono=True)
    _, frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return list(map(float, frames)), [], {}


trackers["librosa"] = lib

summary = {}
for p in inputs:
    mix, sr = sf.read(p, dtype="float32", always_2d=True)
    for name, fn in trackers.items():
        beats, downbeats, frames = fn(p)
        beats, downbeats = [float(b) for b in beats], [float(d) for d in downbeats]
        info = dict(beats=beats, downbeats=downbeats, bpm=round(tempo(beats), 1), meter=meter(beats, downbeats), **frames)
        (out_dir / f"{p.stem}-{name}.json").write_text(json.dumps(info))
        sf.write(out_dir / f"{p.stem}-{name}.flac", render(mix, sr, beats, downbeats), sr, subtype="PCM_16")
        bars = info["meter"]
        common = max(set(bars), key=bars.count) if bars else "-"
        summary[f"{p.stem}-{name}"] = info["bpm"]
        print(f"{p.stem:6} {name:8} {info['bpm']:6.1f} bpm  {len(beats):3} beats  "
              f"{len(downbeats):2} downbeats  beats/bar mostly {common}  {bars}", flush=True)
