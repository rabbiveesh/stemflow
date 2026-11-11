"""Build a blind A/B/C listening set from stemflow outputs.

For each out/ab/src/clipN.wav with clipN-demucs.wav and clipN-scnet.wav next to it,
writes out/ab/clipN/{mix,A,B,C}.flac where A/B/C are demucs, scnet and their 50/50
average (exactly what `--sep ensemble` produces, since the streaming is linear)
in a random order per clip. The answer key goes to out/ab/.key.json.

Usage: .venv/bin/python -I make_ab.py ../out/ab [--rebuild]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(sys.argv[1])
rebuild = "--rebuild" in sys.argv  # re-render audio, keep each clip's existing labels
src = root / "src"
key_path = root / ".key.json"
key = json.loads(key_path.read_text()) if key_path.exists() else {}
rng = random.Random()

for mix_path in sorted(src.glob("clip[0-9]*.wav")):
    name = mix_path.stem
    if "-" in name:
        continue
    d_path, s_path = src / f"{name}-demucs.wav", src / f"{name}-scnet.wav"
    if not (d_path.exists() and s_path.exists()):
        continue
    out = root / name
    if name in key and out.exists() and not rebuild:
        continue  # keep existing labels stable
    mix, sr = sf.read(mix_path, dtype="float32")
    d, _ = sf.read(d_path, dtype="float32")
    s, _ = sf.read(s_path, dtype="float32")
    n = min(len(mix), len(d), len(s))
    versions = {"demucs": d[:n], "scnet": s[:n], "ensemble": 0.5 * (d[:n] + s[:n])}
    if name in key:
        names = [key[name][L] for L in "ABC"]
    else:
        names = list(versions)
        rng.shuffle(names)
    out.mkdir(exist_ok=True)
    # Match RMS across versions so "louder" can't masquerade as "better", then one
    # shared gain keeps the loudest peak at -1 dBFS.
    rms = {k: float(np.sqrt(np.mean(v**2))) + 1e-9 for k, v in versions.items()}
    target = float(np.mean(list(rms.values())))
    matched = {k: v * (target / rms[k]) for k, v in versions.items()}
    gain = 0.89 / max(np.abs(v).max() for v in matched.values())
    for label, v in zip("ABC", names):
        sf.write(out / f"{label}.flac", matched[v] * gain, sr, subtype="PCM_16")
    sf.write(out / "mix.flac", mix[:n] * (0.89 / np.abs(mix[:n]).max()), sr, subtype="PCM_16")
    key[name] = dict(zip("ABC", names))
    print(f"{name}: wrote mix + A/B/C")

key_path.write_text(json.dumps(key, indent=2))
print(f"key: {key_path} (don't peek)")
