"""Build a blind listening set from any stems sitting next to the A/B clips.

For each <src>/clipN.wav with every <src>/clipN-<suffix>.wav present, writes
<out>/clipN/{mix,A,B,...}.flac with the stems in a random order per clip,
RMS-matched, plus the answer key in <out>/.key.json. Existing labels are kept.

Usage: .venv/bin/python -I make_blind.py <src> <out> demucs bsrofo banquet [...]
"""
import json
import random
import string
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

src, root, suffixes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
letters = string.ascii_uppercase[: len(suffixes)]
root.mkdir(parents=True, exist_ok=True)
key_path = root / ".key.json"
key = json.loads(key_path.read_text()) if key_path.exists() else {}
rng = random.Random()

for mix_path in sorted(src.glob("clip[0-9]*.wav")):
    name = mix_path.stem
    if "-" in name:
        continue
    paths = {s: src / f"{name}-{s}.wav" for s in suffixes}
    if not all(p.exists() for p in paths.values()):
        print(f"{name}: skipped (missing {[s for s, p in paths.items() if not p.exists()]})")
        continue
    mix, sr = sf.read(mix_path, dtype="float32", always_2d=True)
    stems = {s: sf.read(p, dtype="float32", always_2d=True)[0] for s, p in paths.items()}
    n = min([len(mix)] + [len(v) for v in stems.values()])
    stems = {s: np.repeat(v[:n], 2, axis=1) if v.shape[1] == 1 else v[:n, :2] for s, v in stems.items()}
    dead = [s for s, v in stems.items() if np.sqrt(np.mean(v**2)) < 10 ** (-70 / 20)]
    if dead:
        print(f"{name}: skipped ({dead} silent; level-matching would blast its noise floor)")
        continue
    old = key.get(name)
    names = [old[L] for L in letters] if old and set(old.values()) == set(suffixes) else rng.sample(suffixes, len(suffixes))
    rms = {s: float(np.sqrt(np.mean(v**2))) + 1e-9 for s, v in stems.items()}
    target = float(np.mean(list(rms.values())))
    matched = {s: v * (target / rms[s]) for s, v in stems.items()}
    gain = 0.89 / max(np.abs(v).max() for v in matched.values())
    out = root / name
    out.mkdir(exist_ok=True)
    for L, s in zip(letters, names):
        sf.write(out / f"{L}.flac", matched[s] * gain, sr, subtype="PCM_16")
    sf.write(out / "mix.flac", mix[:n] * (0.89 / np.abs(mix[:n]).max()), sr, subtype="PCM_16")
    key[name] = dict(zip(letters, names))
    print(f"{name}: wrote {', '.join(letters)}  (rms dB: {', '.join(f'{20*np.log10(rms[s]):.0f}' for s in suffixes)})")

key_path.write_text(json.dumps(key, indent=2))
print(f"key: {key_path} (don't peek)")
