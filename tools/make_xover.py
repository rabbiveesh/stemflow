"""Round-2 blind set: htdemucs_ft vs crossover blends (Demucs lows + SCNet highs).

Round 1 found Demucs cleanest but thin up top, SCNet better up top but bleedier.
This splits the two stems at a crossover frequency with complementary masks
(low + high == input exactly), so each blend is Demucs below fc and SCNet above.

For each out/ab/src/clipN.wav with -demucs/-scnet stems, writes
<out>/clipN/{mix,A,B,C}.flac, where A/B/C are demucs and blends at the given
crossovers, shuffled per clip, RMS-matched. Key goes to <out>/.key.json.

Usage: .venv/bin/python -I make_xover.py ../out/ab/src ../out/ab2 [150 300]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

src, root = Path(sys.argv[1]), Path(sys.argv[2])
cross = [float(x) for x in sys.argv[3:]] or [150.0, 300.0]
root.mkdir(parents=True, exist_ok=True)
key_path = root / ".key.json"
key = json.loads(key_path.read_text()) if key_path.exists() else {}
rng = random.Random()


def lowpass_mask(n, sr, fc, order=4):
    # Butterworth magnitude squared, i.e. Linkwitz-Riley magnitude; zero phase.
    f = np.fft.rfftfreq(n, 1 / sr)
    return 1.0 / (1.0 + (f / fc) ** (2 * order))


def blend(d, s, sr, fc):
    n = len(d)
    h = lowpass_mask(n, sr, fc)[:, None]
    D, S = np.fft.rfft(d, axis=0), np.fft.rfft(s, axis=0)
    return np.fft.irfft(D * h + S * (1 - h), n=n, axis=0).astype(np.float32)


for mix_path in sorted(src.glob("clip[0-9]*.wav")):
    name = mix_path.stem
    if "-" in name:
        continue
    d_path, s_path = src / f"{name}-demucs.wav", src / f"{name}-scnet.wav"
    if not (d_path.exists() and s_path.exists()):
        continue
    mix, sr = sf.read(mix_path, dtype="float32")
    d, _ = sf.read(d_path, dtype="float32")
    s, _ = sf.read(s_path, dtype="float32")
    n = min(len(mix), len(d), len(s))
    d, s = d[:n], s[:n]
    versions = {"demucs": d}
    for fc in cross:
        versions[f"xover{int(fc)}"] = blend(d, s, sr, fc)
    names = [key[name][L] for L in "ABC"] if name in key else rng.sample(list(versions), 3)
    rms = {k: float(np.sqrt(np.mean(v**2))) + 1e-9 for k, v in versions.items()}
    target = float(np.mean(list(rms.values())))
    matched = {k: v * (target / rms[k]) for k, v in versions.items()}
    gain = 0.89 / max(np.abs(v).max() for v in matched.values())
    out = root / name
    out.mkdir(exist_ok=True)
    for label, v in zip("ABC", names):
        sf.write(out / f"{label}.flac", matched[v] * gain, sr, subtype="PCM_16")
    key[name] = dict(zip("ABC", names))
    print(f"{name}: wrote A/B/C")

key_path.write_text(json.dumps(key, indent=2))
print(f"key: {key_path} (don't peek)")
