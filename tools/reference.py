"""Offline reference: stock demucs apply_model with the htdemucs_ft bass specialist,
then score stemflow outputs against it (SDR, dB; higher = closer to offline quality).

Usage: .venv/bin/python -I reference.py mix.wav ref_bass.wav [stemflow_out.wav ...]
"""
import sys

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

mix_path, ref_path, *candidates = sys.argv[1:]
mix, sr = sf.read(mix_path, dtype="float32", always_2d=True)
assert sr == 44100
try:
    ref, _ = sf.read(ref_path, dtype="float32", always_2d=True)
except Exception:
    bag = get_model("htdemucs_ft")
    i = bag.sources.index("bass")
    m = bag.models[int(torch.tensor(bag.weights)[:, i].argmax())].eval()
    x = torch.from_numpy(mix.T.copy())[None]
    with torch.no_grad():
        ref = apply_model(m, x, overlap=0.25, split=True, progress=True, num_workers=0)[0, i].numpy().T
    sf.write(ref_path, ref, sr, subtype="FLOAT")


def sdr(est, ref):
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    return 10 * np.log10(np.sum(ref**2) / (np.sum((ref - est) ** 2) + 1e-12))


for c in candidates:
    est, _ = sf.read(c, dtype="float32", always_2d=True)
    print(f"{c}: SDR vs offline demucs = {sdr(est, ref):.2f} dB")
