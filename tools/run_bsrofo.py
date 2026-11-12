"""Run BS-Roformer SW (6-stem) offline and keep the bass stem.

Input is folded to mono first, the same as stemflow's default. Chunks of the
model's native size are overlap-added with a triangular window.

Usage: .venv-gpu/bin/python -I run_bsrofo.py <model_dir> <out_suffix> in1.wav [in2.wav ...]
  Runs on CUDA in fp16 when available; CPU works but takes minutes per chunk.
  model_dir holds config.yaml + model.ckpt; writes <in>-<out_suffix>.wav next to each input.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "third_party" / "msst"))
from models.bs_roformer.bs_roformer import BSRoformer  # noqa: E402


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))

model_dir, suffix, inputs = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
cfg = yaml.load((model_dir / "config.yaml").read_text(), Loader=Loader)
mcfg = dict(cfg["model"])
mcfg["flash_attn"] = False
model = BSRoformer(**mcfg)
state = torch.load(model_dir / "model.ckpt", map_location="cpu", weights_only=False)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev).eval()
stems = cfg["training"]["instruments"]
bass_idx = stems.index("bass")
chunk = cfg["audio"]["chunk_size"]
hop = chunk // 2
torch.set_num_threads(max(1, torch.get_num_threads()))

for path in inputs:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 44100, sr
    mono = x.mean(axis=1, keepdims=True)
    x = np.repeat(mono, 2, axis=1).T  # (2, n)
    n = x.shape[1]
    pad = hop
    total = n + 2 * pad
    total += (-(total - chunk)) % hop if total > chunk else chunk - total
    xp = np.zeros((2, total), np.float32)
    xp[:, pad:pad + n] = x
    out = np.zeros_like(xp)
    wsum = np.zeros(total, np.float32)
    win = np.concatenate([np.linspace(0, 1, hop, endpoint=False), np.linspace(1, 0, chunk - hop, endpoint=False)]).astype(np.float32) + 1e-3
    with torch.inference_mode():
        for start in range(0, total - chunk + 1, hop):
            seg = torch.from_numpy(xp[None, :, start:start + chunk]).to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                y = model(seg)[0, bass_idx].float().cpu().numpy()
            out[:, start:start + chunk] += y * win
            wsum[start:start + chunk] += win
    bass = (out / wsum)[:, pad:pad + n].T
    dst = path.with_name(f"{path.stem}-{suffix}.wav")
    sf.write(dst, bass, sr, subtype="FLOAT")
    print(f"{dst.name}: done", flush=True)
