"""Export Beat This! (CPJKU, MIT) to LiteRT (.tflite) for the phone.

Same graph as tools/export_beat_this.py's ONNX: input `spect` [1, 1500, 128] (log-mel,
50 fps, 30 s), outputs `beat` and `downbeat` logits [1, 1500]. The mel and the decoder
stay outside, in Rust (src/beats.rs).

The checkpoints are the ones beat_this downloads: ~/.cache/torch/hub/checkpoints/
beat_this-{small0,final0}.ckpt.

Usage: python -I export_beat_this_litert.py OUT_DIR BEAT_THIS_SITE_DIR [REFS_DIR]
  BEAT_THIS_SITE_DIR  a site-packages that has beat_this (tools/.venv-gpu/lib/python3.10/
                      site-packages in the main checkout); only beat_this comes from it
  REFS_DIR            models/beats of the main checkout: ref_beat_this_<name>/{spect,beat,
                      downbeat}.f32, to print the LiteRT-vs-PyTorch error
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

out_dir = Path(sys.argv[1])
sys.path.append(sys.argv[2])
refs = Path(sys.argv[3]) if len(sys.argv) > 3 else None
FRAMES = 1500

import inspect  # noqa: E402

# Not beat_this.inference: it imports soxr, which that other venv built for its own Python.
from beat_this.model.beat_tracker import BeatThis  # noqa: E402


def load_model(name):
    ck = torch.load(Path.home() / f".cache/torch/hub/checkpoints/beat_this-{name}.ckpt", map_location="cpu", weights_only=True)
    keep = set(inspect.signature(BeatThis).parameters)
    m = BeatThis(**{k: v for k, v in ck["hyper_parameters"].items() if k in keep})
    m.load_state_dict({k.removeprefix("model."): v for k, v in ck["state_dict"].items()})
    return m


def rotary(rot, n):
    """cos and sin of the rotary angles for n positions, [n, dim_head], as constants."""
    with torch.no_grad():
        f = rot.forward(torch.arange(n, dtype=torch.float32)).contiguous().clone()
    return f.cos(), f.sin()


def rotate(x, cs):
    # x * cos + rotate_half(x) * sin, where rotate_half maps each pair (a, b) to (-b, a)
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
    return x * cs[0] + torch.stack((-b, a), -1).flatten(-2) * cs[1]


class Wrap(torch.nn.Module):
    """BeatThis.forward with the library's attention written out: rotary_embedding_torch
    and SumHead wrap their maths in torch.autocast, which torch.export's decompositions
    refuse ("expected compiled_fn to be GraphModule"), and the rotary angles become
    constants per sequence length. Same weights, same numbers."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        rot = m.transformer_blocks.layers[0][0].rotary_embed
        # sequence lengths: time, and the frequency counts of the three frontend blocks
        self.cs = {n: rotary(rot, n) for n in (FRAMES, 32, 16, 8)}

    def attn(self, a, x):
        h = a.heads
        y = a.norm(x)
        q, k, v = a.to_qkv(y).unflatten(-1, (3, h, -1)).permute(2, 0, 3, 1, 4).unbind(0)  # b h n d
        cs = self.cs[x.shape[1]]
        o = F.scaled_dot_product_attention(rotate(q, cs), rotate(k, cs), v)
        o = o * a.to_gates(y).transpose(1, 2).unsqueeze(-1).sigmoid()
        return a.to_out(o.transpose(1, 2).flatten(2))

    def forward(self, spect):
        m = self.m
        x = m.frontend.stem(spect)
        for blk in m.frontend.blocks:
            p = blk.partial
            b, c, f, t = x.shape
            x = x.permute(0, 3, 2, 1).reshape(b * t, f, c)
            x = x + self.attn(p.attnF, x)
            x = x + p.ffF(x)
            x = x.reshape(b, t, f, c).transpose(1, 2).reshape(b * f, t, c)
            x = x + self.attn(p.attnT, x)
            x = x + p.ffT(x)
            x = x.reshape(b, f, t, c).permute(0, 3, 1, 2)
            x = blk.activation(blk.norm(blk.conv2d(x)))
        x = m.frontend.linear(m.frontend.concat(x))
        for a, ff in m.transformer_blocks.layers:
            x = self.attn(a, x) + x
            x = ff(x) + x
        y = m.task_heads.beat_downbeat_lin(m.transformer_blocks.norm(x))
        return y[..., 0] + y[..., 1], y[..., 1]


import litert_torch  # noqa: E402
from ai_edge_litert.interpreter import Interpreter  # noqa: E402

out_dir.mkdir(parents=True, exist_ok=True)
for ck in sys.argv[4:] or ("small0", "final0"):
    stock = load_model(ck).eval().requires_grad_(False)
    model = Wrap(stock).eval()
    x = torch.randn(1, FRAMES, 128)
    y = stock(x)
    rewrite = max((a - y[n]).abs().max().item() for a, n in zip(model(x), ("beat", "downbeat")))
    path = out_dir / f"beat_this_{ck}_{FRAMES}.tflite"
    litert_torch.convert(model, (torch.randn(1, FRAMES, 128),)).export(str(path))

    ref = refs / f"ref_beat_this_{ck}" if refs else None
    if ref and ref.exists():
        spect = np.fromfile(ref / "spect.f32", dtype="<f4").reshape(1, FRAMES, 128)
        want = [np.fromfile(ref / f"{n}.f32", dtype="<f4") for n in ("beat", "downbeat")]
    else:
        spect = np.random.randn(1, FRAMES, 128).astype(np.float32)
        want = [t.numpy()[0] for t in model(torch.from_numpy(spect))]
    it = Interpreter(model_path=str(path), num_threads=4)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], spect)
    it.invoke()
    t = time.time()
    it.invoke()
    dt = time.time() - t
    outs = it.get_output_details()
    got = [it.get_tensor(o["index"]).reshape(-1) for o in outs]
    # The converter names the outputs by position; say which is which by comparing.
    err = [np.abs(g - w).max() for g, w in zip(got, want)]
    print(
        f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, rewrite vs stock {rewrite:.1e}, max |litert - torch| beat {err[0]:.2e} "
        f"downbeat {err[1]:.2e} (peak {np.abs(want[0]).max():.1f}), {dt:.2f} s on 4 threads here, "
        f"in {it.get_input_details()[0]['name']} {it.get_input_details()[0]['shape'].tolist()}, "
        f"out {[(o['name'], o['shape'].tolist()) for o in outs]}"
    )
