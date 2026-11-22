"""Export BS-Roformer SW (6-stem, one head: bass by default) to ONNX, with the STFT/iSTFT
moved out (Rust does them), the same split as export_scnet.py.

The graph takes the complex-as-channels spectrogram of the chunk,
[1, 4 (L re, L im, R re, R im), 1025, T], and returns the masked
spectrogram of that stem in the same layout (DC bin zeroed, as the model's zero_dc does).
The STFT is torch.stft: n_fft 2048, hop 512, periodic Hann window, *not*
normalized, center=True, reflect padding; T = 1 + samples / 512.

T is a dynamic axis: the transformers use rotary position embeddings, so the
graph runs on any chunk length (trained at 588800 samples = 13.35 s).

Needs the MSST source checkout (tools/third_party/msst) for the model class and
models/ext/bsrofo/{config.yaml,model.ckpt} (enerjazzer/BS-ROFO-SW-Fixed).
Usage: .venv/bin/python -I export_bsrofo.py ../models/ext/bsrofo ../models [stem]
stem is one of the checkpoint's instruments (bass, drums, other, vocals, guitar,
piano); it writes bsrofo_sw_<stem>.onnx and reference vectors to ref_bsrofo
(bass) or ref_bsrofo_<stem>.

Several stems, comma-separated (e.g. bass,vocals), share one pass through the
trunk: the graph then returns [1, 4 * stems, 1025, T], each stem's 4 channels in
the order given, and is named bsrofo_sw_<stem>+<stem>.onnx.
"""
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "third_party" / "msst"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from einops import pack, rearrange, unpack  # noqa: E402
from models.bs_roformer.bs_roformer import BSRoformer  # noqa: E402

N_FFT, HOP = 2048, 512


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))


def load(src_dir, stems):
    """The checkpoint's BSRoformer, and the indices of the heads for `stems`."""
    cfg = yaml.load((src_dir / "config.yaml").read_text(), Loader=Loader)
    mcfg = dict(cfg["model"], flash_attn=False)
    assert (mcfg["stft_n_fft"], mcfg["stft_hop_length"], mcfg.get("stft_normalized")) == (N_FFT, HOP, False)
    m = BSRoformer(**mcfg).eval()
    m.requires_grad_(False)  # no autograd state while tracing: the export took >12 GB of RAM without this
    state = torch.load(src_dir / "model.ckpt", map_location="cpu", weights_only=False)
    m.load_state_dict(state.get("state_dict", state))
    assert not m.skip_connection and m.zero_dc
    return m, [cfg["training"]["instruments"].index(s) for s in stems]


class Core(torch.nn.Module):
    """BSRoformer.forward between the STFT and the iSTFT, only the heads asked for."""

    def __init__(self, m, heads):
        super().__init__()
        self.m, self.heads = m, torch.nn.ModuleList(m.mask_estimators[h] for h in heads)

    def forward(self, spec_in):
        m = self.m
        _, _, F, T = spec_in.shape
        # [1, (s c), F, T] -> model layout [1, (F s), T, c]
        z = spec_in.reshape(1, 2, 2, F, T).permute(0, 3, 1, 4, 2).reshape(1, F * 2, T, 2)
        x = rearrange(z, "b f t c -> b t (f c)")
        x = m.band_split(x)
        for block in m.layers:
            if len(block) == 3:
                lin, time_t, freq_t = block
                x, ps = pack([x], "b * d")
                x = lin(x)
                (x,) = unpack(x, ps, "b * d")
            else:
                time_t, freq_t = block
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_t(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_t(x)
            (x,) = unpack(x, ps, "* f d")
        x = m.final_norm(x)
        dc = torch.ones(1, 1, F, 1)
        dc[:, :, 0] = 0
        out = []
        for head in self.heads:
            mask = rearrange(head(x), "b t (f c) -> b f t c", c=2)  # [1, F*2, T, 2]
            re = z[..., 0] * mask[..., 0] - z[..., 1] * mask[..., 1]
            im = z[..., 0] * mask[..., 1] + z[..., 1] * mask[..., 0]
            y = torch.stack([re, im], -1).reshape(1, F, 2, T, 2).permute(0, 2, 4, 1, 3).reshape(1, 4, F, T)
            out.append(y * dc)
        return out[0] if len(out) == 1 else torch.cat(out, 1)


def stft(mix):
    w = torch.hann_window(N_FFT)
    z = torch.stft(mix.reshape(2, -1), N_FFT, HOP, N_FFT, window=w, center=True, return_complex=True)
    return torch.view_as_real(z).permute(0, 3, 1, 2).reshape(1, 4, z.shape[1], z.shape[2])


def istft(spec, n):
    """[1, 4 * stems, F, T] -> [stems, 2, n]"""
    if spec.shape[1] > 4:
        return torch.stack([istft(spec[:, i:i + 4], n) for i in range(0, spec.shape[1], 4)])
    s = spec.reshape(2, 2, spec.shape[2], spec.shape[3]).permute(0, 2, 3, 1).contiguous()
    w = torch.hann_window(N_FFT)
    return torch.istft(torch.view_as_complex(s), N_FFT, HOP, N_FFT, window=w, center=True, length=n)


def test_mix(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 44100
    notes = 55 * 2 ** (np.array([0, 3, 5, 7, 10, 7, 5, 3]) / 12)
    f0 = notes[(t * 2).astype(int) % len(notes)]
    bass = 0.4 * np.sin(2 * np.pi * np.cumsum(f0) / 44100)
    chords = sum(0.08 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
    hats = 0.05 * rng.standard_normal(n) * (np.sin(2 * np.pi * 4 * t) > 0.9)
    return torch.from_numpy(np.stack([bass + chords + hats, 0.8 * bass + chords - hats]).astype(np.float32))[None]


def main():
    src_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../models/ext/bsrofo")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "../models")
    STEMS = (sys.argv[3] if len(sys.argv) > 3 else "bass").split(",")
    STEM = "+".join(STEMS)
    m, head_idx = load(src_dir, STEMS)
    core = Core(m, head_idx).eval()
    EXPORT_N = 4 * 44100  # trace length; T stays dynamic
    with torch.no_grad():
        # The 13.35 s check takes ~20 GB of RAM on CPU; STEMFLOW_LONG_CHECK=0 skips it.
        for n in (EXPORT_N, 588800) if os.environ.get("STEMFLOW_LONG_CHECK", "1") != "0" else (EXPORT_N,):
            mix = test_mix(n)
            full = m(mix)[0, head_idx]
            full = full[0] if len(head_idx) == 1 else full
            y = istft(core(stft(mix)), n)
            print(f"n={n}: torch split vs stock forward, max abs err {float((y - full).abs().max()):.2e}")
        z = stft(test_mix(EXPORT_N))

    onnx_path = out_dir / f"bsrofo_sw_{STEM}.onnx"
    with torch.no_grad():
        torch.onnx.export(core, (z,), str(onnx_path), input_names=["spec_in"], output_names=["spec"],
                          dynamic_axes={"spec_in": {3: "frames"}, "spec": {3: "frames"}},
                          opset_version=17, do_constant_folding=True)
    print("wrote", onnx_path, f"{onnx_path.stat().st_size / 1e6:.1f} MB")

    import onnxruntime as ort  # noqa: E402

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref = out_dir / ("ref_bsrofo" if STEM == "bass" else f"ref_bsrofo_{STEM}")
    ref.mkdir(exist_ok=True)
    with torch.no_grad():
        for n in (EXPORT_N, 3 * 44100 + 777):
            mix = test_mix(n, seed=1)
            z = stft(mix)
            spec = core(z)
            (o,) = sess.run(None, {"spec_in": z.numpy()})
            print(f"n={n}: onnx vs torch spec max abs err {float(np.abs(o - spec.numpy()).max()):.2e}, "
                  f"peak {float(spec.abs().max()):.2f}")
        mix.numpy().tofile(ref / "mix.f32")
        full = m(mix)[0]
        for s, h in zip(STEMS, head_idx):
            full[h].numpy().astype(np.float32).tofile(ref / f"{s}.f32")
        z.numpy().tofile(ref / "spec_in.f32")
        print(f"wrote reference vectors ({mix.shape[-1]} samples) to", ref)


if __name__ == "__main__":
    main()
