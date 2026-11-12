"""Export one specialist of htdemucs_ft (bass by default) to ONNX, with STFT/iSTFT moved out.

ONNX can't carry complex tensors, so the graph takes the raw (padded) mix plus its
complex-as-channels spectrogram and returns that source's spectrogram (still
complex-as-channels) plus the time-branch waveform. Rust does the STFT, the iSTFT,
and the final sum, exactly mirroring demucs' HTDemucs._spec / _ispec.

Usage:  .venv/bin/python -I export_htdemucs.py ../models [stem]
stem: bass (default), drums, other or vocals.
Writes: htdemucs_ft_<stem>.onnx, plus reference vectors (mix + expected stem) for
`stemflow verify` in ref/ (bass) or ref_demucs_<stem>/.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from einops import rearrange

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../models")
out_dir.mkdir(parents=True, exist_ok=True)
STEM = sys.argv[2] if len(sys.argv) > 2 else "bass"

bag = get_model("htdemucs_ft")
bass_idx = bag.sources.index(STEM)
# htdemucs_ft is 4 specialists; the bag weights pick one model per source.
weights = torch.tensor(bag.weights)  # [n_models, n_sources]
model_idx = int(weights[:, bass_idx].argmax())
assert weights[model_idx, bass_idx] == 1 and weights[:, bass_idx].sum() == 1, weights
m: HTDemucs = bag.models[model_idx].eval()
m.requires_grad_(False)  # keeps tracing from holding autograd state
# ...but then nn.MultiheadAttention takes its fused fast path, which ONNX can't export.
torch.backends.mha.set_fastpath_enabled(False)
assert m.cac and m.use_train_segment and m.samplerate == 44100
SEG = int(m.segment * m.samplerate)
print(f"{STEM} model = #{model_idx}, sources={m.sources}, segment={SEG} samples, nfft={m.nfft}")


class Core(torch.nn.Module):
    """HTDemucs.forward between `mag = self._magnitude(z)` and `zout = self._mask(...)`."""

    def __init__(self, m, src):
        super().__init__()
        self.m, self.src = m, src

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = m.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, mix.shape[-1])
        xt = xt * stdt[:, None] + meant[:, None]
        # one source only: spec [1, 4 (L re, L im, R re, R im), Fq, T], wave [1, 2, SEG]
        return x[:, self.src], xt[:, self.src]


core = Core(m, bass_idx).eval()

# Deterministic test signal with some musical structure: a walking bass-ish line,
# some chords and noisy hats, stereo-shifted. Only used to check numerics.
rng = np.random.default_rng(0)
t = np.arange(SEG) / 44100
notes = 55 * 2 ** (np.array([0, 3, 5, 7, 10, 7, 5, 3]) / 12)
f0 = notes[(t * 2).astype(int) % len(notes)]
bassline = 0.4 * np.sin(2 * np.pi * np.cumsum(f0) / 44100)
chords = sum(0.08 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
hats = 0.05 * rng.standard_normal(SEG) * (np.sin(2 * np.pi * 4 * t) > 0.9)
mix_np = np.stack([bassline + chords + hats, 0.8 * bassline + chords - hats]).astype(np.float32)
mix = torch.from_numpy(mix_np)[None]

with torch.no_grad():
    z = m._spec(mix)
    mag = m._magnitude(z)
    full = m(mix)[0, bass_idx]  # reference: the stock forward, that source
    spec, wave = core(mix, mag)
    # Re-assemble in torch to prove the split is exact before trusting ONNX.
    zout = m._mask(z, spec[:, None])
    re = (m._ispec(zout, SEG) + wave[:, None])[0, 0]
    print("torch split vs stock forward, max abs err:", float((re - full).abs().max()))

onnx_path = out_dir / f"htdemucs_ft_{STEM}.onnx"
with torch.no_grad():
    torch.onnx.export(
        core,
        (mix, mag),
        str(onnx_path),
        input_names=["mix", "mag"],
        output_names=["spec", "wave"],
        opset_version=17,
        do_constant_folding=True,
    )
print("wrote", onnx_path, f"{onnx_path.stat().st_size / 1e6:.1f} MB")

import onnxruntime as ort

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
o_spec, o_wave = sess.run(None, {"mix": mix_np[None], "mag": mag.numpy()})
print("onnx vs torch spec max abs err:", float(np.abs(o_spec - spec.numpy()).max()))
print("onnx vs torch wave max abs err:", float(np.abs(o_wave - wave.numpy()).max()))

ref = out_dir / ("ref" if STEM == "bass" else f"ref_demucs_{STEM}")
ref.mkdir(exist_ok=True)
mix_np.tofile(ref / "mix.f32")  # [2, SEG] planar
full.numpy().astype(np.float32).tofile(ref / f"{STEM}.f32")  # [2, SEG] planar
mag.numpy().astype(np.float32).tofile(ref / "mag.f32")  # [1, 4, 2048, 336]
print("wrote reference vectors to", ref)
