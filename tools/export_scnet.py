"""Export SCNet XL IHF (ZFTurbo/MSST, MUSDB18-HQ, multisong bass SDR 11.94) to ONNX,
bass source only, with the outer STFT/iSTFT moved out (Rust does them).

The graph takes the complex-as-channels spectrogram of the padded chunk,
[1, 4 (L re, L im, R re, R im), 2049, T], and returns the bass spectrogram in the
same layout. SCNet's STFT is plain torch.stft: n_fft 4096, hop 1024, *no window*
(rectangular), normalized, center=True, reflect padding.

Needs the MSST source checkout (tools/third_party/msst) for the model class.
Usage: .venv/bin/python -I export_scnet.py ../models
"""
import sys
import urllib.request
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here / "third_party" / "msst"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from models.scnet.scnet import SCNet  # noqa: E402
from models.scnet import separation  # noqa: E402

CKPT_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt"
# From config_musdb18_scnet_xl_more_wide_v5.yaml (v1.0.15).
CFG = dict(
    sources=["drums", "bass", "other", "vocals"], audio_channels=2, dims=[4, 64, 128, 256],
    nfft=4096, hop_size=1024, win_size=4096, normalized=True,
    band_SR=[0.230, 0.370, 0.400], band_stride=[1, 4, 4], band_kernel=[3, 4, 4],
    conv_depths=[3, 3, 3], compress=4, conv_kernel=3, num_dplayer=8, expand=1,
)
CHUNK = 485100  # 11 s

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../models")
out_dir.mkdir(parents=True, exist_ok=True)
ckpt = out_dir / "scnet_xl_ihf.ckpt"
if not ckpt.exists():
    print("downloading", CKPT_URL)
    urllib.request.urlretrieve(CKPT_URL, ckpt)

m = SCNet(**CFG).eval()
state = torch.load(ckpt, map_location="cpu", weights_only=True)
state = state.get("state_dict", state) if isinstance(state, dict) else state
m.load_state_dict(state)
bass_idx = CFG["sources"].index("bass")


class DFT(torch.nn.Module):
    """torch.fft.rfft/irfft(norm='ortho') along dim 3 as matmuls, so the graph is
    plain MatMul (ONNX's DFT op support is spotty across runtimes/EPs)."""

    def __init__(self, n):
        super().__init__()
        # Built in numpy so tracing sees constants, not cos/sin ops.
        n = int(n)
        k = np.arange(n // 2 + 1, dtype=np.float64)[:, None]
        t = np.arange(n, dtype=np.float64)[None, :]
        ang = 2 * np.pi * k * t / n
        s = 1 / np.sqrt(n)
        const = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
        self.fr = const((np.cos(ang) * s).T)  # [n, n/2+1]
        self.fi = const((-np.sin(ang) * s).T)
        # irfft: x[t] = s * (Re X0 + 2 sum_k Re(Xk e^{+i..}) + Re X_{n/2}) for even n
        w = np.full((n // 2 + 1, 1), 2.0)
        w[0] = 1
        if n % 2 == 0:
            w[-1] = 1
        self.ir = const(np.cos(ang) * w * s)  # [n/2+1, n]
        self.ii = const(-np.sin(ang) * w * s)


def patch_feature_conversion():
    """Replace FeatureConversion.forward (rfft/irfft over the time axis) with matmul DFTs."""
    FC = separation.FeatureConversion
    orig = FC.forward
    cache = {}

    def forward(self, x):
        n = int(x.shape[3] if not self.inverse else 2 * (x.shape[3] - 1))
        if n not in cache:
            cache[n] = DFT(n)
        d = cache[n]
        if self.inverse:
            x = x.float()
            half = x.shape[1] // 2
            re, im = x[:, :half], x[:, half:]
            return re @ d.ir + im @ d.ii
        x = x.float()
        return torch.cat([x @ d.fr, x @ d.fi], dim=1)

    FC.forward = forward
    return orig


def spectrogram(m, mix):
    padding = m.hop_length - mix.shape[-1] % m.hop_length
    if (mix.shape[-1] + padding) // m.hop_length % 2 == 0:
        padding += m.hop_length
    x = F.pad(mix, (0, padding))
    L = x.shape[-1]
    z = torch.stft(x.reshape(-1, L), **m.stft_config, return_complex=True)
    z = torch.view_as_real(z).permute(0, 3, 1, 2).reshape(1, 4, z.shape[1], z.shape[2])
    return z, padding, L


class Core(torch.nn.Module):
    """SCNet.forward between the STFT and the iSTFT, bass only."""

    def __init__(self, m, src):
        super().__init__()
        self.m, self.src = m, src

    def forward(self, x):
        m = self.m
        B, C, Fr, T = x.shape
        skips, lens, olens = [], [], []
        for sd in m.encoder:
            x, skip, l, ol = sd(x)
            skips.append(skip)
            lens.append(l)
            olens.append(ol)
        x = m.separation_net(x)
        for fusion, su in m.decoder:
            x = fusion(x, skips.pop())
            x = su(x, lens.pop(), olens.pop())
        x = x.view(B, len(m.sources), C, Fr, T)
        return x[:, self.src]


rng = np.random.default_rng(0)
t = np.arange(CHUNK) / 44100
notes = 55 * 2 ** (np.array([0, 3, 5, 7, 10, 7, 5, 3]) / 12)
f0 = notes[(t * 2).astype(int) % len(notes)]
bass = 0.4 * np.sin(2 * np.pi * np.cumsum(f0) / 44100)
chords = sum(0.08 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
hats = 0.05 * rng.standard_normal(CHUNK) * (np.sin(2 * np.pi * 4 * t) > 0.9)
mix_np = np.stack([bass + chords + hats, 0.8 * bass + chords - hats]).astype(np.float32)
mix = torch.from_numpy(mix_np)[None]

with torch.no_grad():
    full = m(mix)[0, bass_idx]  # stock forward (torch.fft), reference
    z, padding, L = spectrogram(m, mix)
    patch_feature_conversion()
    core = Core(m, bass_idx).eval()
    spec = core(z)
    # reassemble in torch: proves both the split and the matmul DFT are exact
    s = spec.reshape(2, 2, spec.shape[2], spec.shape[3]).permute(0, 2, 3, 1).contiguous()
    y = torch.istft(torch.view_as_complex(s), **m.stft_config)[..., :-padding]
    print(f"chunk {CHUNK}, padding {padding}, spec {tuple(z.shape)}")
    print("torch split vs stock forward, max abs err:", float((y - full).abs().max()))

onnx_path = out_dir / "scnet_xl_ihf_bass.onnx"
torch.onnx.export(core, (z,), str(onnx_path), input_names=["spec_in"], output_names=["spec"],
                  opset_version=17, do_constant_folding=True)
print("wrote", onnx_path, f"{onnx_path.stat().st_size / 1e6:.1f} MB")

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
(o,) = sess.run(None, {"spec_in": z.numpy()})
print("onnx vs torch spec max abs err:", float(np.abs(o - spec.numpy()).max()), "peak", float(np.abs(spec.numpy()).max()))

ref = out_dir / "ref_scnet"
ref.mkdir(exist_ok=True)
mix_np.tofile(ref / "mix.f32")
full.numpy().astype(np.float32).tofile(ref / "bass.f32")
z.numpy().astype(np.float32).tofile(ref / "spec_in.f32")
print("wrote reference vectors to", ref)
