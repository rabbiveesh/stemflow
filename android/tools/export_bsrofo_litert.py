"""Export heads of BS-RoFormer SW to LiteRT (.tflite) for the phone GPU.

One head (`bass`) or several (`bass,vocals`): the trunk runs once and each head's
masked spectrogram is stacked on the channel axis, [1, 4 * heads, 1025, T], as
tools/export_bsrofo.py does for the desktop (a second head costs ~1% more).

Same split as tools/export_bsrofo.py (STFT/iSTFT stay outside, in Rust): the graph
takes the complex-as-channels spectrogram [1, 4 (L re, L im, R re, R im), 1025, T]
and returns that stem's masked spectrogram in the same layout. The model is
re-written so that the TFLite GPU delegate runs all of it, in fp16:
  - every tensor is 4-D with batch 1: the token grid is [1, D, bands, T] and linears
    are 1x1 convolutions,
  - attention runs one head at a time, as matmuls over [1, rows, N, 64] that
    tflite_matmul4d.py keeps 4-D (the converter makes them 3-D, which costs the GPU
    three copies of the scores),
  - rotary embeddings are constants (q and k are produced as their even and odd
    halves, so the rotation is elementwise),
  - RMSNorm rescales by the token's max before squaring (fp16 overflows at 65504),
    and its gamma is folded into the linear that follows,
  - the band split is one (n_bins x 1) convolution per band, and the masks of all
    bands are laid out as a spectrogram by a single reshape and transpose,
  - constants are contiguous (see `const`),
  - T is fixed (the window), not dynamic.

Needs the MSST checkout (tools/third_party/msst) and models/ext/bsrofo/{config.yaml,
model.ckpt}, both in the main checkout.
Usage:  python -I export_bsrofo_litert.py REPO_ROOT OUT_DIR [stem[,stem...]] [window_seconds]
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "tools" / "third_party" / "msst"))

import types  # noqa: E402

# The package's __init__ also imports its mel-band models, which need librosa just to
# be defined. BS-RoFormer itself doesn't.
_librosa = types.ModuleType("librosa")
_librosa.filters = types.ModuleType("librosa.filters")
sys.modules.setdefault("librosa", _librosa)
sys.modules.setdefault("librosa.filters", _librosa.filters)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from models.bs_roformer.bs_roformer import BSRoformer  # noqa: E402

N_FFT, HOP = 2048, 512
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
STEM = sys.argv[3] if len(sys.argv) > 3 else "bass"
SEG = round(float(sys.argv[4] if len(sys.argv) > 4 else 3) * 44100 / HOP) * HOP
T = 1 + SEG // HOP


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: tuple(l.construct_sequence(n)))
src_dir = root / "models" / "ext" / "bsrofo"
cfg = yaml.load((src_dir / "config.yaml").read_text(), Loader=Loader)
mcfg = dict(cfg["model"], flash_attn=False)
assert (mcfg["stft_n_fft"], mcfg["stft_hop_length"], mcfg.get("stft_normalized")) == (N_FFT, HOP, False)
m = BSRoformer(**mcfg).eval()
m.requires_grad_(False)
state = torch.load(src_dir / "model.ckpt", map_location="cpu", weights_only=False)
m.load_state_dict(state.get("state_dict", state))
STEMS = STEM.split(",")
heads = [cfg["training"]["instruments"].index(s) for s in STEMS]
assert not m.skip_connection and m.zero_dc and m.audio_channels == 2
BANDS = [d // 4 for d in m.band_split.dim_inputs]  # bins per band
H, DH = mcfg["heads"], mcfg["dim_head"]


def const(t):
    """A constant for the trace. It must be contiguous: the converter gave a clone of a
    permuted tensor (which keeps the permuted strides) the data of another constant of
    the same shape, silently."""
    return t.detach().contiguous().clone()


def conv_w(w):
    """Linear weight [out, in] as a constant 1x1-conv weight."""
    return const(w.detach()[:, :, None, None])


def vec(b):
    return None if b is None else const(b)


# RMSNorm is x / sqrt(mean(x^2)) * gamma over the channels. In fp16 the squares
# overflow, so x is first scaled to a maximum of RMS_PEAK; any scale gives the same
# result. RMS_EPS only keeps an all-zero token (digital silence) from being 0 * inf:
# the mean of squares is at least RMS_PEAK^2 / C > 1.9 otherwise.

# The division is by the peak itself, not a multiplication by its reciprocal: for a
# silent token 1 / PEAK_FLOOR is beyond fp16, and 0 * inf is NaN.

RMS_PEAK, RMS_EPS, PEAK_FLOOR = 32.0, 1e-4, 1e-4


def rms(x):
    """RMSNorm without gamma over the channels of [1, C, H, W]."""
    C = x.shape[1]
    x1 = x / (x.abs().amax(dim=1, keepdim=True) + PEAK_FLOOR) * RMS_PEAK
    return x1 * torch.rsqrt(F.conv2d(x1 * x1, torch.full((1, C, 1, 1), 1.0 / C)) + RMS_EPS)


class Attn:
    """Attention's weights, one set per head, with the RMSNorm gamma folded in."""

    def __init__(self, a, n, along_t):
        self.along_t = along_t
        g = a.norm.gamma.detach()
        W = a.to_qkv.weight.detach() * g  # [3 * H * DH, D]
        part = lambda qkv, h, sl: conv_w(W[qkv * H * DH + h * DH : qkv * H * DH + (h + 1) * DH][sl])
        ev, od = slice(0, None, 2), slice(1, None, 2)
        self.qa = [part(0, h, ev) * a.scale for h in range(H)]  # 1/sqrt(d) folded into q
        self.qb = [part(0, h, od) * a.scale for h in range(H)]
        self.ka = [part(1, h, ev) for h in range(H)]
        self.kb = [part(1, h, od) for h in range(H)]
        self.v = [part(2, h, slice(None)) for h in range(H)]
        gw, gb = a.to_gates.weight.detach() * g, a.to_gates.bias.detach()
        self.gw = [conv_w(gw[h : h + 1]) for h in range(H)]
        self.gb = [const(gb[h : h + 1]) for h in range(H)]
        self.out = conv_w(a.to_out[0].weight)
        # Rotary embedding: pair j of a head is rotated by pos * freq_j.
        ang = a.rotary_embed.freqs.detach()[None, :] * torch.arange(n)[:, None]  # [n, DH / 2]
        shape = (1, DH // 2, 1, n) if along_t else (1, DH // 2, n, 1)
        self.cos, self.sin = const(ang.cos().t().reshape(shape)), const(ang.sin().t().reshape(shape))


def attention(p, x):
    """Attention.forward on the token grid [1, D, bands, T]: along T within each band
    (p.along_t), or along the bands within each frame. All convolutions run on the
    grid as it is (on a Pixel 10a they are 30% slower on [1, D, T, bands], and slower
    on any other shape tried); only the matmul operands are transposed, to
    [1, rows, N, 64] with the sequence as W."""
    xn = rms(x)
    if p.along_t:
        rows_n, rows_d, back = (0, 2, 3, 1), (0, 2, 1, 3), (0, 3, 1, 2)
    else:
        rows_n, rows_d, back = (0, 3, 2, 1), (0, 3, 1, 2), (0, 3, 2, 1)
    outs = []
    for h in range(H):
        rot = []
        for wa, wb in ((p.qa[h], p.qb[h]), (p.ka[h], p.kb[h])):
            a, b = F.conv2d(xn, wa), F.conv2d(xn, wb)
            rot.append(torch.cat([a * p.cos - b * p.sin, b * p.cos + a * p.sin], dim=1))
        q, k = rot  # [1, DH, bands, T]
        att = torch.softmax(q.permute(rows_n) @ k.permute(rows_d), dim=-1)  # [1, rows, N, N]
        o = (att @ F.conv2d(xn, p.v[h]).permute(rows_n)).permute(back)  # [1, DH, bands, T]
        outs.append(o * torch.sigmoid(F.conv2d(xn, p.gw[h], p.gb[h])))
    return F.conv2d(torch.cat(outs, dim=1), p.out)


class FF:
    def __init__(self, ff):
        norm, l1, act, _, l2, _ = ff.net
        assert isinstance(act, torch.nn.GELU)
        self.w1, self.b1 = conv_w(l1.weight.detach() * norm.gamma.detach()), vec(l1.bias)
        self.w2, self.b2 = conv_w(l2.weight), vec(l2.bias)


def feed_forward(p, x):
    return F.conv2d(F.gelu(F.conv2d(rms(x), p.w1, p.b1)), p.w2, p.b2)


def transformer(p, x):
    attn, ff = p
    x = x + attention(attn, x)
    return x + feed_forward(ff, x)


def prep_transformer(t, n, along_t):
    assert len(t.layers) == 1 and isinstance(t.norm, torch.nn.Identity)
    attn, ff = t.layers[0]
    return Attn(attn, n, along_t), FF(ff)


class Band:
    """One band: its slice of bins, the band-split linear as a [bins, 1] convolution
    over [1, 4, bins, T], and the mask MLP. The model's features are ordered (bin,
    channel, re/im); the spectrogram is 4 = (channel, re/im) planes by `bins` rows."""

    def __init__(self, f0, n, split, mlp, final_gamma):
        self.f0, self.n = f0, n
        norm, lin = split
        w = lin.weight.detach() * norm.gamma.detach()  # [D, n * 4]
        self.sw, self.sb = const(w.reshape(-1, n, 4).permute(0, 2, 1)[..., None]), vec(lin.bias)
        self.mean_w = torch.full((1, 4, n, 1), 1.0 / (4 * n))
        l1, act, l2 = mlp[0]
        assert isinstance(act, torch.nn.Tanh) and isinstance(mlp[1], torch.nn.GLU)
        self.w1, self.b1 = conv_w(l1.weight.detach() * final_gamma), vec(l1.bias)
        # The GLU's two halves as two convolutions: value and gate, n * 4 channels each.
        k = n * 4
        self.vw, self.vb = conv_w(l2.weight[:k]), vec(l2.bias[:k])
        self.gw, self.gb = conv_w(l2.weight[k:]), vec(l2.bias[k:])

    def split(self, z):
        x = z[:, :, self.f0 : self.f0 + self.n]  # [1, 4, n, T]
        peak = F.max_pool2d(x.abs().amax(dim=1, keepdim=True), (self.n, 1))  # [1, 1, 1, T]
        x1 = x / (peak + PEAK_FLOOR) * RMS_PEAK
        x1 = x1 * torch.rsqrt(F.conv2d(x1 * x1, self.mean_w) + RMS_EPS)
        return F.conv2d(x1, self.sw, self.sb)  # [1, D, 1, T]

    def mask(self, x):
        """[1, D, 1, T] -> this band's mask as [1, n * 4 (bin, plane), 1, T]"""
        h = torch.tanh(F.conv2d(x, self.w1, self.b1))  # [1, hidden, 1, T]
        return F.conv2d(h, self.vw, self.vb) * torch.sigmoid(F.conv2d(h, self.gw, self.gb))


def perm4(pairs):
    """A constant 1x1 convolution over the 4 (channel, re/im) planes: out[o] = sign * in[i]."""
    w = torch.zeros(4, 4, 1, 1)
    for o, (i, sign) in enumerate(pairs):
        w[o, i] = sign
    return w


class Core(torch.nn.Module):
    """BSRoformer.forward between the STFT and the iSTFT, for the given heads."""

    def __init__(self, m, heads):
        super().__init__()
        nb = len(BANDS)
        self.layers = []
        for block in m.layers:
            time_t, freq_t = block
            self.layers.append((prep_transformer(time_t, T, True), prep_transformer(freq_t, nb, False)))
        fg = m.final_norm.gamma.detach()
        # one set of bands per head; the split (shared by all) runs from the first set
        self.heads = []
        for head in heads:
            est, f0, bands = m.mask_estimators[head], 0, []
            for n, split, mlp in zip(BANDS, m.band_split.to_features, est.to_freqs):
                bands.append(Band(f0, n, split, mlp, fg))
                f0 += n
            self.heads.append(bands)
        self.bands = self.heads[0]
        # Complex product z * mask on (re, im) planes: re = zr mr - zi mi, im = zr mi + zi mr
        self.m_re = perm4([(0, 1), (0, 1), (2, 1), (2, 1)])
        self.m_im = perm4([(1, -1), (1, 1), (3, -1), (3, 1)])
        self.z_swap = perm4([(1, 1), (0, 1), (3, 1), (2, 1)])
        self.dc = torch.ones(1, 1, f0, 1)
        self.dc[:, :, 0] = 0

    def forward(self, z):
        x = torch.cat([b.split(z) for b in self.bands], dim=2)  # [1, D, bands, T]
        for time_p, freq_p in self.layers:
            x = transformer(freq_p, transformer(time_p, x))
        x = rms(x)
        zs = F.conv2d(z, self.z_swap)
        out = []
        for bands in self.heads:
            mask = torch.cat([b.mask(x[:, :, i : i + 1]) for i, b in enumerate(bands)], dim=1)
            mask = mask.reshape(1, -1, 4, mask.shape[-1]).permute(0, 2, 1, 3)  # [1, 4, bins, T]
            y = z * F.conv2d(mask, self.m_re) + zs * F.conv2d(mask, self.m_im)
            out.append(y * self.dc)
        return out[0] if len(out) == 1 else torch.cat(out, dim=1)


def stft(mix):
    w = torch.hann_window(N_FFT)
    z = torch.stft(mix.reshape(2, -1), N_FFT, HOP, N_FFT, window=w, center=True, return_complex=True)
    return torch.view_as_real(z).permute(0, 3, 1, 2).reshape(1, 4, z.shape[1], z.shape[2])


def istft(spec, n):
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


core = Core(m, heads).eval()
with torch.no_grad():
    mix = test_mix(SEG)
    z = stft(mix)
    assert z.shape[-1] == T
    spec = core(z)
    stock = m(mix)[0]
    for k, (s, h) in enumerate(zip(STEMS, heads)):
        err = float((istft(spec[:, 4 * k : 4 * k + 4], SEG) - stock[h]).abs().max())
        print(f"{s}: torch split vs stock forward, max abs err: {err:.2e} (peak {float(stock[h].abs().max()):.2f})")
print("spec", tuple(z.shape), "window", SEG, "samples")

import litert_torch  # noqa: E402

edge = litert_torch.convert(core, (z,))
name = f"bsrofo_sw_{'+'.join(STEMS)}_{SEG}.tflite"
edge.export(str(out_dir / name))
# The converter makes every matmul rank 3, which the GPU delegate runs with three
# extra copies of the attention scores; make them [1, rows, N, N] again.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tflite_matmul4d  # noqa: E402

print("matmuls made 4-D:", tflite_matmul4d.rewrite(str(out_dir / name), str(out_dir / name)))
print("wrote", out_dir / name, f"{(out_dir / name).stat().st_size / 1e6:.1f} MB")

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

it = Interpreter(model_path=str(out_dir / name))
it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]["index"], z.numpy())
it.invoke()
o = it.get_tensor(it.get_output_details()[0]["index"])
print("litert vs torch spec max abs err:", float(np.abs(o - spec.numpy()).max()), "peak", float(spec.abs().max()))
