"""Export one htdemucs_ft specialist to LiteRT (.tflite) for the phone GPU.

Same split as tools/export_htdemucs.py (STFT/iSTFT stay outside, in Rust): inputs
`mix` [1, 2, SEG] and `mag` [1, 4, 2048, frames]; outputs that source's spectrogram
and the time-branch waveform. Differences, all for the GPU delegate's sake:
  - no 5-D tensors: only the wanted source's channels are computed by the last layer,
  - every tensor is 4-D [1, C, H, W] with batch 1: the time branch is [1, C, 1, T],
    the frequency branch [1, C, Fr, T], the transformer [1, C, 1, tokens]. demucs'
    own layers fold Fr into the batch and use 3-D convolutions, which the converter
    wraps in hundreds of reshapes and transposes,
  - whole-tensor means and maxima are taken in stages (see `reduce_w`),
  - positional and frequency embeddings are baked in as constants (traced, they
    become BROADCAST_TO / PAD / SELECT ops the delegate rejects),
  - LayerScale is folded into the linear layer before it (no broadcast MUL),
  - the segment is a parameter, rounded to a multiple of 1024 samples so the time
    branch never needs padding (7.8 s -> 344064 samples, still 336 STFT frames).

Usage:  python -I export_litert.py OUT_DIR [stem] [segment_seconds] [checkpoint.th]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from demucs.transformer import create_2d_sin_embedding

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
STEM = sys.argv[2] if len(sys.argv) > 2 else "bass"
SEG_S = float(sys.argv[3]) if len(sys.argv) > 3 else 7.8

bag = get_model("htdemucs_ft")
src = bag.sources.index(STEM)
weights = torch.tensor(bag.weights)
m: HTDemucs = bag.models[int(weights[:, src].argmax())].eval()
m.requires_grad_(False)
torch.backends.mha.set_fastpath_enabled(False)
SEG = round(SEG_S * m.samplerate / 1024) * 1024
# The transformer's positional embedding is sinusoidal, so any length works;
# use_train_segment would pad back to exactly 7.8 s.
m.use_train_segment = False


def fold_layer_scale(layer):
    """gamma * linear(x) == linear'(x) with the rows of W and b scaled by gamma."""
    attn = getattr(layer, "self_attn", None) or layer.cross_attn
    for lin, name in ((attn.out_proj, "gamma_1"), (layer.linear2, "gamma_2")):
        g = getattr(layer, name)
        if isinstance(g, torch.nn.Identity):
            continue
        lin.weight.data *= g.scale.data[:, None]
        lin.bias.data *= g.scale.data
        setattr(layer, name, torch.nn.Identity())


# Everything runs on 4-D [1, C, H, W] tensors with batch 1 and 2-D convolutions (1x1
# for linear layers), which is what mobile GPU delegates handle: they reject or
# mis-batch [tokens, C] and [1, tokens, C], and the converter surrounds every 3-D
# convolution with RESHAPE and TRANSPOSE ops that cost real GPU time.


class Conv:
    """A demucs (transposed) convolution, 1-D or 2-D, as constant conv2d weights; 1-D
    ones run along W. `out` keeps only some output channels. The weights must be
    sliced and reshaped here, ahead of the trace: sliced inside it, they become
    run-time SLICE outputs feeding CONV_2D, and the GPU delegate then returns NaN."""

    def __init__(self, c, out=slice(None), pad=None, scale=1.0):
        two = lambda v, fill=1: tuple(v) if len(v) == 2 else (fill, v[0])
        self.tr = isinstance(c, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d))
        w = c.weight.detach()
        w = w if w.dim() == 4 else w[:, :, None, :]
        self.w = (w[:, out] if self.tr else w[out]).clone() * scale
        self.b = c.bias.detach()[out].clone() * scale
        self.stride, self.dil = two(c.stride), two(c.dilation)
        self.pad = two(c.padding, 0) if pad is None else pad

    def __call__(self, x):
        if self.tr:
            return F.conv_transpose2d(x, self.w, self.b, self.stride, self.pad)
        return F.conv2d(x, self.w, self.b, self.stride, self.pad, self.dil)


class Lin(Conv):
    """A linear layer (or `rows` of one) as a 1x1 convolution."""

    def __init__(self, weight, bias, rows=slice(None), scale=1.0):
        self.tr = False
        self.w = weight.detach()[rows, :, None, None].clone() * scale
        self.b = bias.detach()[rows].clone() * scale
        self.stride, self.dil, self.pad = (1, 1), (1, 1), (0, 0)


def per_ch(v):
    return v.detach()[None, :, None, None].clone()


# Means over the channel axis are written as a 1x1 convolution with weights 1/C (and
# eps as its bias). Written as x.mean(dim=1), the converter emits MEAN -> [1, 1, T]
# -> ADD eps -> RESHAPE back to 4-D, and the GPU delegate misreads that 3-D tensor:
# the next convolution then returns NaN (bisected on a Pixel 10a, Mali, OpenCL).


def ch_mean(x):
    C = x.shape[1]
    return F.conv2d(x, torch.full((1, C, 1, 1), 1.0 / C))


# Means and maxima over a whole row or tensor are taken in stages of pooling, at most
# STAGE values at a time. One MEAN / REDUCE_MAX over millions of values is a single
# GPU work item that loops over all of them: 14 ms each on a Pixel 10a for the first
# encoder's 16M values, where an elementwise op over the same tensor takes 1 ms.
# Pooling (which accumulates in fp32 even in fp16 mode) spreads the same sum over
# thousands of work items.

STAGE = 64


def stages(n):
    out = []
    while n > 1:
        f = max((d for d in range(2, min(STAGE, n) + 1) if n % d == 0), default=n)
        out.append(f)
        n //= f
    return out


def reduce_w(v, pool):
    """[1, 1, H, W] -> [1, 1, H, 1]"""
    for f in stages(v.shape[3]):
        v = pool(v, (1, f))
    return v


def reduce_hw(v, pool):
    """[1, 1, H, W] -> [1, 1, 1, 1]"""
    v = reduce_w(v, pool)
    for f in stages(v.shape[2]):
        v = pool(v, (f, 1))
    return v


def ch_max(x):
    return x.amax(dim=1, keepdim=True)


# fp16 safety. The GPU only keeps up in fp16, whose largest number is 65504, and a
# variance squares its input: on real music the inputs of some norms reach ~1900
# (3.7e6 squared), so a plain variance overflows to inf and the stem comes out wrong.
# Every norm therefore divides the centred values by s = max|d| / 128 first, so the
# squares stay below 16384, and corrects eps to match: with d1 = d / s,
#   d * rsqrt(mean(d^2) + eps)  ==  d1 * rsqrt(mean(d1^2) + eps / s^2)
# which is the same number in exact arithmetic.


def safe_norm(x, mean, amax, weight, bias, eps):
    d = x - mean(x)
    inv = 1.0 / (amax(d.abs()) * (1.0 / 128) + 1e-4)  # 1 / s
    d1 = d * inv
    r = eps**0.5 * inv  # (eps / s^2) as a square, so s^2 can't underflow
    return d1 * torch.rsqrt(mean(d1 * d1) + r * r) * weight + bias


def row_mean(v):
    return reduce_w(ch_mean(v), F.avg_pool2d)


def row_max(v):
    return reduce_w(ch_max(v), F.max_pool2d)


class Norm:
    """The affine part of a norm layer as constants; `scale` (a LayerScale that follows
    a GLU) is folded into the half of the channels that the GLU passes through."""

    def __init__(self, n, scale=None):
        self.w, self.b, self.eps = per_ch(n.weight), per_ch(n.bias), n.eps
        if scale is not None:
            C = scale.numel()
            self.w[:, :C] *= per_ch(scale)
            self.b[:, :C] *= per_ch(scale)


def ln(n, x):
    """nn.LayerNorm over the channel axis of [1, C, 1, T]."""
    return safe_norm(x, ch_mean, ch_max, n.w, n.b, n.eps)


def rn(n, x):
    """GroupNorm(1, C) with one mean/var per row of [1, C, H, W]. With H = 1 (time
    branch, transformer tokens) that is the whole tensor; in the frequency branch it
    is what DConv does by folding the frequencies into the batch."""
    return safe_norm(x, row_mean, row_max, n.w, n.b, n.eps)


def safe_standardize(x):
    """(x - mean) / (1e-5 + std) over all of [1, C, H, W], as HTDemucs.forward does,
    plus the mean and std."""
    mean = reduce_hw(ch_mean(x), F.avg_pool2d)
    d = x - mean
    s = reduce_hw(ch_max(d.abs()), F.max_pool2d) * (1.0 / 128) + 1e-4
    d1 = d / s
    n = d1.numel()
    var = reduce_hw(ch_mean(d1 * d1), F.avg_pool2d)
    std = s * torch.sqrt(var * (n / (n - 1)))  # unbiased
    return d / (1e-5 + std), mean, std


def glu(x):
    C = x.shape[1] // 2
    return x[:, :C] * torch.sigmoid(x[:, C:])


class GluConv:
    """glu(conv(x)) as two convolutions, one per half of the output channels, so that
    no SLICE has to copy the halves apart."""

    def __init__(self, c):
        C = c.out_channels // 2
        self.a, self.b = Conv(c, slice(0, C)), Conv(c, slice(C, 2 * C))

    def __call__(self, x):
        return self.a(x) * torch.sigmoid(self.b(x))


def no_norm(n):
    assert isinstance(n, torch.nn.Identity), "htdemucs has norms only in DConv (norm_starts=4)"


def prepare_dconv(dc):
    for seq in dc.layers:
        conv1, n1, act, conv2, n2, g, ls = seq
        assert isinstance(act, torch.nn.GELU) and n1.num_groups == 1 and n2.num_groups == 1
        seq.p = (Conv(conv1), Norm(n1), Conv(conv2), Norm(n2, ls.scale.detach()))


def dconv(dc, x):
    """DConv on [1, C, H, T]: residual branches of dilated convolutions along T."""
    for seq in dc.layers:
        c1, n1, c2, n2 = seq.p
        x = x + glu(rn(n2, c2(F.gelu(rn(n1, c1(x))))))
    return x


def prepare_enc(e):
    e.c = Conv(e.conv)
    if e.empty:
        return
    no_norm(e.norm1)
    if e.rewrite:
        no_norm(e.norm2)
        e.rw = GluConv(e.rewrite)
    if e.dconv:
        prepare_dconv(e.dconv)


def enc(e, x, inject=None):
    """HEncLayer.forward, for both branches."""
    y = e.c(x)
    if e.empty:
        return y
    if inject is not None:
        y = y + inject
    y = F.gelu(y)
    if e.dconv:
        y = dconv(e.dconv, y)
    return e.rw(y) if e.rewrite else y


def prepare_dec(d, out=slice(None)):
    # Padding the transposed convolution crops its output: z[pad:-pad], which is also
    # the time branch's z[pad:pad + length] as the segment needs no padding.
    pad = (d.pad, 0) if d.freq else (0, d.pad)
    d.c = Conv(d.conv_tr, out, pad)
    no_norm(d.norm2)
    if d.empty:
        return
    if d.rewrite:
        no_norm(d.norm1)
        d.rw = GluConv(d.rewrite)
    if d.dconv:
        prepare_dconv(d.dconv)


def dec(d, x, skip):
    """HDecLayer.forward, for both branches: (output, pre)."""
    if not d.empty:
        x = x + skip
        if d.rewrite:
            x = d.rw(x)
        if d.dconv:
            x = dconv(d.dconv, x)
    z = d.c(x)
    return (z if d.last else F.gelu(z)), x


def prepare(l):
    """Fold LayerScale and precompute the conv weights of one transformer layer."""
    fold_layer_scale(l)
    attn = getattr(l, "self_attn", None) or l.cross_attn
    C, H = attn.embed_dim, attn.num_heads
    d = C // H
    W, b = attn.in_proj_weight, attn.in_proj_bias
    head = lambda o, h: slice(o * C + h * d, o * C + (h + 1) * d)
    # One convolution per head, with the 1/sqrt(d) of the scores folded into q.
    attn.cq = [Lin(W, b, head(0, h), d**-0.5) for h in range(H)]
    attn.ck = [Lin(W, b, head(1, h)) for h in range(H)]
    attn.cv = [Lin(W, b, head(2, h)) for h in range(H)]
    attn.co = Lin(attn.out_proj.weight, attn.out_proj.bias)
    l.c1 = Lin(l.linear1.weight, l.linear1.bias)
    l.c2 = Lin(l.linear2.weight, l.linear2.bias)
    for name in ("norm1", "norm2", "norm3", "norm_out"):
        n = getattr(l, name, None)
        if n is not None:
            setattr(l, "p_" + name, Norm(n))


def mha(attn, q, k):
    """nn.MultiheadAttention (no mask) on q [1, C, 1, T], k [1, C, 1, S].

    One head at a time, and each product as a 1x1 convolution whose weights are the
    other operand: scores [1, S, 1, T] = conv(q, weight=k), softmax over the channels,
    out [1, d, 1, T] = conv(scores, weight=v). As matmuls over all heads the converter
    emits [H, T, d] @ [H, d, S], and the GPU delegate reshapes any matmul whose batch
    is not 1 to and from [1, H * T, ...], copying the H x T x S scores three times per
    attention (35 ms of 65 ms on a Pixel 10a). Per-head matmuls don't survive either:
    the converter turns them into FULLY_CONNECTED with run-time weights, which aborts."""
    outs = []
    for cq, ck, cv in zip(attn.cq, attn.ck, attn.cv):
        att = torch.softmax(F.conv2d(cq(q), ck(k).permute(3, 1, 0, 2)), dim=1)  # weight [S, d, 1, 1]
        outs.append(F.conv2d(att, cv(k).permute(1, 3, 0, 2)))  # weight [d, S, 1, 1]
    return attn.co(torch.cat(outs, dim=1))


def layer(l, attn, q, res, k, norm_ff):
    """A norm-first (Cross)TransformerEncoderLayer with LayerScale already folded:
    `q`/`k` are the normed attention inputs, `res` the residual stream."""
    x = res + mha(attn, q, k)
    x = x + l.c2(F.gelu(l.c1(ln(norm_ff, x))))
    return rn(l.p_norm_out, x)


class Core(torch.nn.Module):
    """HTDemucs.forward between `mag = self._magnitude(z)` and `zout = self._mask(...)`."""

    def __init__(self, m, src):
        super().__init__()
        self.m, self.src = m, src
        # Constants, filled in by the first (eager) call and baked in by the trace.
        self.femb = self.pos2d = self.pos1d = None

    def transformer(self, x, xt):
        """CrossTransformerEncoder.forward (batch 1) with the embeddings precomputed."""
        ct = self.m.crosstransformer
        _, C, Fr, T1 = x.shape
        tok = lambda t: t.permute(0, 1, 3, 2).reshape(1, C, 1, -1)  # "c fr t1 -> c (t1 fr)"
        x = tok(x)
        if self.pos2d is None:
            p = create_2d_sin_embedding(C, Fr, T1, x.device, ct.max_period)
            self.pos2d = (ct.weight_pos_embed * tok(p)).clone()
            p = ct._get_pos_embedding(xt.shape[-1], 1, C, x.device)  # [T2, 1, C]
            self.pos1d = (ct.weight_pos_embed * p.permute(1, 2, 0)[:, :, None, :]).clone()
        x = ln(self.norm_in, x) + self.pos2d
        xt = ln(self.norm_in_t, xt) + self.pos1d
        for idx in range(ct.num_layers):
            la, lb = ct.layers[idx], ct.layers_t[idx]
            if idx % 2 == ct.classic_parity:
                n, nt = ln(la.p_norm1, x), ln(lb.p_norm1, xt)
                x = layer(la, la.self_attn, n, x, n, la.p_norm2)
                xt = layer(lb, lb.self_attn, nt, xt, nt, lb.p_norm2)
            else:
                old_x = x
                x = layer(la, la.cross_attn, ln(la.p_norm1, x), x, ln(la.p_norm2, xt), la.p_norm3)
                xt = layer(lb, lb.cross_attn, ln(lb.p_norm1, xt), xt, ln(lb.p_norm2, old_x), lb.p_norm3)
        return x.reshape(1, C, T1, Fr).permute(0, 1, 3, 2), xt

    def forward(self, mix, mag):
        m = self.m
        x, mean, std = safe_standardize(mag)
        xt, meant, stdt = safe_standardize(mix[:, :, None, :])  # [1, 2, 1, SEG]

        saved, saved_t = [], []
        for idx, e in enumerate(m.encoder):
            inject = None
            if idx < len(m.tencoder):
                tenc = m.tencoder[idx]
                xt = enc(tenc, xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = enc(e, x, inject)
            if idx == 0 and m.freq_emb is not None:
                if self.femb is None:
                    frs = torch.arange(x.shape[-2], device=x.device)
                    self.femb = (m.freq_emb_scale * m.freq_emb(frs).t()[None, :, :, None]).clone()
                x = x + self.femb  # [1, C, Fr, 1], broadcast along T
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                x, xt = self.up(x), self.up_t(xt)
            x, xt = self.transformer(x, xt)
            if m.bottom_channels:
                x, xt = self.down(x), self.down_t(xt)
        offset = m.depth - len(m.tdecoder)
        for idx, d in enumerate(m.decoder):
            x, pre = dec(d, x, saved.pop(-1))
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                if tdec.empty:
                    xt, _ = dec(tdec, pre, None)
                else:
                    xt, _ = dec(tdec, xt, saved_t.pop(-1))

        x = x * std + mean  # [1, 4, Fq, T]
        xt = (xt * stdt + meant).reshape(1, 2, -1)  # [1, 2, SEG]
        return x, xt


if len(sys.argv) > 4:  # a fine-tune checkpoint (ft1..ft3)
    sd = torch.load(sys.argv[4], map_location="cpu")
    m.load_state_dict(sd.get("state", sd))

ct = m.crosstransformer
assert ct.emb == "sin" and ct.sin_random_shift == 0
assert all(e.freq for e in m.encoder) and not any(e.freq for e in m.tencoder)
for lyr in list(ct.layers) + list(ct.layers_t):
    assert lyr.norm_first and lyr.norm_out and not lyr.sparse
    prepare(lyr)
for e in list(m.encoder) + list(m.tencoder):
    prepare_enc(e)
# The last decoders compute only the wanted source's channels.
for decs, per_src in ((m.decoder, 4), (m.tdecoder, 2)):
    for d in decs:
        prepare_dec(d, slice(per_src * src, per_src * (src + 1)) if d.last else slice(None))
core = Core(m, src).eval()
core.norm_in, core.norm_in_t = Norm(ct.norm_in), Norm(ct.norm_in_t)
if m.bottom_channels:
    core.up, core.up_t = Conv(m.channel_upsampler), Conv(m.channel_upsampler_t)
    core.down, core.down_t = Conv(m.channel_downsampler), Conv(m.channel_downsampler_t)
rng = np.random.default_rng(0)
mix = torch.from_numpy(0.1 * rng.standard_normal((1, 2, SEG)).astype(np.float32))
with torch.no_grad():
    z = m._spec(mix)
    mag = m._magnitude(z)
    spec, wave = core(mix, mag)
    full = m(mix)[0, src]
    re = (m._ispec(m._mask(z, spec[:, None]), SEG) + wave[:, None])[0, 0]
    print("torch split vs stock forward, max abs err:", float((re - full).abs().max()))
print("mix", tuple(mix.shape), "mag", tuple(mag.shape))

import litert_torch

edge = litert_torch.convert(core, (mix, mag))
# e.g. htdemucs_ft_bass_344064.tflite, or ft2_bass_344064.tflite for out/ft2/step600.th
tag = Path(sys.argv[4]).parent.name if len(sys.argv) > 4 else "htdemucs_ft"
name = f"{tag}_{STEM}_{SEG}.tflite"
edge.export(str(out_dir / name))
print("wrote", out_dir / name, f"{(out_dir / name).stat().st_size / 1e6:.1f} MB")
o = edge(mix, mag)
o = sorted(o, key=lambda a: a.ndim)  # wave (3-D), spec (4-D)
print("litert vs torch wave max abs err:", float(np.abs(o[0] - wave.numpy()).max()))
print("litert vs torch spec max abs err:", float(np.abs(o[1] - spec.numpy()).max()))
