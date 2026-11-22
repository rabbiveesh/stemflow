"""Convert BS-Roformer SW to a Core ML package for one live window length, for
`--device coreml` on macOS (ONNX Runtime's Core ML route is slower than the CPU for
this model; a direct conversion run on the GPU is ~7x faster than the CPU).

Same graph as export_bsrofo.py (spectrogram in, masked spectrogram out), but the
frame count is fixed: T = 1 + window / 512. Measured on an M5 at T = 259 (3 s):
fp16 on the GPU 0.31 s per window, fp32 on the GPU 0.78 s, CPU (ONNX) 2.1 s. The
Neural Engine is slow for it (2.6 s), so stemflow runs these on CPU+GPU.

Usage: .venv/bin/python -I export_bsrofo_coreml.py ../models/ext/bsrofo ../models [stem] [seconds,...] [fp16|fp32]
Writes bsrofo_sw_<stem>.t<T>.mlpackage (fp32 adds .fp32 before .mlpackage), which
stemflow picks for --sep bsrofo --window <seconds> on --device coreml. Default
window 3 s, fp16. Conversion works on Linux too; only running it needs a Mac.
"""
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))  # python -I leaves the script's directory off the path

import coremltools as ct  # noqa: E402
import numpy as np  # noqa: E402
import rotary_embedding_torch.rotary_embedding_torch as rot  # noqa: E402
import torch  # noqa: E402

from export_bsrofo import HOP, Core, load, stft, test_mix  # noqa: E402

# coremltools can't convert the einsum('..., f -> ... f') that builds the rotary
# frequency table; it's an outer product.
def _einsum(eq, *xs):
    if eq.replace(" ", "") == "...,f->...f":
        return xs[0][..., None] * xs[1]
    return torch.einsum(eq, *xs)


rot.einsum = _einsum

src_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../models/ext/bsrofo")
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "../models")
STEMS = (sys.argv[3] if len(sys.argv) > 3 else "bass").split(",")
WINDOWS = [float(w) for w in (sys.argv[4] if len(sys.argv) > 4 else "3").split(",")]
FP32 = (sys.argv[5] if len(sys.argv) > 5 else "fp16") == "fp32"
STEM = "+".join(STEMS)

m, head_idx = load(src_dir, STEMS)
core = Core(m, head_idx).eval()
for w in WINDOWS:
    n = round(w * 44100)
    frames = 1 + n // HOP
    z = stft(test_mix(n, seed=1))
    with torch.no_grad():
        # check_trace would trip over the rotary position arange (int64 vs float64),
        # which doesn't matter at a fixed shape.
        traced = torch.jit.trace(core, (z,), check_trace=False)
        want = core(z).numpy()
    mlm = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="spec_in", shape=tuple(z.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name="spec", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32 if FP32 else ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlm.short_description = f"BS-Roformer SW {STEM}, {frames} frames ({n} samples)"
    path = out_dir / f"bsrofo_sw_{STEM}.t{frames}{'.fp32' if FP32 else ''}.mlpackage"
    mlm.save(str(path))
    print("wrote", path)
    if sys.platform == "darwin":
        run = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        got = run.predict({"spec_in": z.numpy()})["spec"]
        print(f"  core ml (gpu) vs torch: max abs err {float(np.abs(got - want).max()):.2e}, "
              f"peak {float(np.abs(want).max()):.2f}")
