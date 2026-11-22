"""Convert one htdemucs_ft specialist to a Core ML package for `--device coreml` on
macOS, the same graph as export_htdemucs.py (mix + magnitude spectrogram in, the
source's spectrogram + time-branch waveform out), run by Core ML directly instead
of through ONNX Runtime, whose Core ML route leaves parts of it on the CPU.

Usage: .venv/bin/python -I export_htdemucs_coreml.py ../models [stem] [fp16|fp32]
Writes htdemucs_ft_<stem>.mlpackage (fp32: htdemucs_ft_<stem>.fp32.mlpackage).
On a Mac it also times each compute-unit choice and checks it against torch.
"""
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))  # python -I leaves the script's directory off the path

import coremltools as ct  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from export_htdemucs import Core, load, segment, test_mix  # noqa: E402

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "../models")
STEM = sys.argv[2] if len(sys.argv) > 2 else "bass"
FP32 = (sys.argv[3] if len(sys.argv) > 3 else "fp16") == "fp32"

m, src = load(STEM)
core = Core(m, src).eval()
mix = torch.from_numpy(test_mix(segment(m)))[None]
with torch.no_grad():
    mag = m._magnitude(m._spec(mix))
    want = [t.numpy() for t in core(mix, mag)]
    traced = torch.jit.trace(core, (mix, mag), check_trace=False)
mlm = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="mix", shape=tuple(mix.shape), dtype=np.float32),
        ct.TensorType(name="mag", shape=tuple(mag.shape), dtype=np.float32),
    ],
    outputs=[ct.TensorType(name="spec", dtype=np.float32), ct.TensorType(name="wave", dtype=np.float32)],
    compute_precision=ct.precision.FLOAT32 if FP32 else ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15,
)
mlm.short_description = f"htdemucs_ft {STEM} specialist"
path = out_dir / f"htdemucs_ft_{STEM}{'.fp32' if FP32 else ''}.mlpackage"
mlm.save(str(path))
print("wrote", path)

if sys.platform == "darwin":
    feed = {"mix": mix.numpy(), "mag": mag.numpy()}
    for name in ("ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"):
        t0 = time.time()
        run = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, name))
        load_s = time.time() - t0
        for _ in range(2):
            run.predict(feed)
        times = []
        for _ in range(5):
            t0 = time.time()
            got = run.predict(feed)
            times.append(time.time() - t0)
        errs = [float(np.abs(got[k] - w).max()) for k, w in zip(("spec", "wave"), want)]
        print(f"  {name:12} load {load_s:5.1f} s  predict p50 {sorted(times)[2]:.3f} s  "
              f"max err spec {errs[0]:.2e} wave {errs[1]:.2e}")
