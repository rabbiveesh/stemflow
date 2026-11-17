"""Export Beat This! (CPJKU, MIT) to ONNX for stemflow's live beat tracking.

Input:  `spect` [1, FRAMES, 128] log-mel spectrogram (see src/beats.rs: 22.05 kHz mono,
        n_fft 1024, hop 441 = 50 fps, slaney mel 30-11000 Hz, log1p(1000 x)).
Output: `beat`, `downbeat` [1, FRAMES] logits (beat already includes downbeat, as
        Beat This!'s SumHead does).

FRAMES is fixed (default 1500 = 30 s, the model's training chunk), so the graph has
static shapes, which the mobile runtimes like. Also writes <name>.ref.npz: a test
spectrogram from a WAV and PyTorch's logits for it, for the Rust parity test.

Usage: .venv/bin/python -I export_beat_this.py <out_dir> <beat_this_site_dir> [clip.wav]
  e.g. export_beat_this.py ../models/beats .venv-gpu/lib/python3.10/site-packages \
       ../out/ab/src/clip1.wav
"""
import sys
from pathlib import Path

import numpy as np
import torch

out_dir = Path(sys.argv[1])
sys.path.append(sys.argv[2])  # after this venv's own packages: only beat_this comes from there
wav = sys.argv[3] if len(sys.argv) > 3 else None
FRAMES = 1500

from beat_this.inference import load_model  # noqa: E402
from beat_this.preprocessing import LogMelSpect  # noqa: E402


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, spect):
        y = self.m(spect)
        return y["beat"], y["downbeat"]


out_dir.mkdir(parents=True, exist_ok=True)
spect = None
if wav:
    import soundfile as sf
    import soxr

    x, sr = sf.read(wav, dtype="float64")
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = soxr.resample(x, sr, 22050)
    spect = LogMelSpect()(torch.tensor(x, dtype=torch.float32))[:FRAMES]
    assert len(spect) == FRAMES, f"{wav} is shorter than {FRAMES / 50} s"

for name, ck in (("beat_this_final0", "final0"), ("beat_this_small0", "small0")):
    model = Wrap(load_model(ck, "cpu")).eval()
    dummy = torch.randn(1, FRAMES, 128)
    path = out_dir / f"{name}_{FRAMES}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), str(path), input_names=["spect"], output_names=["beat", "downbeat"],
            opset_version=17, dynamo=False,
        )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    test = spect[None] if spect is not None else dummy
    with torch.no_grad():
        tb, td = model(test)
    ob, od = sess.run(None, {"spect": test.numpy()})
    err = max(np.abs(ob - tb.numpy()).max(), np.abs(od - td.numpy()).max())
    print(f"{path.name}: {path.stat().st_size / 1e6:.1f} MB, max |onnx - torch| = {err:.2e}")
    if spect is not None:
        np.savez(out_dir / f"{name}.ref.npz", spect=spect.numpy(), beat=tb.numpy()[0], downbeat=td.numpy()[0],
                 audio22k=np.asarray(x[: FRAMES * 441], dtype=np.float32))
