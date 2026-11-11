#!/usr/bin/env bash
# Python env used only to export models to ONNX (and score against stock demucs).
# CPU-only torch: the export doesn't need a GPU, and CUDA torch is ~4 GB more.
set -euo pipefail
cd "$(dirname "$0")"
uv venv -q -p 3.10 .venv
uv pip install -q -p .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  'torch==2.5.1+cpu' 'torchaudio==2.5.1+cpu' 'demucs==4.0.1' \
  'onnx==1.17.0' 'onnxruntime==1.20.1' 'numpy==1.26.4' soundfile
.venv/bin/python -I -c "import torch, demucs; print('export env ready, torch', torch.__version__)"
