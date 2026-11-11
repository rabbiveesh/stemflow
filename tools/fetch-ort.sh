#!/usr/bin/env bash
# Fetch ONNX Runtime (from Microsoft's GitHub releases) into vendor/, and for CUDA,
# the CUDA 12 runtime libs it needs (from NVIDIA's PyPI wheels) next to it.
#   tools/fetch-ort.sh cpu     # ~30 MB
#   tools/fetch-ort.sh cuda    # ~2.5 GB (cuDNN and cuBLAS are most of it)
set -euo pipefail
ORT_VERSION=1.31.1
root="$(cd "$(dirname "$0")/.." && pwd)"
flavor="${1:-cpu}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

case "$flavor" in
  cpu)  asset="onnxruntime-linux-x64-${ORT_VERSION}.tgz";            dest="$root/vendor/ort-cpu" ;;
  cuda) asset="onnxruntime-linux-x64-gpu_cuda12-${ORT_VERSION}.tgz"; dest="$root/vendor/ort-cuda" ;;
  *) echo "usage: $0 cpu|cuda" >&2; exit 1 ;;
esac

gh release download "v${ORT_VERSION}" -R microsoft/onnxruntime -p "$asset" -D "$tmp"
rm -rf "$dest" && mkdir -p "$dest"
tar -xzf "$tmp/$asset" -C "$dest" --strip-components=1
rm -f "$dest"/lib/libonnxruntime_providers_tensorrt.so

if [[ "$flavor" == cuda ]]; then
  uv venv -q -p 3.12 "$tmp/venv"
  uv pip install -q -p "$tmp/venv/bin/python" --no-deps \
    'nvidia-cuda-runtime-cu12==12.9.*' 'nvidia-cublas-cu12==12.9.*' \
    'nvidia-cufft-cu12==11.4.*' 'nvidia-curand-cu12==10.3.*' 'nvidia-cudnn-cu12==9.*'
  nv="$(echo "$tmp"/venv/lib/python3.12/site-packages/nvidia)"
  cp -a "$nv"/cuda_runtime/lib/libcudart.so.12 "$nv"/cublas/lib/libcublas{,Lt}.so.12 \
        "$nv"/cufft/lib/libcufft.so.11 "$nv"/curand/lib/libcurand.so.10 \
        "$nv"/cudnn/lib/libcudnn*.so.9 "$dest/lib/"
fi
du -sh "$dest"
