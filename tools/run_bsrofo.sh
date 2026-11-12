#!/usr/bin/env bash
# Run BS-Roformer SW (GPU, tools/.venv-gpu) on the A/B clips; inputs are folded to mono.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$here/../out/ab/src
"$here/.venv-gpu/bin/python" -I "$here/run_bsrofo.py" "$here/../models/ext/bsrofo" bsrofo "$src"/clip[1-8].wav
