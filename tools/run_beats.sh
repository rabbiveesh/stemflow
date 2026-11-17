#!/usr/bin/env bash
# Install Beat This! into tools/.venv-gpu (once), then run the beat-tracker
# comparison on the A/B clips. Output: out/beats/<clip>-<tracker>.{json,flac}
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=$here/.venv-gpu/bin/python
if ! "$py" -c "import beat_this" 2>/dev/null; then
  uv pip install --python "$py" "beat_this @ git+https://github.com/CPJKU/beat_this@main" soxr
fi
"$py" -I "$here/run_beats.py" "$here/../out/beats" "$here"/../out/ab/src/clip[1-8].wav
