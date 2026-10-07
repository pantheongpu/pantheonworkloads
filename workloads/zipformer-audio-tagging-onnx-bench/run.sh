#!/usr/bin/env bash
# zipformer-audio-tagging-onnx-bench: Zipformer (small, AudioSet, ONNX) audio-tagging real-time factor over six sound clips on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py tag --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" tag --bench
