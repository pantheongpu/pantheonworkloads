#!/usr/bin/env bash
# gtcrn-enhance-onnx-bench: GTCRN (streaming ONNX) speech-enhancement real-time factor over a 9.8 s clip on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py enhance --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" enhance --bench
