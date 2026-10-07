#!/usr/bin/env bash
# gtcrn-enhance-onnx: GTCRN (streaming ONNX, 0.5 MB) denoises a 9.8 s noisy speech clip frame by frame with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py enhance; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" enhance
