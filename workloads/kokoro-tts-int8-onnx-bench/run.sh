#!/usr/bin/env bash
# kokoro-tts-int8-onnx-bench: Kokoro v0.19 (int8 ONNX) text-to-speech real-time factor on two sentences on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py tts --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" tts --bench
