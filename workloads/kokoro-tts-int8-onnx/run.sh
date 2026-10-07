#!/usr/bin/env bash
# kokoro-tts-int8-onnx: Kokoro v0.19 (int8 ONNX) speaks two fixed phoneme strings; the waveform statistics are compared.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py tts; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" tts
