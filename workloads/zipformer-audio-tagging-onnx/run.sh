#!/usr/bin/env bash
# zipformer-audio-tagging-onnx: Zipformer (small, AudioSet, ONNX) tags six 3-10 s sound clips with the top AudioSet classes using onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py tag; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" tag
