#!/usr/bin/env bash
# kws-zipformer-gigaspeech-onnx: Streaming Zipformer2 keyword spotter (3.3M parameters, ONNX) finds keywords in two 16 kHz speech clips with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py kws; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" kws
