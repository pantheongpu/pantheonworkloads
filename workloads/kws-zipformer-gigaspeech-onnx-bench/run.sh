#!/usr/bin/env bash
# kws-zipformer-gigaspeech-onnx-bench: Streaming Zipformer2 keyword spotter (ONNX), real-time factor over two speech clips on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py kws --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" kws --bench
