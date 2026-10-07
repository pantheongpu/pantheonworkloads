#!/usr/bin/env bash
# minilm-l6-v2-onnx: all-MiniLM-L6-v2 (ONNX) embeds six fixed sentences with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py minilm; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" minilm
