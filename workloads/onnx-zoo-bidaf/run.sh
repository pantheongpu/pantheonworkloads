#!/usr/bin/env bash
# onnx-zoo-bidaf: ONNX model zoo BiDAF-9 answers 16 SQuAD questions with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py bidaf; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" bidaf
