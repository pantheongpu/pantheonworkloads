#!/usr/bin/env bash
# onnx-zoo-bertsquad-int8: ONNX model zoo BERT-Squad-12 int8 answers four questions over two short paragraphs with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py bertsquad; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" bertsquad
