#!/usr/bin/env bash
# onnx-zoo-bertsquad-int8-bench: BERT-Squad-12 int8 (ONNX), questions per second at sequence length 256 on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py bertsquad --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" bertsquad --bench
