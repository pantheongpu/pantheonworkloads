#!/usr/bin/env bash
# minilm-l6-v2-onnx-bench: all-MiniLM-L6-v2 (ONNX), sentences per second over a 36-sentence batch on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py minilm --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" minilm --bench
