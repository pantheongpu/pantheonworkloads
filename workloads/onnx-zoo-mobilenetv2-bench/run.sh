#!/usr/bin/env bash
# onnx-zoo-mobilenetv2-bench: ONNX model zoo MobileNetV2-12: images per second at batch 32 on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py mobilenetv2 --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" mobilenetv2 --bench
