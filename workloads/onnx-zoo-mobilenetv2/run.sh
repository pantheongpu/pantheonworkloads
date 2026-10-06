#!/usr/bin/env bash
# onnx-zoo-mobilenetv2: ONNX model zoo MobileNetV2-12 classifies the zoo's own preprocessed test image with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py mobilenetv2; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" mobilenetv2
