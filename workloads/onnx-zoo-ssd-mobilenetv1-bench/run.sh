#!/usr/bin/env bash
# onnx-zoo-ssd-mobilenetv1-bench: ONNX model zoo SSD-MobileNetV1-12 COCO detector throughput on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py ssdmobilenet --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" ssdmobilenet --bench
