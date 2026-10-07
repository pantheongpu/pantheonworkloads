#!/usr/bin/env bash
# onnx-zoo-shufflenet-v2: ONNX model zoo ShuffleNet-v2-12 classifies the zoo's own preprocessed test image with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py shufflenet; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" shufflenet
