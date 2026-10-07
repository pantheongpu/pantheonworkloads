#!/usr/bin/env bash
# yolox-object-detection: YOLOX-S 640 COCO object detector (OpenCV Zoo, Apache-2.0) on three public-domain photos.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py yolox; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" yolox
