#!/usr/bin/env bash
# ppocr-rapidocr: PP-OCRv3 (via rapidocr_onnxruntime) reads the text of a screenshot (detect, classify, recognise).
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py ocr; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" ocr
