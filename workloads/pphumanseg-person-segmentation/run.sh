#!/usr/bin/env bash
# pphumanseg-person-segmentation: PP-HumanSeg person segmentation (OpenCV Zoo, Apache-2.0) on a photo of a person and a photo of a cat.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py pphumanseg; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" pphumanseg
