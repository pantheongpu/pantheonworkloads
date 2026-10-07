#!/usr/bin/env bash
# sface-face-embedding: SFace face-embedding network (OpenCV Zoo, Apache-2.0): embedding of an aligned astronaut face, its mirror image and a non-face.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py sface; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" sface
