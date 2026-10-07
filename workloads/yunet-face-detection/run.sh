#!/usr/bin/env bash
# yunet-face-detection: YuNet face detector (OpenCV Zoo, MIT) finds the face and its five landmarks in a NASA astronaut photo and finds none in a cat photo.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/ort_tasks.py yunet; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/ort_tasks.py" yunet
