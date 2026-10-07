#!/usr/bin/env bash
# wespeaker-resnet34-onnx-bench: WeSpeaker ResNet34 (ONNX), speaker-embedding real-time factor over three 10 s clips on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py speaker --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" speaker --bench
