#!/usr/bin/env bash
# wespeaker-resnet34-onnx: WeSpeaker ResNet34 (VoxCeleb, ONNX) turns three 10 s clips into 256-d speaker embeddings with onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/speech_tasks.py speaker; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/speech_tasks.py" speaker
