#!/usr/bin/env bash
# glove-wiki-gigaword-50-knn: GloVe 50d word vectors (400k words): nearest neighbours and analogies as a MatMul+TopK graph on onnxruntime.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py glove; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" glove
