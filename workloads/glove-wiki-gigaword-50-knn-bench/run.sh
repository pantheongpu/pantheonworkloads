#!/usr/bin/env bash
# glove-wiki-gigaword-50-knn-bench: GloVe 50d nearest-neighbour search (MatMul+TopK over 400k words), queries per second on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py glove --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" glove --bench
