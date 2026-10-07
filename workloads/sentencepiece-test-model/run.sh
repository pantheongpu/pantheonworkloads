#!/usr/bin/env bash
# sentencepiece-test-model: SentencePiece 0.2.2 encodes fixed sentences with the upstream test unigram model, CPU only.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py sentencepiece; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" sentencepiece
