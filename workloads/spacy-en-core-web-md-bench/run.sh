#!/usr/bin/env bash
# spacy-en-core-web-md-bench: spaCy en_core_web_md 3.8.0, words per second over 1000 short documents, on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/spacy-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py spacy-md --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/spacy-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" spacy-md --bench
