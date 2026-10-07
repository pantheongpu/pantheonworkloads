#!/usr/bin/env bash
# spacy-multilingual-sm-bench: spaCy ru/uk/nb/xx small pipelines, words per second per pipeline, on a real GPU.
# Contract: docs/workload-contract.md. Python env: tools/spacy-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py spacy-multi --bench; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/spacy-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" spacy-multi --bench
