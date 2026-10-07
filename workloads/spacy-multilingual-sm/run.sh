#!/usr/bin/env bash
# spacy-multilingual-sm: spaCy Russian, Ukrainian, Norwegian and multilingual-NER small pipelines (3.8.0) on fixed sentences.
# Contract: docs/workload-contract.md. Python env: tools/spacy-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py spacy-multi; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/spacy-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" spacy-multi
