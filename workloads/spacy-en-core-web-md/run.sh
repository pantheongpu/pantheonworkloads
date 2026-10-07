#!/usr/bin/env bash
# spacy-en-core-web-md: spaCy en_core_web_md 3.8.0: parse, entities and static word-vector similarity.
# Contract: docs/workload-contract.md. Python env: tools/spacy-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py spacy-md; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/spacy-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" spacy-md
