#!/usr/bin/env bash
# spacy-en-core-web-sm-bench. Contract: docs/workload-contract.md. Env: tools/spacy-env.sh (exit 77 when absent);
# payload: tools/spacy_task.py; pinned wheel: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/spacy-env.sh"
exec "$PW_PY" -I "$here/tools/spacy_task.py" --bench
