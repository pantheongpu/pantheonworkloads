#!/usr/bin/env bash
# py3langid-wheel: py3langid (langid.py fork, BSD-3-Clause) identifies the language of ten sentences, CPU only.
# Contract: docs/workload-contract.md. Python env: tools/ort-env.sh (exit 77 when absent);
# payload: tools/text_tasks.py langid; pinned files: model.sha256.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$here/tools/ort-env.sh"
exec "$PW_PY" -I "$here/tools/text_tasks.py" langid
