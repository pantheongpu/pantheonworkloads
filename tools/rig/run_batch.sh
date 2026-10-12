#!/usr/bin/env bash
# run_batch.sh: see docs/rigs.md. All logic is in batch.py.
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/batch.py" "$@"
