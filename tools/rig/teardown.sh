#!/usr/bin/env bash
# teardown.sh: see docs/rigs.md. All logic is in rig.py.
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rig.py" teardown "$@"
