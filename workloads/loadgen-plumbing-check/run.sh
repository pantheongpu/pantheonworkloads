#!/usr/bin/env bash
# loadgen-plumbing-check: MLCommons LoadGen (pip mlcommons-loadgen, pinned) drives a trivial CPU
# SUT through the Offline and Server scenarios. NOT an MLPerf result: see manifest.yaml.
# Contract: docs/workload-contract.md. Needs python3 with venv, and network on the first run (pip).
#
#   PW_CACHE   cache directory for the virtualenv (default ~/.cache/pantheonworkloads)
set -uo pipefail
LOADGEN_VERSION=6.0.17
skip() { echo "SKIP: $*"; exit 77; }
case "$PW_TARGET" in cpu) ;; *) skip "target $PW_TARGET is not supported by this workload" ;; esac
command -v python3 >/dev/null || skip "python3 is not installed"
venv="${PW_CACHE:-$HOME/.cache/pantheonworkloads}/loadgen-$LOADGEN_VERSION-venv"
if ! "$venv/bin/python" -c "import mlperf_loadgen" 2>/dev/null; then
  rm -rf "$venv"
  python3 -m venv "$venv" >&2 || skip "python3 -m venv failed"
  "$venv/bin/pip" install -q --only-binary :all: "mlcommons-loadgen==$LOADGEN_VERSION" >&2 \
    || skip "cannot install mlcommons-loadgen==$LOADGEN_VERSION (offline, or no wheel for this Python)"
fi
exec "$venv/bin/python" -I "$PW_WORKLOAD_DIR/plumbing.py" "$PW_OUT/loadgen"
