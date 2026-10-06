#!/usr/bin/env bash
# Source this from a spacy-* workload's run.sh. Sets PW_PY to a Python with spaCy and the pinned
# en_core_web_sm wheel; creates a venv when there is none; exits 77 when that is impossible.
#
#   PW_SPACY_PYTHON  use this Python as it is (it must import spacy and en_core_web_sm)
#   PW_VENV_ROOT     venv parent (default ~/.cache/pantheonworkloads/venvs): spacy-cpu, spacy-gpu
#   PW_NO_INSTALL=1  never create a venv or pip install
# The gpu venv adds cupy-cuda12x (spaCy's NVIDIA GPU path); AMD GPUs are not supported by spaCy here.
pw_skip() { echo "SKIP: $*"; exit 77; }

_sp_have() { [[ -x "$1" ]] && "$1" -I -c 'import spacy, en_core_web_sm' >/dev/null 2>&1; }

_sp_python() {
  local here flavor venv wheel
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -n "${PW_SPACY_PYTHON:-}" ]]; then
    _sp_have "$PW_SPACY_PYTHON" || pw_skip "PW_SPACY_PYTHON cannot import spacy and en_core_web_sm"
    PW_PY=$PW_SPACY_PYTHON; return
  fi
  case "$PW_TARGET" in
    cpu) flavor=cpu ;;
    gpu) command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
           || pw_skip "target gpu: no NVIDIA GPU found (spaCy's GPU path is CUDA only)"
         flavor=gpu ;;
    *) pw_skip "target $PW_TARGET is not supported by the spacy workloads" ;;
  esac
  venv="${PW_VENV_ROOT:-$HOME/.cache/pantheonworkloads/venvs}/spacy-$flavor"
  if ! _sp_have "$venv/bin/python"; then
    [[ "${PW_NO_INSTALL:-0}" != 1 ]] || pw_skip "no spacy-$flavor venv (PW_NO_INSTALL=1)"
    wheel=$(python3 -I "$here/ort_assets.py" "$PW_WORKLOAD_DIR" en_core_web_sm-3.8.0-py3-none-any.whl) \
      || { echo "$wheel"; exit 77; }
    wheel=$(tail -1 <<<"$wheel")
    echo "creating $venv" >&2
    python3 -m venv "$venv" >&2 || { rm -rf "$venv"; pw_skip "python3 -m venv failed"; }
    local extra=(); [[ $flavor == gpu ]] && extra=("cupy-cuda12x")
    "$venv/bin/pip" install -q -r "$here/spacy-requirements.txt" "${extra[@]}" >&2 \
      && "$venv/bin/pip" install -q --no-deps "$wheel" >&2 \
      || { rm -rf "$venv"; pw_skip "could not pip install spaCy and the model wheel (no network?)"; }
  fi
  _sp_have "$venv/bin/python" || pw_skip "the spacy-$flavor venv cannot import spacy and en_core_web_sm"
  PW_PY=$venv/bin/python
}
_sp_python
