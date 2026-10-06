#!/usr/bin/env bash
# Source this from an ort-* workload's run.sh. Sets PW_PY to a Python that has onnxruntime and the
# other packages of tools/ort-requirements.txt, creating a venv when there is none; exits 77 when
# that is impossible. Never torch: these workloads run on onnxruntime only.
#
#   PW_ORT_PYTHON   use this Python as it is (e.g. an onnxruntime-rocm / MIGraphX build for AMD)
#   PW_VENV_ROOT    venv parent (default ~/.cache/pantheonworkloads/venvs); venvs are ort-cpu, ort-gpu
#   PW_NO_INSTALL=1 never create a venv or pip install
# Target cpu -> venv ort-cpu (onnxruntime). Target gpu -> venv ort-gpu (onnxruntime-gpu: its CUDA EP
# also needs CUDA 12 and cuDNN 9 libraries on the machine; tools/ort_tasks.py exits 77 when the GPU
# provider is not usable, it never falls back to the CPU).
pw_skip() { echo "SKIP: $*"; exit 77; }

_ort_have() { [[ -x "$1" ]] && "$1" -I -c 'import onnxruntime, numpy, onnx, rapidocr_onnxruntime, cv2' >/dev/null 2>&1; }

_ort_python() {
  local here flavor venv
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -n "${PW_ORT_PYTHON:-}" ]]; then
    _ort_have "$PW_ORT_PYTHON" || pw_skip "PW_ORT_PYTHON lacks onnxruntime, numpy, onnx or rapidocr_onnxruntime"
    PW_PY=$PW_ORT_PYTHON; return
  fi
  case "$PW_TARGET" in
    cpu) flavor=cpu ;;
    gpu) command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
           || pw_skip "target gpu: no NVIDIA GPU found (for AMD set PW_ORT_PYTHON to a ROCm/MIGraphX onnxruntime build)"
         flavor=gpu ;;
    *) pw_skip "target $PW_TARGET is not supported by the ort workloads" ;;
  esac
  venv="${PW_VENV_ROOT:-$HOME/.cache/pantheonworkloads/venvs}/ort-$flavor"
  if ! _ort_have "$venv/bin/python"; then
    [[ "${PW_NO_INSTALL:-0}" != 1 ]] || pw_skip "no ort-$flavor venv (PW_NO_INSTALL=1)"
    echo "creating $venv" >&2
    python3 -m venv "$venv" >&2 || { rm -rf "$venv"; pw_skip "python3 -m venv failed"; }
    grep -v '^onnxruntime==' "$here/ort-requirements.txt" > "$venv/requirements.txt"
    local rt; rt=$(grep '^onnxruntime==' "$here/ort-requirements.txt")
    [[ $flavor == gpu ]] && rt=${rt/onnxruntime/onnxruntime-gpu}
    "$venv/bin/pip" install -q "$rt" -r "$venv/requirements.txt" >&2 \
      || { rm -rf "$venv"; pw_skip "could not pip install $rt and tools/ort-requirements.txt (no network?)"; }
  fi
  _ort_have "$venv/bin/python" || pw_skip "the ort-$flavor venv cannot import its packages"
  PW_PY=$venv/bin/python
}
_ort_python
