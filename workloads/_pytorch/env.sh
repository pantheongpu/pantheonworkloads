#!/usr/bin/env bash
# Shared by the pytorch-* workloads' run.sh (source it). Finds or builds a Python with the right
# PyTorch for $PW_TARGET and runs a script on it, on the real device or inside pantheonsim.
#
#   pw_torch_run <script.py> <module> ...   exits 77 when there is no usable torch/model here
#
# Environment knobs:
#   PW_TORCH_PYTHON   a Python to use as it is (skips discovery and install)
#   PW_TORCH_FLAVOR   cpu | cu130 | rocm: which PyTorch build (default: by target)
#   PW_VENV_ROOT      where venvs are made (default ~/.cache/pantheonworkloads/venvs)
#   PW_NO_INSTALL=1   never create a venv or pip install (SKIP instead)
#   PW_CACHE          model cache (default ~/.cache/pantheonworkloads); weights are never in the repo
#   PW_TORCH_INDEX_<FLAVOR>  override the PyTorch wheel index URL for a flavour (CPU, CU130, ROCM)
# Discovery also reuses pantheonsim's own venvs (~/.local/share/torch-cu13*, torch-rocm*,
# VGPU_TORCH_CUDA_PYTHON, VGPU_TORCH_PYTHON) when they have the modules the script imports.
# Not run in the environment it was written in (no route to download.pytorch.org or
# huggingface.co): see docs/models.md.

pw_skip() { echo "SKIP: $*"; exit 77; }

_pw_have() {  # python flavor modules...
  local py=$1 flavor=$2; shift 2
  [[ -x "$py" ]] || return 1
  "$py" - "$flavor" "$@" >/dev/null 2>&1 <<'PY'
import importlib, sys
import torch
flavor, mods = sys.argv[1], sys.argv[2:]
if flavor == "cu130" and not (torch.version.cuda or "").startswith("13"): sys.exit(1)
if flavor == "rocm" and not torch.version.hip: sys.exit(1)
for m in mods: importlib.import_module(m)
PY
}

_pw_flavor() {   # sets PW_FLAVOR (not a subshell: it may exit 77)
  if [[ -n "${PW_TORCH_FLAVOR:-}" ]]; then PW_FLAVOR=$PW_TORCH_FLAVOR; return; fi
  case "$PW_TARGET" in
    cpu) PW_FLAVOR=cpu ;;
    sim:nvidia/*) PW_FLAVOR=cu130 ;;
    sim:amd/*) PW_FLAVOR=rocm ;;
    gpu)
      if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then PW_FLAVOR=cu130
      elif [[ -e /dev/kfd ]]; then PW_FLAVOR=rocm
      else pw_skip "target gpu, but no NVIDIA or AMD GPU was found"; fi ;;
    *) pw_skip "target $PW_TARGET is not supported by the pytorch workloads" ;;
  esac
}

_pw_install() {  # flavor venv
  local flavor=$1 venv=$2 here index
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  case "$flavor" in
    cpu) index="${PW_TORCH_INDEX_CPU:-https://download.pytorch.org/whl/cpu}" ;;
    cu130) index="${PW_TORCH_INDEX_CU130:-https://download.pytorch.org/whl/cu130}" ;;
    rocm) index="${PW_TORCH_INDEX_ROCM:-https://download.pytorch.org/whl/rocm7.1}" ;;   # unverified index name: override if needed
    *) pw_skip "unknown PW_TORCH_FLAVOR $flavor" ;;
  esac
  [[ "${PW_NO_INSTALL:-0}" != 1 ]] || pw_skip "no Python with $flavor PyTorch (PW_NO_INSTALL=1)"
  echo "creating $venv ($flavor PyTorch from $index)" >&2
  python3 -m venv "$venv" >&2 || { rm -rf "$venv"; pw_skip "python3 -m venv failed"; }
  "$venv/bin/pip" install -q --index-url "$index" --extra-index-url https://pypi.org/simple \
      -r "$here/requirements-torch.txt" >&2 \
    && "$venv/bin/pip" install -q -r "$here/requirements-common.txt" >&2 \
    || { rm -rf "$venv"; pw_skip "could not install PyTorch from $index (no network, or the pin is not on that index)"; }
}

_pw_python() {  # modules... -> sets PW_PY
  local flavor c venv
  _pw_flavor; flavor=$PW_FLAVOR
  venv="${PW_VENV_ROOT:-$HOME/.cache/pantheonworkloads/venvs}/$flavor"
  local cands=("${PW_TORCH_PYTHON:-}" "$venv/bin/python")
  case "$flavor" in
    cu130) cands+=("${VGPU_TORCH_CUDA_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-cu13*/bin/python 2>/dev/null)) ;;
    rocm) cands+=("${VGPU_TORCH_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-rocm*/bin/python 2>/dev/null)) ;;
  esac
  for c in "${cands[@]}"; do
    [[ -n "$c" ]] && _pw_have "$c" "$flavor" "$@" && { PW_PY=$c; return; }
  done
  [[ -z "${PW_TORCH_PYTHON:-}" ]] || pw_skip "PW_TORCH_PYTHON lacks torch or one of: $*"
  _pw_install "$flavor" "$venv"
  _pw_have "$venv/bin/python" "$flavor" "$@" || pw_skip "the new venv cannot import: $*"
  PW_PY="$venv/bin/python"
}

# pw_torch_run script.py module...
pw_torch_run() {
  local script=$1; shift
  _pw_python torch "$@"
  export PW_CACHE="${PW_CACHE:-$HOME/.cache/pantheonworkloads}" PW_RESULT_FILE="$PW_OUT/$(basename "$PW_WORKLOAD_DIR").result.json"
  export PW_TORCH_FLAVOR_USED=$PW_FLAVOR
  rm -f "$PW_RESULT_FILE"
  local log="$PW_OUT/$(basename "$PW_WORKLOAD_DIR").run.log" status tmp
  local cap=() run=()
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  export TRITON_CACHE_DIR="$tmp/triton" TORCHINDUCTOR_CACHE_DIR="$tmp/inductor"
  case "$PW_TARGET" in
    cpu) PW_DEVICE=cpu; export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=; run=("$PW_PY" "$script") ;;
    gpu) PW_DEVICE=cuda; run=("$PW_PY" "$script") ;;
    sim:*)
      [[ -n "${PANTHEONSIM_DIR:-}" ]] || pw_skip "set PANTHEONSIM_DIR to a built pantheonsim checkout"
      local build="$PANTHEONSIM_DIR/build"
      [[ -x "$build/vgpu" ]] || pw_skip "$build/vgpu is not built"
      PW_DEVICE=cuda
      # PyTorch on a simulated device can take many GB of host memory and these machines are
      # shared: hold the run to a cap where systemd can, as pantheonsim's own PyTorch tests do.
      if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
        cap=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-10G}" -p MemorySwapMax=0)
      fi
      if [[ $PW_TARGET == sim:nvidia/* ]]; then
        [[ -e "$build/shim/libcudart.so.13" ]] || pw_skip "no CUDA 13 runtime shim in $build/shim"
        run=("$build/vgpu" run --gpu "$PW_SIM_PROFILE" --preload "$PW_PY" "$script")
      else
        # PyTorch for ROCm loads the libamdhip64 beside its own libraries, so a tree of links to
        # the installation is made with the simulator's HIP and ROCm SMI libraries in their place
        # (as amd/tests/e2e/run_pytorch.sh does).
        local shim="$build/shim/libamdhip64.so.7" pkg f
        [[ -e "$shim" ]] || pw_skip "no HIP shim in $build/shim"
        pkg=$("$PW_PY" -c 'import os, torch; print(os.path.dirname(torch.__file__))')
        mkdir -p "$tmp/tree/torch/lib"
        for f in "$pkg"/*; do [[ "$(basename "$f")" == lib ]] || ln -s "$f" "$tmp/tree/torch/"; done
        for f in "$pkg"/lib/*; do
          case "$(basename "$f")" in libamdhip64.so|librocm_smi64.so) ;; *) ln -s "$f" "$tmp/tree/torch/lib/" ;; esac
        done
        ln -s "$(readlink -f "$shim")" "$tmp/tree/torch/lib/libamdhip64.so"
        ln -s "$(readlink -f "$build/shim/librocm_smi64.so")" "$tmp/tree/torch/lib/librocm_smi64.so"
        export PYTHONPATH="$tmp/tree${PYTHONPATH:+:$PYTHONPATH}" VGPU_GPU="$PW_SIM_PROFILE" VGPU_DEVICE_COUNT=1 \
               VGPU_MEMORY_RAM_MB="${VGPU_MEMORY_RAM_MB:-4096}"
        run=("$PW_PY" "$script")
      fi ;;
  esac
  export PW_DEVICE
  (cd "$tmp" && "${cap[@]+"${cap[@]}"}" "${run[@]}") >"$log" 2>&1
  status=$?
  if [[ $status == 77 ]]; then pw_skip "$(grep -m1 '^SKIP' "$log" | sed 's/^SKIP: *//' || true) (see $log)"; fi
  # The simulator reports a kernel or library call it could not run as "VirtualGPU error [...]"
  # and PyTorch can carry on with garbage, so any such line fails the run.
  if grep -q 'VirtualGPU error \[' "$log"; then
    echo "the simulator refused something:" >&2; grep -m5 'VirtualGPU error \[' "$log" >&2; exit 1
  fi
  if [[ $status != 0 ]]; then echo "exit $status; end of $log:" >&2; tail -8 "$log" >&2; exit 1; fi
  [[ -s "$PW_RESULT_FILE" ]] || { echo "the script wrote no result; end of $log:" >&2; tail -8 "$log" >&2; exit 1; }
  cat "$PW_RESULT_FILE"
}
