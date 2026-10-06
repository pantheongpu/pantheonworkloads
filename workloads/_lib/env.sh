# Sourced by the lib-* workloads' run.sh. Runs _lib/libkernels.py on the workload's target.
#
#   source "$PW_WORKLOAD_DIR/../_lib/env.sh"
#   lib_run lib-fft-linalg [--bench]        prints the result's JSON as the last stdout line; exits 77 / 1 as the contract says
#
# Target handling is ../_shared/torch_env.sh (cpu, gpu, sim:nvidia/*, sim:amd/*). Extra knobs:
#   PW_PYTHON         a Python with PyTorch; when unset and `python3` has none, the conda-forge CPU environment
#                     made by tools/pw-conda-torch.sh is used if it exists (cpu target only)
#   PW_LIB_SIZE       tiny | large: sizes of the benchmark variants (default large; tiny is a smoke test)
# Any "VirtualGPU error [" in the output fails the run (the simulator's way of saying it could not run a call
# and PyTorch carried on), as in the other PyTorch workloads.
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"

lib_run() {
  local group=$1; shift
  if [[ -z "${PW_PYTHON:-}" && "$PW_TARGET" == cpu ]] && ! python3 -c 'import torch' 2>/dev/null; then
    local cached="${PW_CONDA_PREFIX:-$HOME/.cache/pantheonworkloads/conda-torch}/envs/torch/bin/python"
    [[ -x "$cached" ]] && export PW_PYTHON="$cached"
  fi
  pw_setup torch
  case "$PW_TARGET" in
    cpu) export PW_DEVICE=cpu ;;
    *) export PW_DEVICE=cuda ;;
  esac
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  local out="$PW_OUT/$group.stdout" err="$PW_OUT/$group.stderr" status
  mkdir -p "$PW_OUT"
  pw_python "$PW_WORKLOAD_DIR/../_lib/libkernels.py" "$group" "$@" >"$out" 2>"$err"
  status=$?
  if grep -qh 'VirtualGPU error \[' "$out" "$err"; then
    echo "the simulator refused something:" >&2; grep -h -m5 'VirtualGPU error \[' "$out" "$err" >&2; exit 1
  fi
  if [[ $status == 77 ]]; then pw_skip "$(grep -m1 '^SKIP' "$out" | sed 's/^SKIP: *//') (see $out)"; fi
  if [[ $status != 0 ]]; then echo "exit $status; end of $err:" >&2; tail -8 "$err" >&2; exit 1; fi
  tail -n 1 "$out"
}
