# Sourced by the arch-* workloads' run.sh. Sets the target up (torch_env.sh) and gives `pw_arch`, which
# runs arch_cases.py with the arguments given.
#
#   PW_PYTHON          a Python with PyTorch and transformers (cpu, gpu targets)
#   PW_ARCH_AUTO_ENV=1 on cpu/gpu without PW_PYTHON: build one with tools/torch-cpu-env.sh (CPU-only
#                      PyTorch + transformers from conda-forge; downloads ~1 GB once) and use that
#   PW_ARCH_ATTN, PW_ARCH_EXPERTS, PW_ARCH_ONLY: see arch_cases.py
# These workloads build tiny random-weight models from transformers' configs: no weights, no downloads.
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
if [[ "${PW_ARCH_AUTO_ENV:-}" == 1 && -z "${PW_PYTHON:-}" && "$PW_TARGET" == cpu ]]; then
  PW_PYTHON=$("$PW_WORKLOAD_DIR/../../tools/torch-cpu-env.sh") || pw_skip "could not build a CPU PyTorch + transformers environment"
  export PW_PYTHON
fi
pw_setup torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # deterministic cuBLAS (torch.use_deterministic_algorithms)
pw_arch() { pw_python "$PW_WORKLOAD_DIR/../_shared/arch_cases.py" "$@"; }
