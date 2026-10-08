#!/usr/bin/env bash
# sdxl-base-bench. Contract: docs/workload-contract.md; PyTorch discovery/install, simulator launch: ../_pytorch/env.sh.
# Exits 77 when there is no usable PyTorch/diffusers/transformers or no model download. Real GPU only.
set -uo pipefail
export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/requirements-diffusion.txt"
export PW_SDXL_MODE=bench
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/../sdxl-base-diffusers/main.py" diffusers accelerate
