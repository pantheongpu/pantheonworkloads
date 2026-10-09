#!/usr/bin/env bash
# Contract: docs/workload-contract.md; PyTorch discovery/install, simulator launch: ../_pytorch/env.sh.
# Exits 77 when there is no usable PyTorch/transformers or no model download. Real GPU only.
set -uo pipefail
export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/requirements-vlm.txt"
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/main.py" transformers accelerate PIL
