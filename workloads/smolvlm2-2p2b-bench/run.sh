#!/usr/bin/env bash
# Bench twin of smolvlm2-2p2b-pytorch: the same script in bench mode. Contract: docs/workload-contract.md; PyTorch setup: ../_pytorch/env.sh.
# Exits 77 when there is no usable PyTorch/transformers or no model download. Real GPU only.
set -uo pipefail
export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/requirements-vlm.txt"
export PW_VLM_MODE=bench
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/../smolvlm2-2p2b-pytorch/main.py" transformers accelerate PIL
