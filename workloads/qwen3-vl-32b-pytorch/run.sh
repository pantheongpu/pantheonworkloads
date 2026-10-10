#!/usr/bin/env bash
# qwen3-vl-32b-pytorch: Qwen3-VL-32B-Instruct on PyTorch, functional. WRITTEN, NEVER RUN (docs/model-registry.md).
# Contract: docs/workload-contract.md; PyTorch discovery/install: ../_pytorch/env.sh. Exits 77 when the host lacks the GPUs, the disk,
# PyTorch or the model download. Real GPUs only.
set -uo pipefail
[[ "${PW_TARGET:-}" == gpu ]] || { echo "SKIP: target ${PW_TARGET:-?}: Qwen3-VL-32B-Instruct needs real GPUs (1 x 80 GiB)"; exit 77; }
. "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"
pw_require_gpus 1 80
pw_require_disk 63
export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/requirements-vlm.txt"
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/main.py" transformers accelerate PIL
