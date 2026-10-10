#!/usr/bin/env bash
# qwen3-vl-235b-a22b-bench: Qwen3-VL-235B-A22B-Instruct on PyTorch, bench mode. WRITTEN, NEVER RUN (docs/model-registry.md).
# Contract: docs/workload-contract.md; PyTorch discovery/install: ../_pytorch/env.sh. Exits 77 when the host lacks the GPUs, the disk,
# PyTorch or the model download. Real GPUs only.
set -uo pipefail
[[ "${PW_TARGET:-}" == gpu ]] || { echo "SKIP: target ${PW_TARGET:-?}: Qwen3-VL-235B-A22B-Instruct needs real GPUs (8 x 80 GiB)"; exit 77; }
. "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"
pw_require_gpus 8 80
pw_require_disk 439
export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/requirements-vlm.txt"
export PW_VLM_MODE=bench
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/../qwen3-vl-235b-a22b-pytorch/main.py" transformers accelerate PIL
