#!/usr/bin/env bash
# bge-reranker-v2-m3-bench: bge-reranker-v2-m3 on PyTorch, bench mode. Run on an A10G (stage 2a, docs/model-registry.md).
# Contract: docs/workload-contract.md; PyTorch discovery/install: ../_pytorch/env.sh. Exits 77 when the host lacks the GPUs, the disk,
# PyTorch or the model download. Real GPUs only.
set -uo pipefail
[[ "${PW_TARGET:-}" == gpu ]] || { echo "SKIP: target ${PW_TARGET:-?}: bge-reranker-v2-m3 needs real GPUs (1 x 10 GiB)"; exit 77; }
. "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"
pw_require_gpus 1 10
pw_require_disk 3
export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/requirements-vlm.txt"
export PW_RETRIEVAL_MODE=bench
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/../bge-reranker-v2-m3-pytorch/main.py" transformers accelerate
