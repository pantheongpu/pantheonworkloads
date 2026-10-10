#!/usr/bin/env bash
# cogvideox-5b-bench: CogVideoX-5b on PyTorch, bench mode. VERIFIED on an L40S (stage 2a, Tier B; docs/model-registry.md).
# Contract: docs/workload-contract.md; PyTorch discovery/install: ../_pytorch/env.sh. Exits 77 when the host lacks the GPUs, the disk,
# PyTorch or the model download. Real GPUs only.
set -uo pipefail
[[ "${PW_TARGET:-}" == gpu ]] || { echo "SKIP: target ${PW_TARGET:-?}: CogVideoX-5b needs real GPUs (1 x 40 GiB)"; exit 77; }
. "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"
pw_require_gpus 1 40
pw_require_disk 21
export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/requirements-diffusion.txt"
export PW_DIFFUSION_MODE=bench
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/../cogvideox-5b-diffusers/main.py" diffusers accelerate
