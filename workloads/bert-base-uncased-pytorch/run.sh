#!/usr/bin/env bash
# bert-base-uncased-pytorch. Contract: docs/workload-contract.md; PyTorch discovery/install, simulator launch: ../_pytorch/env.sh.
# Exits 77 when there is no usable PyTorch, no pantheonsim build (sim: targets) or no model download.
set -uo pipefail
source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"
pw_torch_run "$PW_WORKLOAD_DIR/main.py" transformers
