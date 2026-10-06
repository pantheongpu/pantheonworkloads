#!/usr/bin/env bash
# gpt-train-bf16: 20 AdamW steps of a tiny character-level GPT in bf16; the output is the 20 losses.
# Contract: docs/workload-contract.md. The engine, data and the target handling are shared with
# the other gpt-train workloads (../_shared). Exits 77 when torch or a GPU is missing, or the GPU has no bf16.
#
#   PW_PYTHON   a Python with PyTorch (cpu, gpu targets); sim: targets use pantheonsim's variables,
#               see ../_shared/torch_env.sh
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
pw_setup torch
# cuBLAS needs this for deterministic results (torch.use_deterministic_algorithms)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
pw_python "$PW_WORKLOAD_DIR/../_shared/gpt_train.py" --mode functional --dtype bf16
