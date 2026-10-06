#!/usr/bin/env bash
# gpt-train-bench: training throughput of nanoGPT's "baby GPT" shape (6 layers, 6 heads, 384 wide,
# batch 32 x 256 tokens), 30 timed AdamW steps after 1 warm-up step, in fp32 and bf16 autocast.
# Contract: docs/workload-contract.md. Engine and target handling: ../_shared.
# Exits 77 when torch or a GPU is missing. Where the GPU has no bf16 only fp32 is measured (and the
# metrics say so by lacking the bf16 keys); with no bf16 and no fp32 it exits 77.
#
#   PW_PYTHON   a Python with PyTorch (cpu, gpu targets)
#   PW_BENCH_STEPS   timed steps (default 30)
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
case "$PW_TARGET" in sim:*) pw_skip "benchmark numbers are for real targets only; a simulated GPU's time is not GPU time" ;; esac
pw_setup torch
pw_python "$PW_WORKLOAD_DIR/../_shared/gpt_train.py" --mode bench --dtype fp32,bf16 --steps "${PW_BENCH_STEPS:-30}"
