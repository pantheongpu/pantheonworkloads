#!/usr/bin/env bash
# arch-bench: prefill and decode tokens/s of a random-weight decoder-only architecture at a larger size,
# for real GPUs. Engine: ../_shared/arch_cases.py --mode bench. Contract: docs/workload-contract.md.
#
#   PW_ARCH_BENCH_ARCH   llama (default) | mistral | qwen2 | gemma | phi3 | mixtral
#   PW_ARCH_BENCH_SIZE   small (4 layers, 512 wide; default) | medium (16 x 2048) | large (32 x 4096, ~6.5 B parameters)
#   PW_ARCH_BENCH_REPS   timed repetitions (default 3), PW_ARCH_ATTN (sdpa | eager)
# Exits 77 for simulated targets (a simulated GPU's time is not GPU time), or without torch / transformers / a GPU.
set -o pipefail
case "$PW_TARGET" in sim:*) echo "SKIP: benchmark numbers are for real targets only"; exit 77 ;; esac
source "$PW_WORKLOAD_DIR/../_shared/arch_run.sh"
pw_arch --mode bench
