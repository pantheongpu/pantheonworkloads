#!/usr/bin/env bash
# lib-kernels-bench: throughput of the lib-* groups' library paths at benchmark sizes; real GPUs only.
# Contract: docs/workload-contract.md. Exits 77 on a simulated target or without a GPU / PyTorch.
#   PW_LIB_SIZE  large (default) | tiny (a smoke test of the code path; numbers from it mean nothing)
source "$PW_WORKLOAD_DIR/../_lib/env.sh"
case "$PW_TARGET" in sim:*) pw_skip "benchmark numbers are for real targets only; a simulated GPU's time is not GPU time" ;; esac
lib_run lib-fft-linalg,lib-sparse-embedding,lib-rnn-conv,lib-attention-precision,lib-composite-blocks --bench
