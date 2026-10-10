#!/usr/bin/env bash
# llamacpp-bench-llama4-scout-17b-16e-instruct: llama-bench tokens/s with Llama-4-Scout-17B-16E-Instruct (Q4_K_M GGUF, 60.9 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 61 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Llama-4-Scout-17B-16E-Instruct needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/42675345da11ade9203a5187595da7b74d4ff2ac" "$here/model.sha256" 61
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
