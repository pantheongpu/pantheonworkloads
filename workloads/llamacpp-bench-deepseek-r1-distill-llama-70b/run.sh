#!/usr/bin/env bash
# llamacpp-bench-deepseek-r1-distill-llama-70b: llama-bench tokens/s with DeepSeek-R1-Distill-Llama-70B (Q4_K_M GGUF, 39.6 GiB, 1 x 48 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 40 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: DeepSeek-R1-Distill-Llama-70B needs real GPUs (1 x 48 GiB)"
lc_require_gpus 1 48
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/resolve/732dd974083ea5877d7b6d788b36fe7c2e5eab36" "$here/model.sha256" 40
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
