#!/usr/bin/env bash
# llamacpp-bench-qwen3-embedding-8b: llama-bench tokens/s with Qwen3-Embedding-8B (Q8_0 GGUF, 7.5 GiB, 1 x 15 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 8 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3-Embedding-8B needs real GPUs (1 x 15 GiB)"
lc_require_gpus 1 15
export LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS="${LLAMACPP_EXTRA_TARGETS:-llama-embedding}"
lc_setup
lc_fetch_shards "https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/69d0e58a13e463cd99a9b83e3f5fee7c10265fab" "$here/model.sha256" 8
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
