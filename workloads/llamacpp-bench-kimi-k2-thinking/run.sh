#!/usr/bin/env bash
# llamacpp-bench-kimi-k2-thinking: llama-bench tokens/s with Kimi-K2-Thinking (Q4_K_M GGUF, 578.6 GiB, 4 x 180 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 579 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Kimi-K2-Thinking needs real GPUs (4 x 180 GiB)"
lc_require_gpus 4 180
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF/resolve/7e4f5d413a9bb469174b2cc9e47d457d16bc6702" "$here/model.sha256" 579
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
