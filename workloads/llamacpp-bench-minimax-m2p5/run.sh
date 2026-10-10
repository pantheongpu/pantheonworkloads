#!/usr/bin/env bash
# llamacpp-bench-minimax-m2p5: llama-bench tokens/s with MiniMax-M2.5 (Q4_K_M GGUF, 128.8 GiB, 2 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 129 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: MiniMax-M2.5 needs real GPUs (2 x 80 GiB)"
lc_require_gpus 2 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/MiniMax-M2.5-GGUF/resolve/7c50dca0e5592483ad308ecffc876aecac725660" "$here/model.sha256" 129
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
