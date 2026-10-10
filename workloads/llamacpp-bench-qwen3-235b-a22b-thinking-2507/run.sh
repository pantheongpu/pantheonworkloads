#!/usr/bin/env bash
# llamacpp-bench-qwen3-235b-a22b-thinking-2507: llama-bench tokens/s with Qwen3-235B-A22B-Thinking-2507 (Q4_K_M GGUF, 132.4 GiB, 2 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 133 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3-235B-A22B-Thinking-2507 needs real GPUs (2 x 80 GiB)"
lc_require_gpus 2 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF/resolve/3eba50b4e3ea2c0df1c43e8c8f5ce2db22645e81" "$here/model.sha256" 133
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
