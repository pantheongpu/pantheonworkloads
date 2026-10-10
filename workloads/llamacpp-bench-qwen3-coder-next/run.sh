#!/usr/bin/env bash
# llamacpp-bench-qwen3-coder-next: llama-bench tokens/s with Qwen3-Coder-Next (Q4_K_M GGUF, 45.1 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 46 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3-Coder-Next needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF/resolve/b82fb7382639d97b38fa7672e526c760c2fb358e" "$here/model.sha256" 46
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
