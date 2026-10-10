#!/usr/bin/env bash
# llamacpp-bench-trinity-large-thinking: llama-bench tokens/s with Trinity-Large-Thinking (Q4_K_M GGUF, 225.2 GiB, 4 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 226 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Trinity-Large-Thinking needs real GPUs (4 x 80 GiB)"
lc_require_gpus 4 80
lc_setup
lc_fetch_shards "https://huggingface.co/arcee-ai/Trinity-Large-Thinking-GGUF/resolve/352ed5f50f0148466e596950c29aecb51bed4cb3" "$here/model.sha256" 226
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
