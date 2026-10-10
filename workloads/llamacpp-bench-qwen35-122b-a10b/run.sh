#!/usr/bin/env bash
# llamacpp-bench-qwen35-122b-a10b: llama-bench tokens/s with Qwen3.5-122B-A10B (Q4_K_M GGUF, 71.3 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 72 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3.5-122B-A10B needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF/resolve/51eab4d59d53f573fb9206cb3ce613f1d0aa392b" "$here/model.sha256" 72
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
