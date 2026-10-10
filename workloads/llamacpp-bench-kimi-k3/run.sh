#!/usr/bin/env bash
# llamacpp-bench-kimi-k3: llama-bench tokens/s with Kimi-K3 (UD-Q4_K_XL GGUF, 1405.1 GiB, 8 x 192 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 1406 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Kimi-K3 needs real GPUs (8 x 192 GiB)"
lc_require_gpus 8 192
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Kimi-K3-GGUF/resolve/a0836360ce58dfec088d966a97f2ddc8a606279b" "$here/model.sha256" 1406
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
