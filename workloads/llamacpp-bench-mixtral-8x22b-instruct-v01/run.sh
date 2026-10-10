#!/usr/bin/env bash
# llamacpp-bench-mixtral-8x22b-instruct-v01: llama-bench tokens/s with Mixtral-8x22B-Instruct-v0.1 (Q4_K_M GGUF, 79.7 GiB, 2 x 48 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 80 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Mixtral-8x22B-Instruct-v0.1 needs real GPUs (2 x 48 GiB)"
lc_require_gpus 2 48
lc_setup
lc_fetch_shards "https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF/resolve/9f2f6c5ec37f9bce5f5f3a7ff07b11d573443e62" "$here/model.sha256" 80
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
