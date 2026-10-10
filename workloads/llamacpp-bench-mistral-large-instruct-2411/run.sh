#!/usr/bin/env bash
# llamacpp-bench-mistral-large-instruct-2411: llama-bench tokens/s with Mistral-Large-Instruct-2411 (Q4_K_M GGUF, 68.2 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 69 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Mistral-Large-Instruct-2411 needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/Mistral-Large-Instruct-2411-GGUF/resolve/cc2729f0da879f162f8d2a7e74c7324bab6e51b7" "$here/model.sha256" 69
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
