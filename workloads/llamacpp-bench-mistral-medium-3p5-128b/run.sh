#!/usr/bin/env bash
# llamacpp-bench-mistral-medium-3p5-128b: llama-bench tokens/s with Mistral-Medium-3.5-128B (Q4_K_M GGUF, 69.8 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 70 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Mistral-Medium-3.5-128B needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Mistral-Medium-3.5-128B-GGUF/resolve/c8f5b1477e1b22cd2d819157d450f001f7047298" "$here/model.sha256" 70
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
