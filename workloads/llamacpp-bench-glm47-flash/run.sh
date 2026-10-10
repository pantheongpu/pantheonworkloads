#!/usr/bin/env bash
# llamacpp-bench-glm47-flash: llama-bench tokens/s with GLM-4.7-Flash (Q4_K_M GGUF, 17.1 GiB, 1 x 22 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 18 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: GLM-4.7-Flash needs real GPUs (1 x 22 GiB)"
lc_require_gpus 1 22
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/0d32489ecb9db6d2a4fc93bd27ef01519f95474d" "$here/model.sha256" 18
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
