#!/usr/bin/env bash
# llamacpp-bench-glm52: llama-bench tokens/s with GLM-5.2 (UD-Q4_K_M GGUF, 433.8 GiB, 8 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 434 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: GLM-5.2 needs real GPUs (8 x 80 GiB)"
lc_require_gpus 8 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/GLM-5.2-GGUF/resolve/abc55e72527792c6e77069c99b4cb7de16fa9f23" "$here/model.sha256" 434
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
