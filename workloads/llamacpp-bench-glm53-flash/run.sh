#!/usr/bin/env bash
# llamacpp-bench-glm53-flash: llama-bench tokens/s with GLM-5.3-Flash (UD-Q4_K_XL GGUF, 186.0 GiB, 4 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 186 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: GLM-5.3-Flash needs real GPUs (4 x 80 GiB)"
lc_require_gpus 4 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/a38483c8cd5df544f53d70fb281afe97369d5ab6" "$here/model.sha256" 186
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
