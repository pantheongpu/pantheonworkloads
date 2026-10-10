#!/usr/bin/env bash
# llamacpp-bench-step-3p7-flash: llama-bench tokens/s with Step-3.7-Flash (IQ4_XS GGUF, 97.8 GiB, 2 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 98 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Step-3.7-Flash needs real GPUs (2 x 80 GiB)"
lc_require_gpus 2 80
lc_setup
lc_fetch_shards "https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF/resolve/0b69336d2fd2adfdef9c66e425f7778196c31482" "$here/model.sha256" 98
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
