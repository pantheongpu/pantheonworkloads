#!/usr/bin/env bash
# llamacpp-bench-llama31-405b-instruct: llama-bench tokens/s with Llama-3.1-405B-Instruct (Q4_K_M GGUF, 228.8 GiB, 4 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 229 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Llama-3.1-405B-Instruct needs real GPUs (4 x 80 GiB)"
lc_require_gpus 4 80
lc_setup
lc_fetch_shards "https://huggingface.co/bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/ad731614180e6fd90426ce87b5ef44f64c897aad" "$here/model.sha256" 229
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
