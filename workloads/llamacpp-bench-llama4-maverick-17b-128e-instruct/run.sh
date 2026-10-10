#!/usr/bin/env bash
# llamacpp-bench-llama4-maverick-17b-128e-instruct: llama-bench tokens/s with Llama-4-Maverick-17B-128E-Instruct (Q4_K_M GGUF, 226.1 GiB, 4 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 227 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Llama-4-Maverick-17B-128E-Instruct needs real GPUs (4 x 80 GiB)"
lc_require_gpus 4 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF/resolve/41032e5471dd6ea5b349062d978626215d6c5bba" "$here/model.sha256" 227
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
