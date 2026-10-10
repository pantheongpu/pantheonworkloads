#!/usr/bin/env bash
# llamacpp-bench-mistral-small-4-119b-2603: llama-bench tokens/s with Mistral-Small-4-119B-2603 (Q4_K_M GGUF, 67.6 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 68 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Mistral-Small-4-119B-2603 needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF/resolve/f569d04efea9546de10ec040819908fa4e8f1c92" "$here/model.sha256" 68
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
