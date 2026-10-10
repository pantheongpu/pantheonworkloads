#!/usr/bin/env bash
# llamacpp-bench-nemotron-3-super-120b-a12b: llama-bench tokens/s with NVIDIA-Nemotron-3-Super-120B-A12B (Q4_K_M GGUF, 80.1 GiB, 2 x 48 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 81 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: NVIDIA-Nemotron-3-Super-120B-A12B needs real GPUs (2 x 48 GiB)"
lc_require_gpus 2 48
lc_setup
lc_fetch_shards "https://huggingface.co/lmstudio-community/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/resolve/16410500b9bdc62081c34cc2dff52471e001f970" "$here/model.sha256" 81
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
