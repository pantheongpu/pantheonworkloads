#!/usr/bin/env bash
# llamacpp-bench-nemotron-3-ultra-550b-a55b: llama-bench tokens/s with NVIDIA-Nemotron-3-Ultra-550B-A55B (UD-Q4_K_M GGUF, 334.6 GiB, 8 x 48 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 335 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: NVIDIA-Nemotron-3-Ultra-550B-A55B needs real GPUs (8 x 48 GiB)"
lc_require_gpus 8 48
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF/resolve/2fb7d5b3f4eae7aedb18b4839b6a6300111e46f6" "$here/model.sha256" 335
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
