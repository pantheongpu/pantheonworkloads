#!/usr/bin/env bash
# llamacpp-bench-deepseek-v3p2: llama-bench tokens/s with DeepSeek-V3.2 (Q4_K_M GGUF, 377.6 GiB, 8 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 378 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: DeepSeek-V3.2 needs real GPUs (8 x 80 GiB)"
lc_require_gpus 8 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/DeepSeek-V3.2-GGUF/resolve/a787696863bafd5c736955ef81cc869a0bf6178a" "$here/model.sha256" 378
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
