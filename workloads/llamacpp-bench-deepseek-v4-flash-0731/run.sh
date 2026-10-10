#!/usr/bin/env bash
# llamacpp-bench-deepseek-v4-flash-0731: llama-bench tokens/s with DeepSeek-V4-Flash-0731 (MXFP4 GGUF, 144.3 GiB, 4 x 44 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 145 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: DeepSeek-V4-Flash-0731 needs real GPUs (4 x 44 GiB)"
lc_require_gpus 4 44
lc_setup
lc_fetch_shards "https://huggingface.co/ggml-org/DeepSeek-V4-Flash-0731-GGUF/resolve/f559fd6005309e5f6bd650342ee8711ff189b3b8" "$here/model.sha256" 145
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
