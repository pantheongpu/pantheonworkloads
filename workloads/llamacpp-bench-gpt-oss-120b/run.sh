#!/usr/bin/env bash
# llamacpp-bench-gpt-oss-120b: llama-bench tokens/s with gpt-oss-120b (MXFP4 GGUF, 59.0 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 60 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: gpt-oss-120b needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/238abdd290bb874b90a5da1b4549881b7d05c091" "$here/model.sha256" 60
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
