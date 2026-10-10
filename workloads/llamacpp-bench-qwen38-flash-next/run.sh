#!/usr/bin/env bash
# llamacpp-bench-qwen38-flash-next: llama-bench tokens/s with Qwen3.8-Flash-Next (Q4_K_M GGUF, 111.4 GiB, 2 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 112 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3.8-Flash-Next needs real GPUs (2 x 80 GiB)"
lc_require_gpus 2 80
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF/resolve/928589fdb66c6ff07f22ac561e3fbce76553548f" "$here/model.sha256" 112
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
