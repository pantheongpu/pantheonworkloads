#!/usr/bin/env bash
# llamacpp-bench-qwen3-coder-480b-a35b-instruct: llama-bench tokens/s with Qwen3-Coder-480B-A35B-Instruct (Q4_K_M GGUF, 270.1 GiB, 4 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 271 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3-Coder-480B-A35B-Instruct needs real GPUs (4 x 80 GiB)"
lc_require_gpus 4 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/b86deeefd82f1a3374c5536dfc1dd0ce27ac092d" "$here/model.sha256" 271
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
