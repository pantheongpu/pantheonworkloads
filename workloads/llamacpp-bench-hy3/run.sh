#!/usr/bin/env bash
# llamacpp-bench-hy3: llama-bench tokens/s with Tencent Hy3 (Q4_K_M GGUF, 169.6 GiB, 4 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 170 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Tencent Hy3 needs real GPUs (4 x 80 GiB)"
lc_require_gpus 4 80
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/Hy3-GGUF/resolve/3ee5a3c4d76226edd9cdaec017969d24b589b64e" "$here/model.sha256" 170
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
