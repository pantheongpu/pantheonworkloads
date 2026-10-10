#!/usr/bin/env bash
# llamacpp-bench-command-a-03-2025: llama-bench tokens/s with c4ai-command-a-03-2025 (Q4_K_M GGUF, 62.5 GiB, 1 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 63 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: c4ai-command-a-03-2025 needs real GPUs (1 x 80 GiB)"
lc_require_gpus 1 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/c4ai-command-a-03-2025-GGUF/resolve/5ca94ecdf2f60896faaf110a93902bcb1eaefd22" "$here/model.sha256" 63
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
