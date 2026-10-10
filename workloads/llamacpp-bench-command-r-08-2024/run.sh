#!/usr/bin/env bash
# llamacpp-bench-command-r-08-2024: llama-bench tokens/s with c4ai-command-r-08-2024 (Q4_K_M GGUF, 18.4 GiB, 1 x 24 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 19 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: c4ai-command-r-08-2024 needs real GPUs (1 x 24 GiB)"
lc_require_gpus 1 24
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/c4ai-command-r-08-2024-GGUF/resolve/0390adf30f4e73160774db7d8fcd1f6172a16e65" "$here/model.sha256" 19
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
