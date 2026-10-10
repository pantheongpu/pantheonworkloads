#!/usr/bin/env bash
# llamacpp-bench-command-a-plus-05-2026: llama-bench tokens/s with command-a-plus-05-2026 (Q4_K_M GGUF, 125.8 GiB, 2 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 126 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: command-a-plus-05-2026 needs real GPUs (2 x 80 GiB)"
lc_require_gpus 2 80
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/command-a-plus-05-2026-GGUF/resolve/a26ba921242484b5e1a8ac8519169fe7421c3ab3" "$here/model.sha256" 126
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
