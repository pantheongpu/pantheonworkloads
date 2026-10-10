#!/usr/bin/env bash
# llamacpp-bench-mistral-large-3-675b-instruct-2512: llama-bench tokens/s with Mistral-Large-3-675B-Instruct-2512 (Q4_K_M GGUF, 379.0 GiB, 8 x 80 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 380 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Mistral-Large-3-675B-Instruct-2512 needs real GPUs (8 x 80 GiB)"
lc_require_gpus 8 80
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Mistral-Large-3-675B-Instruct-2512-GGUF/resolve/922217b859d90a8d94088a1c900fdf03f8fd078e" "$here/model.sha256" 380
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
