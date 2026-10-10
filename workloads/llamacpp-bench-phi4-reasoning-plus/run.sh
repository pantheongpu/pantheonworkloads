#!/usr/bin/env bash
# llamacpp-bench-phi4-reasoning-plus: llama-bench tokens/s with Phi-4-reasoning-plus (Q4_K_M GGUF, 8.4 GiB, 1 x 15 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 9 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Phi-4-reasoning-plus needs real GPUs (1 x 15 GiB)"
lc_require_gpus 1 15
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF/resolve/80fff8542dc7b88dba725b660beefd80e91e80c9" "$here/model.sha256" 9
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
