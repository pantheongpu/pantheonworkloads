#!/usr/bin/env bash
# llamacpp-bench-ministral-3-14b-reasoning-2512: llama-bench tokens/s with Ministral-3-14B-Reasoning-2512 (Q8_0 GGUF, 13.4 GiB, 1 x 22 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 14 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Ministral-3-14B-Reasoning-2512 needs real GPUs (1 x 22 GiB)"
lc_require_gpus 1 22
lc_setup
lc_fetch_shards "https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512-GGUF/resolve/fe3b038f30334729263d860d5dadbaa34e0f2a18" "$here/model.sha256" 14
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
