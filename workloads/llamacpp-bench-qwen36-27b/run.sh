#!/usr/bin/env bash
# llamacpp-bench-qwen36-27b: llama-bench tokens/s with Qwen3.6-27B (Q4_K_M GGUF, 17.8 GiB, 1 x 24 GiB GPU(s)).
# VERIFIED on an L40S (stage 2a, Tier B; docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 18 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3.6-27B needs real GPUs (1 x 24 GiB)"
lc_require_gpus 1 24
lc_setup
lc_fetch_shards "https://huggingface.co/ggml-org/Qwen3.6-27B-GGUF/resolve/8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8" "$here/model.sha256" 18
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
