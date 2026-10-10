#!/usr/bin/env bash
# llamacpp-bench-nemotron-3-nano-30b-a3b: llama-bench tokens/s with NVIDIA-Nemotron-3-Nano-30B-A3B (Q4_K_M GGUF, 20.9 GiB, 1 x 40 GiB GPU(s)).
# VERIFIED on an L40S (stage 2a, Tier B; docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 21 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: NVIDIA-Nemotron-3-Nano-30B-A3B needs real GPUs (1 x 40 GiB)"
lc_require_gpus 1 40
lc_setup
lc_fetch_shards "https://huggingface.co/ggml-org/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF/resolve/f9d9b441a049bf27473c3cdd9cec220f5657b862" "$here/model.sha256" 21
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
