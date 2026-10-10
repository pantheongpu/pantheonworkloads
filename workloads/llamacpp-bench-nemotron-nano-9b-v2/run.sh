#!/usr/bin/env bash
# llamacpp-bench-nemotron-nano-9b-v2: llama-bench tokens/s with NVIDIA-Nemotron-Nano-9B-v2 (Q8_0 GGUF, 8.8 GiB, 1 x 15 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 9 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: NVIDIA-Nemotron-Nano-9B-v2 needs real GPUs (1 x 15 GiB)"
lc_require_gpus 1 15
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF/resolve/b2daccafb3d123ea94653720a0db1f5db9c698ff" "$here/model.sha256" 9
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
