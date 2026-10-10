#!/usr/bin/env bash
# llamacpp-bench-falcon3-10b-instruct: llama-bench tokens/s with Falcon3-10B-Instruct (Q8_0 GGUF, 10.2 GiB, 1 x 15 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 11 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Falcon3-10B-Instruct needs real GPUs (1 x 15 GiB)"
lc_require_gpus 1 15
lc_setup
lc_fetch_shards "https://huggingface.co/tiiuae/Falcon3-10B-Instruct-GGUF/resolve/0072d525c72df8cbb1af9561672da481bd9e5595" "$here/model.sha256" 11
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
