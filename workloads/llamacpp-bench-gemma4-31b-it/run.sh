#!/usr/bin/env bash
# llamacpp-bench-gemma4-31b-it: llama-bench tokens/s with gemma-4-31B-it (Q4_0 GGUF, 16.4 GiB, 1 x 22 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 17 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: gemma-4-31B-it needs real GPUs (1 x 22 GiB)"
lc_require_gpus 1 22
lc_setup
lc_fetch_shards "https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf/resolve/59dde24573e7e61570dba08b18a2e1fe246955ed" "$here/model.sha256" 17
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
