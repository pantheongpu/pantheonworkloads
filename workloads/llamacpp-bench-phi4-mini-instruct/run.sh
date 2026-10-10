#!/usr/bin/env bash
# llamacpp-bench-phi4-mini-instruct: llama-bench tokens/s with Phi-4-mini-instruct (Q8_0 GGUF, 3.8 GiB, 1 x 10 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 4 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Phi-4-mini-instruct needs real GPUs (1 x 10 GiB)"
lc_require_gpus 1 10
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/78eb92a46fc37e6b524df991ed9aca9bc6aa7b80" "$here/model.sha256" 4
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
