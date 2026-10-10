#!/usr/bin/env bash
# llamacpp-bench-falcon-h1-34b-instruct: llama-bench tokens/s with Falcon-H1-34B-Instruct (Q4_K_M GGUF, 18.9 GiB, 1 x 24 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 19 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Falcon-H1-34B-Instruct needs real GPUs (1 x 24 GiB)"
lc_require_gpus 1 24
lc_setup
lc_fetch_shards "https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct-GGUF/resolve/ceb29e9c241944eb64c44325972c4e9470271d92" "$here/model.sha256" 19
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
