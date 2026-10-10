#!/usr/bin/env bash
# llamacpp-bench-codestral-22b-v01: llama-bench tokens/s with Codestral-22B-v0.1 (Q4_K_M GGUF, 12.4 GiB, 1 x 22 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 13 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Codestral-22B-v0.1 needs real GPUs (1 x 22 GiB)"
lc_require_gpus 1 22
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF/resolve/0e6abe14d6aeaf2c99d5dc9973205e8e38692d90" "$here/model.sha256" 13
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
