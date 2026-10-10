#!/usr/bin/env bash
# llamacpp-bench-qwen38-2p4t-a95b: llama-bench tokens/s with Qwen3.8-2.4T-A95B (UD-IQ4_XS GGUF, 1220.8 GiB, 8 x 180 GiB GPU(s)).
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 1221 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3.8-2.4T-A95B needs real GPUs (8 x 180 GiB)"
lc_require_gpus 8 180
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/resolve/567d3e6ac26c5474b18311e619c04350fb9a5556" "$here/model.sha256" 1221
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
