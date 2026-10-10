#!/usr/bin/env bash
# llamacpp-bench-devstral-small-2-24b-2512: llama-bench tokens/s with Devstral-Small-2-24B-Instruct-2512 (Q4_K_M GGUF, 13.3 GiB, 1 x 22 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 14 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Devstral-Small-2-24B-Instruct-2512 needs real GPUs (1 x 22 GiB)"
lc_require_gpus 1 22
lc_setup
lc_fetch_shards "https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/resolve/6e458b8add42681bfd023de5eab93637694aaf82" "$here/model.sha256" 14
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
