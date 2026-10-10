#!/usr/bin/env bash
# llamacpp-bench-kimi-linear-48b-a3b-instruct: llama-bench tokens/s with Kimi-Linear-48B-A3B-Instruct (Q4_K_M GGUF, 28.0 GiB, 1 x 40 GiB GPU(s)).
# VERIFIED on an L40S (stage 2a, Tier B; docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 28 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Kimi-Linear-48B-A3B-Instruct needs real GPUs (1 x 40 GiB)"
lc_require_gpus 1 40
lc_setup
lc_fetch_shards "https://huggingface.co/bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF/resolve/228dbe476e5a02091624a19068f4c962caa8a1c5" "$here/model.sha256" 28
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
