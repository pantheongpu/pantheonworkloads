#!/usr/bin/env bash
# llamacpp-bench-phi4-14b: llama-bench prompt-processing and token-generation
# tokens/s. Contract: docs/workload-contract.md. Meant for real GPUs; run with
#   bin/pw run llamacpp-bench-phi4-14b --target gpu --bench --repeat 5 [--device NAME]
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_URL / PW_MODEL_SHA256 / PW_MODEL_FILE   as in the functional workload
#   PW_BENCH_PP      prompt tokens (default 512)       PW_BENCH_TG   generated tokens (default 128)
#   PW_BENCH_REPS    llama-bench -r (default 3)        PW_NGL        layers offloaded (default 99)
#   PW_BENCH_EXTRA   extra llama-bench arguments, e.g. "-fa 1"
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
lc_setup
default_url="https://huggingface.co/microsoft/phi-4-gguf/resolve/6edc2ef6664b739a8e11e62f2672ff6afe0c15ac/phi-4-Q4_K.gguf"
sha="${PW_MODEL_SHA256:-$(cat "$here/model.sha256" 2>/dev/null | tr -d '[:space:]')}"
lc_fetch_model "${PW_MODEL_URL:-$default_url}" "$sha" "phi-4-Q4_K.gguf"
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
