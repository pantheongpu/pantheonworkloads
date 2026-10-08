#!/usr/bin/env bash
# llamacpp-bench-deepseek-r1-distill-qwen-1p5b: llama-bench prompt-processing and token-generation
# tokens/s. Contract: docs/workload-contract.md. Meant for real GPUs; run with
#   bin/pw run llamacpp-bench-deepseek-r1-distill-qwen-1p5b --target gpu --bench --repeat 5 [--device NAME]
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
default_url="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/9cc28b17e86fa2415fcb070f8ee5ec27c965aa61/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
sha="${PW_MODEL_SHA256:-$(cat "$here/model.sha256" 2>/dev/null | tr -d '[:space:]')}"
lc_fetch_model "${PW_MODEL_URL:-$default_url}" "$sha" "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
