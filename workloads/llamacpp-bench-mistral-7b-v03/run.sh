#!/usr/bin/env bash
# llamacpp-bench-mistral-7b-v03: llama-bench on a Mistral-7B-v0.3 GGUF. Real GPUs only.
# There is no default download: this repository does not name a third-party GGUF
# conversion it has not checked. Give one of
#   PW_MODEL_FILE=/path/to/mistral-7b-v0.3-q4_k_m.gguf
#   PW_MODEL_URL=<direct .gguf link>   (+ PW_MODEL_SHA256 once you have hashed it)
# Other knobs as in llamacpp-bench-smollm2-135m (docs/llamacpp.md).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
lc_setup
sha="${PW_MODEL_SHA256:-$(cat "$here/model.sha256" 2>/dev/null | tr -d '[:space:]')}"
lc_fetch_model "${PW_MODEL_URL:-}" "$sha" "${PW_MODEL_NAME:-mistral-7b-v0.3.gguf}"
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
