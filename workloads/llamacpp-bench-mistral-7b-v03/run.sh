#!/usr/bin/env bash
# llamacpp-bench-mistral-7b-v03: llama-bench on a Mistral-7B-v0.3 GGUF. Real GPUs only.
# Default: mradermacher/Mistral-7B-v0.3-GGUF @ 76804246, Q8_0 (sha256 in model.sha256).
# Override with PW_MODEL_FILE=/path/to.gguf or PW_MODEL_URL=<direct link> (+ PW_MODEL_SHA256).
# Other knobs as in llamacpp-bench-smollm2-135m (docs/llamacpp.md).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
lc_setup
sha="${PW_MODEL_SHA256:-$(cat "$here/model.sha256" 2>/dev/null | tr -d '[:space:]')}"
default_url="https://huggingface.co/mradermacher/Mistral-7B-v0.3-GGUF/resolve/768042462c239dab9e96b9ec65a43775665d6959/Mistral-7B-v0.3.Q8_0.gguf"
lc_fetch_model "${PW_MODEL_URL:-$default_url}" "$sha" "Mistral-7B-v0.3.Q8_0.gguf"
. "$here/../../tools/llamacpp/bench.sh"
lc_bench
