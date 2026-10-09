#!/usr/bin/env bash
# llamacpp-granite40-h-small: llama.cpp's llama-completion greedy-decodes 3 tokens after
# a fixed prompt from a GGUF. Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_URL     direct link to the .gguf (default: see below)
#   PW_MODEL_SHA256  expected sha256 (default: the one in model.sha256 next to this file, if any)
#   PW_MODEL_FILE    a local .gguf instead of downloading
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
lc_setup
default_url="https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/6522095069a86fc3186b8632850e4373dd850cac/granite-4.0-h-small-Q4_K_M.gguf"
sha="${PW_MODEL_SHA256:-$(cat "$here/model.sha256" 2>/dev/null | tr -d '[:space:]')}"
lc_fetch_model "${PW_MODEL_URL:-$default_url}" "$sha" "granite-4.0-h-small-Q4_K_M.gguf"

# Greedy (temp 0), fixed seed, no chat template (-no-cnv), prompt not echoed: stdout is the 3 new tokens.
env "${LC_ENV[@]}" "$LC_BIN/llama-completion" -m "$LC_MODEL" -p "The capital of France is" -n 3 \
  --temp 0 -s 1 -no-cnv --no-display-prompt -ngl "$LC_NGL" -t "${PW_THREADS:-2}" -c 256 \
  >"$PW_OUT/completion.out" 2>"$PW_OUT/completion.err" \
  || { echo "llama-completion failed" >&2; tail -20 "$PW_OUT/completion.err" >&2; exit 1; }
PW_OUTFILE="$PW_OUT/completion.out" PW_DETAIL="llama.cpp $LLAMACPP_TAG $LC_BACKEND, $(basename "$LC_MODEL")" python3 - <<'PY'
import json, os
text = open(os.environ["PW_OUTFILE"], encoding="utf-8", errors="replace").read()
print(json.dumps({"output": text, "detail": os.environ["PW_DETAIL"], "metrics": {}}))
PY
