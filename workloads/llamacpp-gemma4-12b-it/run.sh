#!/usr/bin/env bash
# llamacpp-gemma4-12b-it: llama-completion greedy-decodes 3 tokens with gemma-4-12B-it (Q4_0 GGUF, 6.5 GiB, 1 x 15 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 7 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: gemma-4-12B-it needs real GPUs (1 x 15 GiB)"
lc_require_gpus 1 15
lc_setup
lc_fetch_shards "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/29d097773436b69ff9feafd636ab4cf873786537" "$here/model.sha256" 7

# Greedy (temp 0), fixed seed, prompt not echoed: stdout is the new tokens (at most 3). This checkpoint answers only inside its own chat
# format, so the prompt is the model's chat template (Gemma 4: <|turn>user ... <turn|> <|turn>model, thinking off = the empty thought
# block the template adds) written out, sent as plain text (-no-cnv: no second template is applied). See the manifest notes.
prompt=$'<|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
env "${LC_ENV[@]}" "$LC_BIN/llama-completion" -m "$LC_MODEL" -p "$prompt" -n 3 \
  --temp 0 -s 1 -no-cnv --no-display-prompt -ngl "$LC_NGL" -t "${PW_THREADS:-2}" -c 256 \
  >"$PW_OUT/completion.out" 2>"$PW_OUT/completion.err" \
  || { echo "llama-completion failed" >&2; tail -20 "$PW_OUT/completion.err" >&2; exit 1; }
PW_OUTFILE="$PW_OUT/completion.out" PW_DETAIL="llama.cpp $LLAMACPP_TAG $LC_BACKEND, $(basename "$LC_MODEL")" python3 - <<'PY'
import json, os
text = open(os.environ["PW_OUTFILE"], encoding="utf-8", errors="replace").read()
print(json.dumps({"output": text, "detail": os.environ["PW_DETAIL"], "metrics": {}}))
PY
