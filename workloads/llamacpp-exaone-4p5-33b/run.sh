#!/usr/bin/env bash
# llamacpp-exaone-4p5-33b: llama-completion greedy-decodes 3 tokens with EXAONE-4.5-33B (Q4_K_M GGUF, 18.7 GiB, 1 x 24 GiB GPU(s)).
# VERIFIED on an L40S (stage 2a, Tier B; docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 19 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: EXAONE-4.5-33B needs real GPUs (1 x 24 GiB)"
lc_require_gpus 1 24
lc_setup
lc_fetch_shards "https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-GGUF/resolve/0e969634ef24db05151b435970297a6dee634b7e" "$here/model.sha256" 19

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
