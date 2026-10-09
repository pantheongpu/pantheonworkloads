#!/usr/bin/env bash
# llamacpp-qwen3-embedding-4b: llama.cpp's llama-embedding embeds four fixed texts with Qwen3-Embedding-4B
# (last-token pooling, L2-normalised) and the workload reports their cosine similarities and the ranking
# of the three passages for the query. Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
# llama-embedding is an example program, not a tool: tools/llamacpp/build.sh needs
# LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS=llama-embedding to build it (set below before the first build).
#
#   PW_MODEL_URL     direct link to the .gguf (default: see below)
#   PW_MODEL_SHA256  expected sha256 (default: the one in model.sha256 next to this file, if any)
#   PW_MODEL_FILE    a local .gguf instead of downloading
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
export LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS="${LLAMACPP_EXTRA_TARGETS:-llama-embedding}"
lc_setup
[[ -x "$LC_BIN/llama-embedding" ]] || lc_skip "no llama-embedding in $LC_BIN (build with LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS=llama-embedding, see tools/llamacpp/build.sh)"
default_url="https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/f4602530db1d980e16da9d7d3a70294cf5c190be/Qwen3-Embedding-4B-Q8_0.gguf"
sha="${PW_MODEL_SHA256:-$(cat "$here/model.sha256" 2>/dev/null | tr -d '[:space:]')}"
lc_fetch_model "${PW_MODEL_URL:-$default_url}" "$sha" "Qwen3-Embedding-4B-Q8_0.gguf"

# Text 0 is a query in Qwen3-Embedding's instruction format; 1 to 3 are passages. "<|endoftext|>" is the end-of-sequence
# token the model's last-token pooling reads (the GGUF does not add it itself). Texts are joined with a separator that
# is not a newline (the query has one).
sep='<#sep#>'
eos='<|endoftext|>'
q=$'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of France?'
texts="${q}${eos}${sep}Paris is the capital and most populous city of France.${eos}${sep}The mitochondrion is the organelle that produces most of the cell's ATP.${eos}${sep}Berlin is the capital of Germany.${eos}"
env "${LC_ENV[@]}" "$LC_BIN/llama-embedding" -m "$LC_MODEL" -p "$texts" --embd-separator "$sep" --pooling last \
  --embd-normalize 2 --embd-output-format array -ngl "$LC_NGL" -t "${PW_THREADS:-2}" -c 512 -b 512 -ub 512 \
  >"$PW_OUT/embedding.out" 2>"$PW_OUT/embedding.err" \
  || { echo "llama-embedding failed" >&2; tail -20 "$PW_OUT/embedding.err" >&2; exit 1; }
PW_OUTFILE="$PW_OUT/embedding.out" PW_DETAIL="llama.cpp $LLAMACPP_TAG $LC_BACKEND, $(basename "$LC_MODEL")" python3 -I - <<'PY'
import json, math, os
raw = open(os.environ["PW_OUTFILE"], encoding="utf-8", errors="replace").read()
embs = json.loads(raw[raw.index("["):])
assert len(embs) == 4, len(embs)
dim = len(embs[0])
assert all(len(e) == dim for e in embs)
norms = [math.sqrt(sum(x * x for x in e)) for e in embs]
assert all(abs(n - 1.0) < 1e-3 for n in norms), norms          # --embd-normalize 2
def cos(a, b): return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
sims = {"%d-%d" % p: round(cos(embs[p[0]], embs[p[1]]), 4) for p in pairs}
ranking = sorted((1, 2, 3), key=lambda j: -sims["0-%d" % j])
out = {"dim": dim, "cosine": sims, "query_ranking": " ".join(map(str, ranking)),
       "finite": all(math.isfinite(x) for e in embs for x in e)}
print(json.dumps({"output": out, "detail": os.environ["PW_DETAIL"] + ", dim %d" % dim, "metrics": {}}))
PY
