#!/usr/bin/env bash
# llamacpp-qwen3-embedding-8b: llama-embedding on four fixed texts with Qwen3-Embedding-8B (Q8_0 GGUF, 7.5 GiB, 1 x 15 GiB GPU(s)).
# Run on an A10G (stage 2a, docs/model-registry.md). Contract: docs/workload-contract.md.
# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.
#
#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)
#   PW_CACHE              where models and builds are kept; the weights need about 8 GiB there
#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: Qwen3-Embedding-8B needs real GPUs (1 x 15 GiB)"
lc_require_gpus 1 15
export LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS="${LLAMACPP_EXTRA_TARGETS:-llama-embedding}"
lc_setup
[[ -x "$LC_BIN/llama-embedding" ]] || lc_skip "no llama-embedding in $LC_BIN (build with LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS=llama-embedding, see tools/llamacpp/build.sh)"
lc_fetch_shards "https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/69d0e58a13e463cd99a9b83e3f5fee7c10265fab" "$here/model.sha256" 8

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
