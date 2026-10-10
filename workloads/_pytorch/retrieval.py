"""Shared body of the catalog's retrieval workloads (embedding and reranking models), imported by their main.py.

The inputs are the four texts of llamacpp-qwen3-embedding-4b: a query ("What is the capital of France?") and three
passages (Paris, the mitochondrion, Berlin). Heads:
  embed-cls          BGE-M3 style dense embedding: CLS token of the last layer, L2-normalised; output = cosine
                     similarities and the passage ranking
  seqcls             a cross-encoder reranker (BAAI/bge-reranker-*): one logit per (query, passage) pair; output =
                     the logits and the ranking
  qwen3-reranker     Qwen3-Reranker: a causal LM scored by P("yes") at the last position of a fixed judging prompt
                     (the model card's recipe); output = the scores and the ranking
Bench mode (PW_RETRIEVAL_MODE=bench): forward passes per second over the four texts / three pairs. Never run on a GPU yet.
"""
import os
import sys
import time

import torch

import catalog_common as cc
import common

QUERY = "What is the capital of France?"
PASSAGES = ["Paris is the capital and most populous city of France.",
            "The mitochondrion is the organelle that produces most of the cell's ATP.",
            "Berlin is the capital of Germany."]
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def run(model_id, revision, head, label=None, dtype="bfloat16", margin=1e-3):
    mode = os.environ.get("PW_RETRIEVAL_MODE", "functional")
    if common.DEVICE == "cpu":
        common.skip("this retrieval model is run on a GPU only")
    common.setup()
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    path = cc.fetch_snapshot(model_id, revision)
    td = getattr(torch, dtype)
    dmap = os.environ.get("PW_DEVICE_MAP") or "auto"
    tok = AutoTokenizer.from_pretrained(path, padding_side="left" if head == "qwen3-reranker" else "right")

    def fwd_embed():
        model = AutoModel.from_pretrained(path, dtype=td, device_map=dmap).eval()
        texts = [QUERY] + PASSAGES

        def go():
            enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(common.DEVICE)
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, 0]
            e = torch.nn.functional.normalize(h.float(), dim=-1)
            return [float(e[0] @ e[i]) for i in range(1, 4)]
        return go

    def fwd_seqcls():
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=td, device_map=dmap).eval()

        def go():
            enc = tok([QUERY] * 3, PASSAGES, padding=True, truncation=True, max_length=512, return_tensors="pt").to(common.DEVICE)
            with torch.no_grad():
                return [float(x) for x in model(**enc).logits.float().view(-1)]
        return go

    def fwd_qwen():
        model = AutoModelForCausalLM.from_pretrained(path, dtype=td, device_map=dmap).eval()
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                  "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        texts = [f"{prefix}<Instruct>: {INSTRUCTION}\n<Query>: {QUERY}\n<Document>: {p}{suffix}" for p in PASSAGES]

        def go():
            enc = tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(common.DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]
            two = torch.stack([logits[:, no], logits[:, yes]], dim=1).float()
            return [float(x) for x in torch.softmax(two, dim=1)[:, 1]]
        return go

    t0 = time.perf_counter()
    go = {"embed-cls": fwd_embed, "seqcls": fwd_seqcls, "qwen3-reranker": fwd_qwen}[head]()
    load_s = time.perf_counter() - t0
    scores = go()
    if not all(map(lambda x: x == x and abs(x) != float("inf"), scores)):
        sys.exit(f"non-finite scores {scores}")
    order = sorted(range(3), key=lambda i: -scores[i])
    ranked = [scores[i] for i in order]
    gap = min(a - b for a, b in zip(ranked[:-1], ranked[1:]))
    if gap < margin:
        sys.exit(f"two passages score within {gap:g} (< {margin:g}): too close to compare across backends")
    if order[0] != 0:
        sys.exit(f"the Paris passage is not ranked first for the Paris query: scores {scores}")
    out = {"scores": [round(s, 3) for s in scores], "ranking": " ".join(str(i + 1) for i in order)}
    metrics = {}
    if mode == "bench":
        for _ in range(3):
            go()
        common.sync()
        t = time.perf_counter()
        n = 20
        for _ in range(n):
            go()
        common.sync()
        metrics = {"forward_passes_per_s": round(n / (time.perf_counter() - t), 2),
                   "peak_gpu_memory_gb": round(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9, 2)}
        out = f"{label or model_id} bench ok"
    common.finish(out, f"{common.device_name()}, torch {torch.__version__}, {dtype}, head {head}, load {load_s:.0f} s, scores {scores}, min gap {gap:.3g}", metrics)
