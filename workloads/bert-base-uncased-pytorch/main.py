"""bert-base-uncased-pytorch: BERT-base encoder forward pass, masked-word prediction, float32."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import common  # noqa: E402

MODEL = "google-bert/bert-base-uncased"
REVISION = os.environ.get("PW_HF_REVISION") or None
TEXT = "The capital of France is [MASK]."

common.setup()
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

tok = common.hf_load(lambda m, r, c: AutoTokenizer.from_pretrained(m, revision=r, cache_dir=c), MODEL, REVISION)
model = common.hf_load(lambda m, r, c: AutoModelForMaskedLM.from_pretrained(
    m, revision=r, cache_dir=c, torch_dtype=torch.float32, attn_implementation="sdpa"), MODEL, REVISION)
model = model.eval().to(common.DEVICE)

enc = tok(TEXT, return_tensors="pt").to(common.DEVICE)
pos = int((enc.input_ids[0] == tok.mask_token_id).nonzero()[0])
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
logits = out.logits[0, pos].float().cpu()
top = torch.topk(logits, 6)   # the sixth only proves the fifth is not tied
gap = common.margin_ok(top.values, "mask position", floor=1e-3)
hidden = out.hidden_states[-1][0].float().cpu()
output = {
    "mask_top5_ids": " ".join(map(str, top.indices.tolist()[:5])),            # exact
    "mask_top5_tokens": " ".join(tok.convert_ids_to_tokens(top.indices.tolist()[:5])),   # exact
    "mask_top1_logit": float(top.values[0]),                                # tolerance
    "last_hidden": [float(hidden.abs().mean()), float(hidden.pow(2).mean().sqrt()), float(hidden[pos, 0])],
}
metrics = {}
if common.REAL and common.DEVICE != "cpu" and os.environ.get("PW_NO_BENCH") != "1":
    batch = {k: torch.randint(1000, 20000, (32, 128), device=common.DEVICE) if k == "input_ids"
             else torch.ones(32, 128, dtype=torch.long, device=common.DEVICE)
             for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        sec = common.bench(lambda: model.bert(**batch), iters=5, rounds=3)
    metrics["forward_sequences_per_s_b32_s128"] = round(32 / sec, 1)
lic = common.hub_licence(MODEL)
common.finish(output, f"{common.device_name()}, torch {torch.__version__}, top1-top2 gap {gap:.3g}, "
              f"card licence now: {lic}", metrics)
