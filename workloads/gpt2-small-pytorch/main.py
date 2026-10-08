"""gpt2-small-pytorch: GPT-2 small (124M), five greedy tokens after a fixed prompt, float32."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import common  # noqa: E402

MODEL = "openai-community/gpt2"
REVISION = os.environ.get("PW_HF_REVISION") or "607a30d783dfa663caf39e06633721c8d4cfcd7e"   # pinned (manifest); PW_HF_REVISION overrides
PROMPT = "The capital of France is"
NEW = 5

common.setup()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = common.hf_load(lambda m, r, c: AutoTokenizer.from_pretrained(m, revision=r, cache_dir=c), MODEL, REVISION)
# eager attention: plain matmul/softmax kernels; the micro-suite and the BERT workload cover SDPA.
model = common.hf_load(lambda m, r, c: AutoModelForCausalLM.from_pretrained(
    m, revision=r, cache_dir=c, torch_dtype=torch.float32, attn_implementation="eager"), MODEL, REVISION)
common.verify_pinned(MODEL, REVISION, "model.safetensors")
model = model.eval().to(common.DEVICE)

ids = tok(PROMPT, return_tensors="pt").input_ids.to(common.DEVICE)
prompt_ids = ids[0].tolist()
first_logits, gaps = None, []
with torch.no_grad():
    past = None
    cur = ids
    for step in range(NEW):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float().cpu()
        top = torch.topk(logits, 2)
        gaps.append(common.margin_ok(top.values, f"step {step}"))
        if first_logits is None:
            first_logits = float(top.values[0])
        nxt = top.indices[0].view(1, 1).to(common.DEVICE)
        ids = torch.cat([ids, nxt], 1)
        cur = nxt

new_ids = ids[0].tolist()[len(prompt_ids):]
output = {
    "token_ids": " ".join(map(str, new_ids)),            # exact
    "text": tok.decode(new_ids),                         # exact
    "first_step_top1_logit": first_logits,               # tolerance (manifest)
}
metrics = {}
if common.REAL and common.DEVICE != "cpu" and os.environ.get("PW_NO_BENCH") != "1":
    with torch.no_grad():
        n = 64
        def gen():
            torch.manual_seed(0)
            model.generate(ids[:, :len(prompt_ids)], max_new_tokens=n, min_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        sec = common.bench(gen, iters=1, rounds=3, warmup=1)
        metrics["decode_tokens_per_s"] = round(n / sec, 1)
lic = common.hub_licence(MODEL)
common.finish(output, f"{common.device_name()}, torch {torch.__version__}, min top1-top2 gap {min(gaps):.3g}, "
              f"card licence now: {lic}", metrics)
