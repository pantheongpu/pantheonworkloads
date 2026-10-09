"""Shared body of the vision-language-model workloads (imported by their main.py, beside common.py and pinned.py).

One fixed public-domain image and one fixed prompt go in, greedy-decoded text comes out. Functional mode (default):
the text and the generated token ids are compared exactly, and the run fails when any decoding step's top-1 and top-2
logits are within `margin` (fp16 logits carry noise of about 1e-2, bf16 of about 1e-1), so a reference is never a near
tie that another card could flip. Bench mode (PW_VLM_MODE=bench, set by the -bench twins): decode tokens per second at
batch 1 (forced 128 new tokens, prefill timed separately and subtracted), prefill images per second (image + prompt
through the vision tower and the language model, one token out) and peak GPU memory. The model load is excluded.
"""
import os
import statistics
import sys
import time

import torch

import common
import pinned

# scikit-image's sample photo of Eileen Collins (NASA, public domain per scikit-image's data docstring), 512x512.
IMAGE_URL = ("https://raw.githubusercontent.com/scikit-image/scikit-image/441fe68b95a86d4ae2a351311a0c39a4232b6521/"
             "skimage/data/astronaut.png")
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"
PROMPT = "Describe this image in one sentence."   # PW_VLM_PROMPT overrides it (for choosing a prompt; a workload pins its own)
BENCH_NEW = 128
BENCH_REPS = 5


def run(model_id, revision, dtype="float16", new=32, margin=0.1, label=None, prompt=PROMPT):
    mode = os.environ.get("PW_VLM_MODE", "functional")
    margin = float(os.environ.get("PW_VLM_MARGIN", margin))
    dtype = os.environ.get("PW_VLM_DTYPE", dtype)
    prompt_text = os.environ.get("PW_VLM_PROMPT", prompt)
    new = int(os.environ.get("PW_VLM_NEW", new))
    if common.DEVICE == "cpu":
        common.skip("this vision-language model is run on a GPU only")
    common.setup()
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rev = os.environ.get("PW_HF_REVISION") or revision
    cache_root = os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads"))
    path = common.hf_load(lambda m, r, c: snapshot_download(
        m, revision=r, cache_dir=c, allow_patterns=["*.json", "*.txt", "*.safetensors"]), model_id, rev)
    if not os.environ.get("PW_HF_REVISION"):
        for name, want in pinned.read_sums(os.path.join(os.environ["PW_WORKLOAD_DIR"], "model.sha256")).items():
            if name.endswith(".safetensors"):
                got = pinned.sha256_file(os.path.join(path, name))
                if got != want:
                    sys.exit(f"{name} has sha256 {got}, the pin is {want}")
    img_path = common.hf_load(lambda m, r, c: pinned.fetch_asset(
        IMAGE_URL, IMAGE_SHA256, os.path.join(cache_root, "assets", "astronaut.png")), "astronaut.png", None)
    image = Image.open(img_path).convert("RGB")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=getattr(torch, dtype), device_map=common.DEVICE).eval()
    load_s = time.perf_counter() - t0

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(common.DEVICE)
    n_prompt = inputs["input_ids"].shape[1]
    inputs = {k: (v.to(getattr(torch, dtype)) if v.is_floating_point() else v) for k, v in inputs.items()}

    def generate(new_tokens, forced=False, scores=False):
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=new_tokens, min_new_tokens=new_tokens if forced else None,
                                  do_sample=False, num_beams=1, repetition_penalty=1.0, temperature=None, top_p=None,
                                  top_k=None, output_scores=scores, return_dict_in_generate=scores)

    res = generate(new, scores=True)
    new_ids = res.sequences[0, n_prompt:].tolist()
    gaps = []
    for step, sc in enumerate(res.scores):
        top = torch.topk(sc[0].float().cpu(), 2).values
        if not torch.isfinite(top).all():
            sys.exit(f"step {step}: non-finite logits in {dtype}")
        gaps.append(common.margin_ok(top, f"step {step}", margin))
    text = processor.batch_decode([new_ids], skip_special_tokens=True)[0].strip()
    out = {"text": text, "token_ids": " ".join(map(str, new_ids))}
    metrics = {}
    if mode == "bench":
        for _ in range(2):
            generate(BENCH_NEW, forced=True)
        pre, full = [], []
        for _ in range(BENCH_REPS):
            common.sync()
            t = time.perf_counter()
            generate(1, forced=True)
            common.sync()
            pre.append(time.perf_counter() - t)
            t = time.perf_counter()
            generate(BENCH_NEW, forced=True)
            common.sync()
            full.append(time.perf_counter() - t)
        t_pre, t_full = statistics.median(pre), statistics.median(full)
        metrics["decode_tokens_per_s_b1"] = round((BENCH_NEW - 1) / (t_full - t_pre), 2)
        metrics["prefill_images_per_s_b1"] = round(1 / t_pre, 3)
        metrics["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        out = f"{label or model_id} bench ok"
    common.finish(out, f"{common.device_name()}, torch {torch.__version__}, {dtype}, load {load_s:.0f} s, prompt {n_prompt} tokens {prompt_text!r}, "
                  f"text {text!r}, min top1-top2 gap {min(gaps):.3g} (floor {margin:g}), "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", metrics)
