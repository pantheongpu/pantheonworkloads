"""blip2-opt-2p7b-pytorch: BLIP-2 (OPT-2.7b) captions one fixed image, fp16, greedy decoding.

Functional mode (default): the caption (text and token ids, compared exactly) of the scikit-image sample astronaut.png.
The run fails when any step's top-1 and top-2 logits are within PW_BLIP2_MARGIN (0.1): fp16 logits carry noise of
about 1e-2 and a reference decided by a near tie would flip on another card. Bench mode (PW_BLIP2_MODE=bench, set by
blip2-opt-2p7b-bench): captions per second at batch 8 and decode tokens per second at batch 1.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import common  # noqa: E402
import pinned  # noqa: E402

MODEL = "Salesforce/blip2-opt-2.7b"
REVISION = "59a1ef6c1e5117b3f65523d1c6066825bcf315e3"
IMAGE_URL = ("https://raw.githubusercontent.com/scikit-image/scikit-image/441fe68b95a86d4ae2a351311a0c39a4232b6521/"
             "skimage/data/astronaut.png")
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"
NEW = 20
MODE = os.environ.get("PW_BLIP2_MODE", "functional")
MARGIN = float(os.environ.get("PW_BLIP2_MARGIN", "0.1"))
ALLOW = ["*.json", "merges.txt", "vocab.json", "model-0000*-of-00002.safetensors"]

if common.DEVICE == "cpu":
    common.skip("BLIP-2 is run on a GPU only")
common.setup()
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Blip2ForConditionalGeneration, Blip2Processor  # noqa: E402

rev = os.environ.get("PW_HF_REVISION") or REVISION
cache = os.path.join(os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads")), "hf")
path = common.hf_load(lambda m, r, c: snapshot_download(m, revision=r, cache_dir=c, allow_patterns=ALLOW), MODEL, rev)
sums = os.path.join(os.environ["PW_WORKLOAD_DIR"], "model.sha256")
if not os.environ.get("PW_HF_REVISION"):
    for name, want in pinned.read_sums(sums).items():
        if name.endswith(".safetensors"):
            got = pinned.sha256_file(os.path.join(path, name))
            if got != want:
                sys.exit(f"{name} has sha256 {got}, the pin is {want}")
img_path = common.hf_load(lambda m, r, c: pinned.fetch_asset(IMAGE_URL, IMAGE_SHA256, os.path.join(
    os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads")), "assets", "astronaut.png")), "astronaut.png", None)
image = Image.open(img_path).convert("RGB")

t0 = time.perf_counter()
processor = Blip2Processor.from_pretrained(path)
model = Blip2ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to(common.DEVICE).eval()
load_s = time.perf_counter() - t0


def caption(images, min_new=None):
    inputs = processor(images=images, return_tensors="pt").to(common.DEVICE, torch.float16)
    with torch.no_grad():
        return model.generate(**inputs, max_new_tokens=NEW, min_new_tokens=min_new, do_sample=False, num_beams=1,
                              output_scores=True, return_dict_in_generate=True)


res = caption(image)
seq = res.sequences[0]
gaps = []
for step, sc in enumerate(res.scores):
    top = torch.topk(sc[0].float().cpu(), 2).values
    gaps.append(common.margin_ok(top, f"step {step}", MARGIN))
new_ids = seq[-len(res.scores):].tolist()    # the sequence starts with the 32 image placeholder tokens and the BOS; the new tokens are last
text = processor.batch_decode([new_ids], skip_special_tokens=True)[0].strip()
out = {"caption": text, "token_ids": " ".join(map(str, new_ids))}
metrics = {}
if MODE == "bench":
    common.sync()
    reps = 3
    t = time.perf_counter()
    for _ in range(reps):
        caption([image] * 8, min_new=NEW)
    common.sync()
    metrics["captions_per_s_b8"] = round(reps * 8 / (time.perf_counter() - t), 3)
    t = time.perf_counter()
    for _ in range(reps):
        caption(image, min_new=NEW)
    common.sync()
    metrics["decode_tokens_per_s_b1"] = round(reps * NEW / (time.perf_counter() - t), 2)   # includes the vision tower and Q-Former once per caption
    out = "blip2-bench ok"
common.finish(out, f"{common.device_name()}, torch {torch.__version__}, load {load_s:.0f} s, caption {text!r}, "
              f"min top1-top2 gap {min(gaps):.3g}", metrics)
