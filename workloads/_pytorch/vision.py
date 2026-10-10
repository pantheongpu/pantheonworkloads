"""Shared body of the catalog's vision-encoder, segmentation and depth workloads, imported by their main.py.

All heads look at the pinned public-domain photo of the other workloads (scikit-image's astronaut.png, NASA).
  siglip2   zero-shot image-text matching: sigmoid probabilities of three captions; output = the probabilities
            (the astronaut caption must win)
  dino      (DINOv2, DINOv3; "dinov3" is accepted too) the L2-normalised CLS embedding: output = its norm before normalising, 8 coarse statistics of it and the
            cosine to the embedding of the horizontally flipped image
  sam2      SAM 2.x with one point prompt on the astronaut's face: output = the fraction of the image the best mask
            covers and its bounding box
  depth     Depth Anything V2: output = mean/std of the normalised depth map and an 8x8 grid of block means
Bench mode (PW_VISION_MODE=bench): images per second at batch 1. Run on an A10G (stage 2a).
"""
import os
import sys
import time

import numpy as np
import torch

import catalog_common as cc
import common
import pinned

CAPTIONS = ["a photo of an astronaut in an orange space suit", "a photo of a cat on a sofa", "a photo of a mountain landscape"]
IMAGE_URL = ("https://raw.githubusercontent.com/scikit-image/scikit-image/441fe68b95a86d4ae2a351311a0c39a4232b6521/"
             "skimage/data/astronaut.png")
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"
FACE_POINT = (256, 100)   # x, y of the astronaut's face in the 512x512 image


def run(model_id, revision, head, label=None, dtype="float32"):
    mode = os.environ.get("PW_VISION_MODE", "functional")
    if common.DEVICE == "cpu":
        common.skip("this vision model is run on a GPU only")
    common.setup()
    from PIL import Image
    from transformers import AutoModel, AutoModelForDepthEstimation, AutoProcessor
    path = cc.fetch_snapshot(model_id, revision)
    img_path = common.hf_load(lambda m, r, c: pinned.fetch_asset(
        IMAGE_URL, IMAGE_SHA256, os.path.join(cc.cache_root(), "assets", "astronaut.png")), "astronaut.png", None)
    image = Image.open(img_path).convert("RGB")
    td = getattr(torch, dtype)
    dmap = os.environ.get("PW_DEVICE_MAP") or "auto"
    t0 = time.perf_counter()
    proc = None if head == "sam2" else AutoProcessor.from_pretrained(path)

    if head == "siglip2":
        model = AutoModel.from_pretrained(path, dtype=td, device_map=dmap).eval()

        def go():
            enc = proc(text=CAPTIONS, images=image, padding="max_length", max_length=64, return_tensors="pt").to(common.DEVICE)
            enc = {k: (v.to(td) if v.is_floating_point() else v) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits_per_image[0].float()
            return [round(float(p), 4) for p in torch.sigmoid(logits)]

        def check(r):
            if max(range(3), key=lambda i: r[i]) != 0:
                sys.exit(f"the astronaut caption does not win: {r}")
            return {"probabilities": r}
    elif head in ("dino", "dinov3"):
        model = AutoModel.from_pretrained(path, dtype=td, device_map=dmap).eval()

        def go():
            enc = proc(images=[image, image.transpose(Image.FLIP_LEFT_RIGHT)], return_tensors="pt").to(common.DEVICE)
            enc = {k: (v.to(td) if v.is_floating_point() else v) for k, v in enc.items()}
            with torch.no_grad():
                h = model(**enc).last_hidden_state[:, 0].float()
            e = torch.nn.functional.normalize(h, dim=-1)
            return {"cls_norm": round(float(h[0].norm()), 3), "stats": [round(float(x), 4) for x in
                    (e[0].mean(), e[0].std(), e[0].min(), e[0].max(), e[0][:64].mean(), e[0][64:].mean(), e[0].abs().mean(), e[0].pow(2).sum())],
                    "flip_cosine": round(float(e[0] @ e[1]), 4)}

        def check(r):
            if not (0.0 < r["flip_cosine"] <= 1.0):
                sys.exit(f"implausible cosine to the flipped image: {r}")
            return r
    elif head == "sam2":
        from transformers import Sam2Model, Sam2Processor
        proc = Sam2Processor.from_pretrained(path)
        model = Sam2Model.from_pretrained(path, dtype=td, device_map=dmap).eval()

        def go():
            enc = proc(images=image, input_points=[[[list(FACE_POINT)]]], input_labels=[[[1]]], return_tensors="pt").to(common.DEVICE)
            enc = {k: (v.to(td) if v.is_floating_point() else v) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, multimask_output=False)
            masks = proc.post_process_masks(out.pred_masks.cpu(), enc["original_sizes"].cpu())[0]
            m = masks[0, 0].bool().numpy() if masks.ndim == 4 else masks[0].bool().numpy()
            ys, xs = np.where(m)
            return {"area_fraction": round(float(m.mean()), 4),
                    "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]}

        def check(r):
            if not (0.001 < r["area_fraction"] < 0.9):
                sys.exit(f"implausible mask: {r}")
            return r
    elif head == "depth":
        model = AutoModelForDepthEstimation.from_pretrained(path, dtype=td, device_map=dmap).eval()

        def go():
            enc = proc(images=image, return_tensors="pt").to(common.DEVICE)
            enc = {k: (v.to(td) if v.is_floating_point() else v) for k, v in enc.items()}
            with torch.no_grad():
                d = model(**enc).predicted_depth.float()[0]
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            h, w = d.shape
            grid = d[: h // 8 * 8, : w // 8 * 8].reshape(8, h // 8, 8, w // 8).mean(dim=(1, 3))
            return {"mean": round(float(d.mean()), 4), "std": round(float(d.std()), 4),
                    "grid_8x8": [round(float(x), 3) for x in grid.flatten()]}

        def check(r):
            if r["std"] < 0.02:
                sys.exit(f"a flat depth map: {r}")
            return r
    else:
        sys.exit(f"unknown head {head}")

    load_s = time.perf_counter() - t0
    out = check(go())
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
        metrics = {"images_per_s_b1": round(n / (time.perf_counter() - t), 2),
                   "peak_gpu_memory_gb": round(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9, 2)}
        out = f"{label or model_id} bench ok"
    common.finish(out, f"{common.device_name()}, torch {torch.__version__}, {dtype}, head {head}, load {load_s:.0f} s", metrics)
