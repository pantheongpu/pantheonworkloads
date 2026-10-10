"""Shared body of the catalog's diffusers workloads (image and video models), imported by their main.py.

One fixed prompt and seed go in, a picture or a clip comes out. As in sdxl-base-diffusers the functional output is
statistics, never pixels (per-channel mean and standard deviation, an 8x8 grid of block means, mean absolute spatial
gradient and, for video, the mean absolute change between frames), because GPU kernels are not bit-stable across cards
and library versions and a sampler amplifies tiny differences into different fine detail. Bench mode
(PW_DIFFUSION_MODE=bench, set by the -bench twins): seconds per generation and peak memory, after one warm-up.

Which files are downloaded is whatever the workload's model.sha256 lists (weights, pinned) plus the small configs and
tokenizers; every listed file is verified before the pipeline is built. With PW_DIFFUSION_DEVICE_MAP=balanced (the
default when the host has more than one GPU) diffusers places whole pipeline components on different GPUs.
Never run on a GPU yet: the first run decides whether the tolerances and keyword arguments here hold.
"""
import hashlib
import os
import sys
import time

import numpy as np
import torch

import common
import pinned

PROMPT = "a photograph of an astronaut riding a horse on a beach at golden hour, sharp focus, detailed"
NEGATIVE = "blurry, low quality"
SEED = 1234


def stats(frames, size_hw):
    """frames: float array F x H x W x 3 in 0..1 (F = 1 for an image)."""
    a = np.asarray(frames, dtype=np.float64)
    if not np.isfinite(a).all():
        sys.exit("the output has NaN or infinite values")
    if a.std() < 0.03:
        sys.exit(f"the output is nearly uniform (std {a.std():.4f}): a failed generation")
    f, h, w, _ = a.shape
    k_h, k_w = h // 8, w // 8
    grid = a[:, : k_h * 8, : k_w * 8].reshape(f, 8, k_h, 8, k_w, 3).mean(axis=(0, 2, 4))
    gx = np.abs(np.diff(a, axis=2)).mean()
    gy = np.abs(np.diff(a, axis=1)).mean()
    r = lambda x, n=4: [round(float(v), n) for v in np.ravel(x)]   # noqa: E731
    out = {"frames_hw": [f, h, w], "mean_rgb": r(a.mean(axis=(0, 1, 2))), "std_rgb": r(a.std(axis=(0, 1, 2))),
           "block_means_8x8": r(grid), "mean_abs_gradient": r([gx, gy])}
    if f > 1:
        out["mean_abs_frame_change"] = r(np.abs(np.diff(a, axis=0)).mean())
    return out


def run(model_id, revision, kind, steps, guidance, size=None, width=None, height=None, frames=None, variant=None,
        label=None, kwargs=None, negative=True, dtype="bfloat16", vae_tiling=False):
    mode = os.environ.get("PW_DIFFUSION_MODE", "functional")
    if common.DEVICE == "cpu":
        common.skip("this diffusion model is run on a GPU only")
    common.setup()
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download

    rev = os.environ.get("PW_HF_REVISION") or revision
    sums_path = os.path.join(os.environ["PW_WORKLOAD_DIR"], "model.sha256")
    sums = pinned.read_sums(sums_path)
    allow = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*/*", *sums]
    path = common.hf_load(lambda m, r, c: snapshot_download(m, revision=r, cache_dir=c, allow_patterns=allow), model_id, rev)
    if not os.environ.get("PW_HF_REVISION"):
        pinned.verify_snapshot(path, sums_path)

    ngpu = torch.cuda.device_count()
    dmap = os.environ.get("PW_DIFFUSION_DEVICE_MAP") or ("balanced" if ngpu > 1 else None)
    t0 = time.perf_counter()
    load = dict(torch_dtype=getattr(torch, dtype), use_safetensors=True)
    if variant:
        load["variant"] = variant
    if dmap:
        load["device_map"] = dmap
    pipe = DiffusionPipeline.from_pretrained(path, **load)
    if not dmap:
        pipe.to(common.DEVICE)
    pipe.set_progress_bar_config(disable=True)
    if vae_tiling:   # HunyuanVideo-1.5: the untiled VAE decode of 33 frames at 848x480 ran out of memory on a 44 GiB card (stage 2a, L40S)
        pipe.vae.enable_tiling()
    load_s = time.perf_counter() - t0
    gen_dev = "cuda:0" if dmap else common.DEVICE

    def generate(seed):
        g = torch.Generator(device=gen_dev).manual_seed(seed)
        args = dict(prompt=PROMPT, num_inference_steps=steps, generator=g, output_type="np")
        if guidance is not None:   # None: the pipeline has no guidance_scale argument (Qwen-Image-2.1 takes true_cfg_scale, off by default)
            args["guidance_scale"] = guidance
        if negative:
            args["negative_prompt"] = NEGATIVE
        if kind == "video":
            args.update(height=height, width=width, num_frames=frames)
        else:
            args.update(height=size, width=size)
        args.update(kwargs or {})
        res = pipe(**args)
        arr = np.asarray(res.frames[0] if kind == "video" else res.images[0], dtype=np.float32)
        if arr.shape[-1] == 4:   # RGBA (Qwen-Image-2.1's VAE has 4 channels): composite over white, as the pipeline does for its own vision encoder
            alpha = arr[..., 3:4]
            arr = arr[..., :3] * alpha + (1.0 - alpha)
        return arr[None] if arr.ndim == 3 else arr   # F x H x W x 3

    out_frames = generate(SEED)
    out = stats(out_frames, None)
    px = hashlib.sha256((np.clip(out_frames, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()).hexdigest()[:16]
    detail = (f"{common.device_name()} x{ngpu}, torch {torch.__version__}, diffusers {sys.modules['diffusers'].__version__}, "
              f"{dtype}, {kind} {steps} steps seed {SEED}, load {load_s:.0f} s, pixel sha256 {px} (informational)")
    metrics = {}
    if mode == "bench":
        n = int(os.environ.get("PW_DIFFUSION_RUNS", "2"))   # the generation above was the warm-up
        common.sync()
        t = time.perf_counter()
        for i in range(n):
            generate(SEED + 1 + i)
        common.sync()
        sec = (time.perf_counter() - t) / n
        metrics = {"seconds_per_generation": round(sec, 3), "steps_per_s": round(steps / sec, 3),
                   "peak_gpu_memory_gb": round(max(torch.cuda.max_memory_allocated(i) for i in range(ngpu)) / 1e9, 2)}
        out = f"{label or model_id} bench ok"
    common.finish(out, detail, metrics)
