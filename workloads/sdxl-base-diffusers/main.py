"""sdxl-base-diffusers: Stable Diffusion XL base 1.0 (fp16, diffusers), one 1024x1024 image from a fixed prompt and seed.

Functional mode (default): the output is image statistics, never pixels: GPU kernels are not bit-stable across
cards or library versions, and a diffusion sampler amplifies tiny differences into different fine detail while the
composition stays the same. Compared (manifest tolerance): per-channel mean and standard deviation, an 8x8 grid of
block means (the coarse layout and colours) and the mean absolute horizontal/vertical gradient (overall sharpness),
all on a 0..1 scale. Bench mode (PW_SDXL_MODE=bench, set by sdxl-base-bench): images per second.
"""
import hashlib
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import common  # noqa: E402
import pinned  # noqa: E402

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
PROMPT = "a photograph of an astronaut riding a horse on a beach at golden hour, sharp focus, detailed"
NEGATIVE = "blurry, low quality"
STEPS = 25
SIZE = 1024
GUIDANCE = 5.0
SEED = 1234
MODE = os.environ.get("PW_SDXL_MODE", "functional")
ALLOW = ["model_index.json", "scheduler/*.json", "tokenizer/*", "tokenizer_2/*", "text_encoder/config.json",
         "text_encoder/model.fp16.safetensors", "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
         "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
         "vae/diffusion_pytorch_model.fp16.safetensors"]

if common.DEVICE == "cpu":
    common.skip("SDXL base is run on a GPU only")
common.setup()
from diffusers import StableDiffusionXLPipeline  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

rev = os.environ.get("PW_HF_REVISION") or REVISION
cache = os.path.join(os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads")), "hf")
path = common.hf_load(lambda m, r, c: snapshot_download(m, revision=r, cache_dir=c, allow_patterns=ALLOW), MODEL, rev)
if not os.environ.get("PW_HF_REVISION"):
    pinned.verify_snapshot(path, os.path.join(os.environ["PW_WORKLOAD_DIR"], "model.sha256"))

t0 = time.perf_counter()
pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to(common.DEVICE)
pipe.set_progress_bar_config(disable=True)
load_s = time.perf_counter() - t0


def generate(seed):
    g = torch.Generator(device=common.DEVICE).manual_seed(seed)
    return pipe(PROMPT, negative_prompt=NEGATIVE, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                height=SIZE, width=SIZE, generator=g, output_type="np").images[0]   # float32 HxWx3 in 0..1


def stats(img):
    a = np.asarray(img, dtype=np.float64)
    if not np.isfinite(a).all():
        sys.exit("the image has NaN or infinite values")
    if a.std() < 0.05:
        sys.exit(f"the image is nearly uniform (std {a.std():.4f}): a failed generation")
    k = SIZE // 8
    grid = a.reshape(8, k, 8, k, 3).mean(axis=(1, 3))
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    r = lambda x, n=4: [round(float(v), n) for v in np.ravel(x)]   # noqa: E731
    return {"mean_rgb": r(a.mean(axis=(0, 1))), "std_rgb": r(a.std(axis=(0, 1))), "block_means_8x8": r(grid),
            "mean_abs_gradient": r([gx, gy])}


img = generate(SEED)
if os.environ.get("PW_OUT"):   # keep the picture next to the run log, for a human to look at (not compared)
    from PIL import Image
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(os.environ["PW_OUT"], "sdxl-base-diffusers.png"))
out = stats(img)
px = hashlib.sha256((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()).hexdigest()[:16]
detail = (f"{common.device_name()}, torch {torch.__version__}, diffusers {sys.modules['diffusers'].__version__}, "
          f"{SIZE}x{SIZE} {STEPS} steps seed {SEED}, load {load_s:.0f} s, pixel sha256 {px} (informational)")
metrics = {}
if MODE == "bench":
    n = int(os.environ.get("PW_SDXL_IMAGES", "3"))   # the first image above was the warm-up
    common.sync()
    t = time.perf_counter()
    for i in range(n):
        generate(SEED + 1 + i)
    common.sync()
    sec = time.perf_counter() - t
    metrics = {"images_per_s_1024_25steps": round(n / sec, 4), "unet_steps_per_s": round(n * STEPS / sec, 3)}
    out = "sdxl-bench ok"
common.finish(out, detail, metrics)
