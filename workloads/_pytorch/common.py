"""Helpers shared by the pytorch-* workloads (imported by their main.py).

The contract (docs/workload-contract.md): the result is written to $PW_RESULT_FILE as one JSON
object {output, detail, metrics}; env.sh prints it as the last stdout line. Exit 77 = cannot run here.
"""
import json
import os
import sys

import torch

REAL = os.environ.get("PW_TARGET", "cpu") in ("cpu", "gpu")   # metrics only on real targets
DEVICE = os.environ.get("PW_DEVICE", "cpu")


def skip(why):
    print(f"SKIP: {why}", flush=True)
    sys.exit(77)


def setup():
    """Deterministic, full-precision settings: no TF32 (it would make GPU and CPU differ by ~1e-3)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        skip(f"PW_DEVICE={DEVICE} but torch.cuda is not available in this build/target")
    torch.manual_seed(0)


def device_name():
    return torch.cuda.get_device_name(0) if DEVICE != "cpu" else "cpu"


def sync():
    if DEVICE != "cpu":
        torch.cuda.synchronize()


def bench(fn, iters=20, rounds=5, warmup=3):
    """Median seconds per call of fn() over `rounds` rounds of `iters` calls."""
    import statistics
    import time
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        times.append((time.perf_counter() - t) / iters)
    return statistics.median(times)


def margin_ok(top_values, what, floor=1e-3):
    """top_values: descending, one more than the ids kept. The kept ids (and their order) are only
    a stable reference when every neighbouring gap is above floating-point noise. Fails the run
    (not a SKIP) otherwise."""
    gap = min(float(a - b) for a, b in zip(top_values[:-1], top_values[1:]))
    if gap < floor:
        sys.exit(f"{what}: two neighbouring top-k scores differ by only {gap:g} (< {floor:g}); too close to compare across backends")
    return gap


def finish(output, detail, metrics=None):
    rec = {"output": output, "detail": detail, "metrics": (metrics or {}) if REAL else {}}
    with open(os.environ["PW_RESULT_FILE"], "w") as f:
        json.dump(rec, f)
        f.write("\n")
    print(json.dumps(rec), flush=True)


def hf_load(factory, model_id, revision):
    """Run factory(model_id, revision, cache_dir) with downloads from the Hub; SKIP when it cannot."""
    cache = os.path.join(os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads")), "hf")
    os.makedirs(cache, exist_ok=True)
    try:
        return factory(model_id, revision, cache)
    except Exception as e:   # no network, no such revision, gated...: not a failure of the GPU
        skip(f"cannot fetch {model_id} ({type(e).__name__}: {str(e).splitlines()[0][:150]})")


def verify_pinned(model_id, revision, filename):
    """Fail (not SKIP) when the weights file in the cache is not the one pinned in the workload's model.sha256
    ("<sha256>  <filename>" lines). Only checked when the run uses the pinned revision (PW_HF_REVISION unset)."""
    if os.environ.get("PW_HF_REVISION"):
        return
    import hashlib
    from huggingface_hub import hf_hub_download
    sums = {}
    with open(os.path.join(os.environ["PW_WORKLOAD_DIR"], "model.sha256")) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                s, n = line.split()
                sums[n] = s
    path = hf_hub_download(model_id, filename, revision=revision,
                           cache_dir=os.path.join(os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads")), "hf"))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != sums[filename]:
        sys.exit(f"{filename} of {model_id}@{revision} has sha256 {h.hexdigest()}, the pin is {sums[filename]}")


def hub_licence(model_id):
    """The licence the model card declares right now, for the run's detail line (None if unknown)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id).card_data.license
    except Exception:
        return None
