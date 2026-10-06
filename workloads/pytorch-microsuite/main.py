"""pytorch-microsuite: matmul, conv, attention and friends on seeded inputs, no model download.

Inputs are generated on the CPU from fixed seeds and moved to the device (a GPU's RNG would give
different numbers). Each op reports a few summary numbers, so the output stays small and the
comparison is tolerance-based (manifest.yaml explains the bounds). Op results are also computed
in float64 on the CPU from the same inputs, and the run fails if the device is further from that
than a bound of its own: that catches a backend that is wrong in the same way every time, which a
recorded reference could not.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import common  # noqa: E402

common.setup()
dev = common.DEVICE
g = torch.Generator().manual_seed(1234)


def rnd(*shape, dtype=torch.float32):
    return torch.randn(*shape, generator=g).to(dtype)


def stats(y):
    """[mean |y|, rms, y at three fixed positions]: sign-safe aggregates plus a few raw elements."""
    y = y.detach().double().cpu().flatten()
    n = y.numel()
    return [float(y.abs().mean()), float(y.pow(2).mean().sqrt()), float(y[0]), float(y[n // 3]), float(y[-1])]


# name -> (function of (device, dtype) -> tensor, dtype, allowed max |error| against the float64 CPU
# result, as a fraction of that result's rms). Bounds: see the manifest's notes.
A, B = rnd(192, 256), rnd(256, 160)
BA, BB = rnd(4, 64, 96), rnd(4, 96, 80)
X = rnd(2, 3, 32, 32)
W = rnd(8, 3, 3, 3) * 0.2
Q, K, V = rnd(2, 4, 48, 32), rnd(2, 4, 48, 32), rnd(2, 4, 48, 32)
V1 = rnd(4096)
M2 = rnd(64, 100)
LN = rnd(32, 96)


def attention_manual(q, k, v):
    s = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    mask = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).tril()
    return torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v


OPS = {
    "matmul_fp32": (lambda d, t: A.to(d, t) @ B.to(d, t), torch.float32, 1e-4),
    "matmul_fp16": (lambda d, t: A.to(d, t) @ B.to(d, t), torch.float16, 1e-2),
    "matmul_bf16": (lambda d, t: A.to(d, t) @ B.to(d, t), torch.bfloat16, 1e-1),
    "bmm_fp32": (lambda d, t: torch.bmm(BA.to(d, t), BB.to(d, t)), torch.float32, 1e-4),
    "linear_gelu_fp32": (lambda d, t: F.gelu(F.linear(A.to(d, t), B.t().contiguous().to(d, t))), torch.float32, 1e-4),
    "conv2d_fp32": (lambda d, t: F.conv2d(X.to(d, t), W.to(d, t), padding=1), torch.float32, 1e-4),
    "conv2d_stride2_fp32": (lambda d, t: F.conv2d(X.to(d, t), W.to(d, t), stride=2), torch.float32, 1e-4),
    "sdpa_causal_fp32": (lambda d, t: F.scaled_dot_product_attention(Q.to(d, t), K.to(d, t), V.to(d, t), is_causal=True), torch.float32, 1e-4),
    "attention_manual_fp32": (lambda d, t: attention_manual(Q.to(d, t), K.to(d, t), V.to(d, t)), torch.float32, 1e-4),
    "softmax_fp32": (lambda d, t: torch.softmax(M2.to(d, t) * 3, 1), torch.float32, 1e-4),
    "layernorm_fp32": (lambda d, t: F.layer_norm(LN.to(d, t), (96,)), torch.float32, 1e-4),
    "reductions_fp32": (lambda d, t: torch.stack([V1.to(d, t).sum(), V1.to(d, t).amax(), V1.to(d, t).std()]), torch.float32, 1e-4),
    "cumsum_fp32": (lambda d, t: torch.cumsum(V1.to(d, t), 0), torch.float32, 1e-4),
}

output, worst = {}, 0.0
for name, (fn, dtype, bound) in OPS.items():
    y = fn(dev, dtype).detach().double().cpu()
    ref = fn("cpu", torch.float64)       # same seeded inputs, float64, on the CPU
    rms = float(ref.pow(2).mean().sqrt()) or 1.0
    err = float((y - ref).abs().max()) / rms
    if err > bound:
        sys.exit(f"{name}: differs from the float64 CPU result by {err:.3g} of rms, allowed {bound:g}")
    worst = max(worst, err)
    output[name] = stats(y)

# Exact (integer) results: sort and top-k of the seeded input are computed on the device from
# values that are identical everywhere, so only the sort itself is under test.
v = V1.to(dev)
output["topk_ids"] = " ".join(map(str, torch.topk(v, 8).indices.cpu().tolist()))
output["argsort_head_ids"] = " ".join(map(str, torch.argsort(v)[:8].cpu().tolist()))
output["argmax_rows"] = " ".join(map(str, torch.argmax(M2.to(dev), 1).cpu().tolist()))


def benchmark():
    """Throughput on a real GPU, sizes big enough to fill one. Skips an op that does not fit."""
    out = {}
    n = int(os.environ.get("PW_BENCH_N", 4096))

    def put(key, flops, fn):
        try:
            out[key] = round(flops / common.bench(fn) / 1e12, 3)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    for tag, dt, prec in (("fp32", torch.float32, "highest"), ("tf32", torch.float32, "high"),
                          ("fp16", torch.float16, "highest"), ("bf16", torch.bfloat16, "highest")):
        torch.backends.cuda.matmul.allow_tf32 = prec == "high"
        torch.set_float32_matmul_precision(prec)
        a = torch.randn(n, n, device=dev, dtype=dt)
        b = torch.randn(n, n, device=dev, dtype=dt)
        put(f"matmul_{tag}_tflops", 2 * n ** 3, lambda: a @ b)
        del a, b
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    x = torch.randn(32, 128, 56, 56, device=dev, dtype=torch.float16)
    w = torch.randn(128, 128, 3, 3, device=dev, dtype=torch.float16)
    put("conv2d_fp16_tflops", 2 * 32 * 128 * 128 * 9 * 56 * 56, lambda: F.conv2d(x, w, padding=1))
    del x, w
    b_, h_, s_, d_ = 8, 16, 2048, 64
    q, k, vv = (torch.randn(b_, h_, s_, d_, device=dev, dtype=torch.float16) for _ in range(3))
    put("sdpa_causal_fp16_tflops", 2 * b_ * h_ * s_ * s_ * d_, lambda: F.scaled_dot_product_attention(q, k, vv, is_causal=True))
    del q, k, vv
    buf = torch.empty(2 ** 28, device=dev, dtype=torch.float32)   # 1 GiB
    dst = torch.empty_like(buf)
    sec = common.bench(lambda: dst.copy_(buf))
    out["copy_gb_s"] = round(2 * buf.numel() * 4 / sec / 1e9, 1)
    return out


metrics = benchmark() if (dev != "cpu" and common.REAL and os.environ.get("PW_NO_BENCH") != "1") else {}
common.finish(output, f"{len(OPS)} ops + 3 index checks on {common.device_name()}, "
              f"worst error {worst:.2e} of rms vs the float64 CPU result, torch {torch.__version__}", metrics)
