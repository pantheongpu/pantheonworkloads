"""lib-attention-precision: scaled-dot-product attention through each backend (math, memory-efficient, flash,
cuDNN), int8 / fp8 / quantised matmul, fp16 / bf16 / TF32 behaviour and autocast."""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import libkernels as lk

SUITE = lk.Suite("lib-attention-precision")
op = SUITE.op
D = lk.Data(20250104)

# ---------------------------------------------------------------------------------------------- SDPA
B_, H_, HK, L, LK, E = 2, 4, 2, 50, 77, 64      # L = 50 and LK = 77: not multiples of any tile size
Q = D.randn(B_, H_, L, E)
K = D.randn(B_, H_, L, E)
V = D.randn(B_, H_, L, E)
QG, KG, VG = D.randn(B_, 8, L, E), D.randn(B_, HK, L, E), D.randn(B_, HK, L, E)    # 8 query heads share 2 kv heads
QX, KX, VX = D.randn(B_, H_, L, E), D.randn(B_, H_, LK, E), D.randn(B_, H_, LK, E)  # cross attention, L != LK
Q40, K40, V40 = D.randn(B_, H_, L, 40), D.randn(B_, H_, L, 40), D.randn(B_, H_, L, 40)   # head dim not a multiple of 16
QD, KD, VD = D.randn(B_, 8, 1, E), D.randn(B_, HK, LK, E), D.randn(B_, HK, LK, E)         # one decode step with a KV cache
BMASK = D.rand(B_, 1, L, L) < 0.6
BMASK |= torch.eye(L, dtype=torch.bool)                        # at least the diagonal: no empty row
FMASK = D.randn(B_, H_, L, L) * 2                              # additive bias (ALiBi / relative-position style)

BACKENDS = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
DTYPES = {"fp32": (torch.float32, 1e-4), "fp16": (torch.float16, 2e-2), "bf16": (torch.bfloat16, 1e-1)}


def sdpa(c, backend, q, k, v, **kw):
    # the reference (float64, CPU) always takes the default path; only the run is pinned to one backend
    ctx = contextlib.nullcontext() if c.is_ref else sdpa_kernel([BACKENDS[backend]])
    with ctx:
        return F.scaled_dot_product_attention(c(q), c(k), c(v), dropout_p=0.0, **{
            n: (c.raw(x) if x.dtype == torch.bool else c(x)) if torch.is_tensor(x) else x for n, x in kw.items()})


VARIANTS = {
    "plain": lambda c, b: sdpa(c, b, Q, K, V),
    "causal": lambda c, b: sdpa(c, b, Q, K, V, is_causal=True),
    "gqa_causal": lambda c, b: sdpa(c, b, QG, KG, VG, is_causal=True, enable_gqa=True),
    "cross_lengths": lambda c, b: sdpa(c, b, QX, KX, VX),
    "head_dim_40": lambda c, b: sdpa(c, b, Q40, K40, V40, is_causal=True),
    "bool_mask": lambda c, b: sdpa(c, b, Q, K, V, attn_mask=BMASK),
    "float_bias": lambda c, b: sdpa(c, b, Q, K, V, attn_mask=FMASK),
    "decode_gqa_kv_cache": lambda c, b: sdpa(c, b, QD, KD, VD, enable_gqa=True),
}
# which (backend, dtype, variants) are exercised: the full cross product is large and mostly redundant
PLAN = {
    ("math", "fp32"): list(VARIANTS), ("math", "fp16"): ["causal", "gqa_causal"], ("math", "bf16"): ["causal"],
    ("efficient", "fp32"): ["plain", "causal", "float_bias", "cross_lengths"],
    ("efficient", "fp16"): ["causal", "bool_mask", "float_bias", "head_dim_40"],
    ("efficient", "bf16"): ["causal", "gqa_causal"],
    ("flash", "fp32"): ["causal"],
    ("flash", "fp16"): ["plain", "causal", "gqa_causal", "cross_lengths", "head_dim_40", "decode_gqa_kv_cache"],
    ("flash", "bf16"): ["causal", "gqa_causal"],
    ("cudnn", "fp16"): ["plain", "causal"], ("cudnn", "bf16"): ["causal"],
}
for (_b, _t), _vs in PLAN.items():
    for _v in _vs:
        @op(f"sdpa_{_b}_{_t}_{_v}", dtype=DTYPES[_t][0], bound=DTYPES[_t][1])
        def _(c, _b=_b, _v=_v):
            return VARIANTS[_v](c, _b)


@op("sdpa_backward_math_fp32", bound=1e-3)
def _(c):
    q, k, v = c.leaf(Q), c.leaf(K), c.leaf(V)
    ctx = contextlib.nullcontext() if c.is_ref else sdpa_kernel([SDPBackend.MATH])
    with ctx:
        F.scaled_dot_product_attention(q, k, v, is_causal=True).pow(2).sum().backward()
    return q.grad, k.grad, v.grad


for _b in ("flash", "efficient"):
    @op(f"sdpa_backward_{_b}_fp16", dtype=torch.float16, bound=5e-2)
    def _(c, _b=_b):
        q, k, v = c.leaf(Q), c.leaf(K), c.leaf(V)
        ctx = contextlib.nullcontext() if c.is_ref else sdpa_kernel([BACKENDS[_b]])
        with ctx:
            F.scaled_dot_product_attention(q, k, v, is_causal=True).pow(2).sum().backward()
        return q.grad, k.grad, v.grad


@op("sdpa_default_dispatch_fp16_causal", dtype=torch.float16, bound=2e-2, note="whatever backend torch picks")
def _(c):
    return F.scaled_dot_product_attention(c(Q), c(K), c(V), is_causal=True)


@op("multi_head_attention_module_fp32", bound=1e-4)
def _(c):
    m = c.module(lk.seeded(21, lambda: nn.MultiheadAttention(64, 4, batch_first=True)).eval())
    x = c(Q[:, 0])
    with torch.no_grad():
        return m(x, x, x, need_weights=False)[0]


# ---------------------------------------------------------------------------------------------- TF32 / half precision
A = D.randn(64, 256)
Bm = D.randn(256, 96)
ALONG, BLONG = D.randn(32, 2048), D.randn(2048, 40)
XC = D.randn(2, 16, 28, 28)
WC = D.randn(16, 16, 3, 3) * 0.1
HALFV = D.randn(4096) * 0.5 + 0.1


class tf32:
    """Allow TF32 inside the block, restore after (a no-op on CPU and on GPUs without TF32)."""

    def __enter__(self):
        self.old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def __exit__(self, *a):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.old[:2]
        torch.set_float32_matmul_precision(self.old[2])


@op("matmul_tf32_enabled", bound=6e-3, prec=torch.float16, note="TF32 keeps 10 mantissa bits: the bound is that, not fp32's")
def _(c):
    with tf32():
        return c(A) @ c(Bm)


@op("matmul_fp32_strict_after_tf32", bound=5e-5, note="the flags are restored: exact fp32 again")
def _(c):
    with tf32():
        pass
    return c(A) @ c(Bm)


@op("conv2d_tf32_enabled", bound=6e-3, prec=torch.float16)
def _(c):
    with tf32():
        return F.conv2d(c(XC), c(WC), padding=1)


@op("bmm_tf32_enabled", bound=6e-3, prec=torch.float16)
def _(c):
    with tf32():
        return torch.bmm(c(A).view(4, 16, 256), c(Bm[:, :64]).T.contiguous().view(4, 16, 256).transpose(1, 2))


@op("matmul_fp16_long_k_fp32_accumulate", dtype=torch.float16, bound=1e-2, note="K = 2048: fp16 accumulation would fail this")
def _(c):
    return c(ALONG) @ c(BLONG)


@op("matmul_bf16_long_k_fp32_accumulate", dtype=torch.bfloat16, bound=1e-1)
def _(c):
    return c(ALONG) @ c(BLONG)


@op("sum_mean_fp16_wide_accumulator", dtype=torch.float16, bound=1e-2)
def _(c):
    x = c(HALFV)
    return torch.stack([x.sum(), x.mean(), x.sum() / 4096, x.float().sum().half()])


@op("cumsum_softmax_logsumexp_bf16", dtype=torch.bfloat16, bound=1e-1)
def _(c):
    x = c(HALFV[:1024].view(32, 32))
    return torch.cumsum(x, 1), torch.softmax(x * 8, 1), torch.logsumexp(x * 8, 1)


@op("softmax_fp16_large_logits", dtype=torch.float16, bound=2e-2)
def _(c):
    return torch.softmax(c(A * 25), 1)               # logits up to about 100: needs the max subtraction


@op("half_overflow_and_underflow_counts", dtype=torch.float16, kind="int",
    ref=lambda c: torch.tensor([3, 3, 1]), note="counts: inf after x+x, nan after y-y, zeros (1e-8 underflows)")
def _(c):
    x = torch.tensor([1.0, 65504.0, 70000.0, -70000.0, 1e-8]).to(c.dev).to(torch.float16)   # the cast itself is on the device
    y = x + x
    return torch.stack([torch.isinf(y).sum(), torch.isnan(y - y).sum(), (y == 0).sum()])


CASTV = torch.cat([D.randn(300) * 3, torch.tensor([1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 2.0 ** -24, 2.0 ** -25 * 1.5, 448.0, 449.0, 57344.0])])


def cast_roundtrip(name, dtype, bound):
    @op(f"cast_roundtrip_{name}", bound=bound, note="the cast runs on the device: rounding mode and ties")
    def _(c):
        x = c.raw(CASTV.double() if c.is_ref else CASTV)
        return x.to(dtype).to(torch.float64 if c.is_ref else torch.float32)


cast_roundtrip("fp16", torch.float16, 1e-9)
cast_roundtrip("bf16", torch.bfloat16, 1e-9)
cast_roundtrip("fp8_e4m3fn", torch.float8_e4m3fn, 1e-9)
cast_roundtrip("fp8_e5m2", torch.float8_e5m2, 1e-9)

# ---------------------------------------------------------------------------------------------- int8 / fp8 / quantised
QA = D.randint(-127, 128, (32, 64)).to(torch.int8)
QB = D.randint(-127, 128, (64, 48)).to(torch.int8)
SA, SB = D.rand(32) * 0.02 + 0.005, D.rand(48) * 0.02 + 0.005
XF = D.randn(32, 64)
WQ = D.randint(-127, 128, (48, 64)).to(torch.int8)       # (out, in) as _weight_int8pack_mm wants it


@op("int8_matmul_int32_exact", kind="int", ref=lambda c: QA.long() @ QB.long(), note="torch._int_mm: exact int32 accumulation")
def _(c):
    return torch._int_mm(c.raw(QA), c.raw(QB))


@op("int8_linear_with_dequantise", bound=1e-5)
def _(c):
    if c.is_ref:
        return (QA.double() @ QB.double()) * SA.double()[:, None] * SB.double()[None, :]
    return torch._int_mm(c.raw(QA), c.raw(QB)).to(c.dtype) * c(SA)[:, None] * c(SB)[None, :]


@op("int8_symmetric_quantise_round_clamp", kind="int", ref=lambda c: torch.clamp(torch.round(XF.double() * 40 / 3), -128, 127).long())
def _(c):
    s = 3.0 / 40
    return torch.clamp(torch.round(c(XF) / s), -128, 127).to(torch.int8)


@op("quantize_per_tensor_qint8_roundtrip", bound=1e-6)
def _(c):
    if c.is_ref:
        return torch.clamp(torch.round(c(XF) / 0.05) + 3, -128, 127).sub(3).mul(0.05)
    return torch.dequantize(torch.quantize_per_tensor(c(XF), 0.05, 3, torch.qint8))


@op("fake_quantize_per_channel_affine", bound=1e-6)
def _(c):
    s = c(SA)
    if c.is_ref:
        return torch.clamp(torch.round(c(XF) / s[:, None]), -128, 127) * s[:, None]
    return torch.fake_quantize_per_channel_affine(c(XF), s, torch.zeros(32, dtype=torch.int32, device=c.dev), 0, -128, 127)


@op("weight_only_int8_linear_bf16", dtype=torch.bfloat16, bound=1e-1, note="torch._weight_int8pack_mm (weight-only int8 as used by LLM inference)")
def _(c):
    if c.is_ref:
        return c(XF) @ (WQ.double() * SB.double()[:, None]).T
    return torch._weight_int8pack_mm(c(XF), c.raw(WQ), c(SB))


@op("dynamic_quantized_linear", bound=1.5e-1, prec=torch.bfloat16, note="torch.ao dynamic int8 Linear: a CPU (fbgemm/onednn) path; quantisation noise bounds it")
def _(c):
    if c.is_ref:
        lin = lk.seeded(22, lambda: nn.Linear(64, 48))
        return c(XF) @ c(lin.weight.data).T + c(lin.bias.data)
    if c.dev != "cpu":
        raise NotImplementedError("dynamic quantized Linear runs on the CPU only")
    lin = lk.seeded(22, lambda: nn.Sequential(nn.Linear(64, 48)))
    return torch.ao.quantization.quantize_dynamic(lin, {nn.Linear}, dtype=torch.qint8)(c(XF))


FP8A, FP8B = D.randn(32, 64), D.randn(48, 64)


@op("fp8_scaled_mm_e4m3", bound=1e-1, prec=torch.bfloat16, note="torch._scaled_mm; needs fp8 matrix cores (sm89+/gfx942+)")
def _(c):
    if c.is_ref:
        a = FP8A.to(torch.float8_e4m3fn).double()
        b = FP8B.to(torch.float8_e4m3fn).double()
        return a @ b.T
    a = c.raw(FP8A).to(torch.float8_e4m3fn)
    b = c.raw(FP8B).to(torch.float8_e4m3fn)
    one = torch.ones((), device=c.dev, dtype=torch.float32)
    return torch._scaled_mm(a, b.T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)


# ---------------------------------------------------------------------------------------------- autocast
AUTO_X = D.randn(8, 32, 64)
AUTO_W1, AUTO_W2 = D.randn(128, 64) * 0.15, D.randn(64, 128) * 0.1
AUTO_B1 = D.randn(128) * 0.1


def autocast_ctx(c, dtype):
    if c.is_ref:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(c.dev).type, dtype=dtype)


for _name, _dt, _bound in (("fp16", torch.float16, 2e-2), ("bf16", torch.bfloat16, 1e-1)):
    @op(f"autocast_{_name}_linear_dtype_and_value", bound=_bound, prec=_dt)
    def _(c, _dt=_dt):
        with autocast_ctx(c, _dt):
            y = F.linear(c(AUTO_X), c(AUTO_W1), c(AUTO_B1))
        if not c.is_ref and y.dtype != _dt:
            raise lk.Fail(f"linear under autocast({_dt}) gave {y.dtype}")
        return y

    @op(f"autocast_{_name}_mlp_chain_with_layer_norm", bound=_bound, prec=_dt)
    def _(c, _dt=_dt):
        with autocast_ctx(c, _dt):
            h = F.gelu(F.linear(c(AUTO_X), c(AUTO_W1), c(AUTO_B1)))
            y = F.layer_norm(F.linear(h, c(AUTO_W2)) + c(AUTO_X), (64,))
        if not c.is_ref and y.dtype != torch.float32:
            raise lk.Fail(f"layer_norm under autocast({_dt}) should return float32, got {y.dtype}")
        return y

    @op(f"autocast_{_name}_softmax_and_loss_stay_fp32", bound=_bound * 2, prec=_dt, note="the CPU autocast policy leaves softmax in the low dtype, CUDA's makes it fp32: the bound covers both")
    def _(c, _dt=_dt):
        with autocast_ctx(c, _dt):
            logits = F.linear(c(AUTO_X), c(AUTO_W1[:100]))
            p = torch.softmax(logits, -1)
            loss = F.cross_entropy(logits.flatten(0, 1), c.raw(torch.arange(256) % 100))
        if not c.is_ref and c.dev != "cpu" and (p.dtype != torch.float32 or loss.dtype != torch.float32):   # CUDA's autocast policy; CPU's list differs
            raise lk.Fail(f"softmax/cross_entropy under autocast({_dt}) should be float32, got {p.dtype}/{loss.dtype}")
        return p, loss

    @op(f"autocast_{_name}_conv_bmm_sdpa", bound=_bound * 2, prec=_dt)
    def _(c, _dt=_dt):
        with autocast_ctx(c, _dt):
            y1 = F.conv2d(c(XC), c(WC), padding=1)
            y2 = torch.bmm(c(AUTO_X), c(AUTO_X).transpose(1, 2))
            y3 = F.scaled_dot_product_attention(c(Q), c(K), c(V), is_causal=True)
        if not c.is_ref and not (y1.dtype == y2.dtype == y3.dtype == _dt):
            raise lk.Fail(f"conv/bmm/sdpa under autocast({_dt}) gave {y1.dtype}/{y2.dtype}/{y3.dtype}")
        return y1, y2 / 8, y3

    @op(f"autocast_{_name}_backward_keeps_fp32_grads", bound=_bound * 2, prec=_dt)
    def _(c, _dt=_dt):
        w = c.leaf(AUTO_W1)
        with autocast_ctx(c, _dt):
            y = F.linear(c(AUTO_X), w)
        y.float().pow(2).mean().backward()
        if not c.is_ref and w.grad.dtype != torch.float32:
            raise lk.Fail(f"the gradient of an fp32 parameter under autocast should be fp32, got {w.grad.dtype}")
        return w.grad


def bench(b):
    dev = lk.DEVICE
    bs, h, s, e = b.size(8, 1), b.size(16, 2), b.size(2048, 64), 64
    for tag, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
        q, k, v = (torch.randn(bs, h, s, e, device=dev, dtype=dt) for _ in range(3))
        for name, backend in BACKENDS.items():
            def run(backend=backend, q=q, k=k, v=v):
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            b.put(f"sdpa_{name}_{tag}_causal_tflops", 2 * bs * h * s * s * e, 1e12, run)   # 4 B H S^2 E, halved by causality
    qg = torch.randn(bs, 8 * h // 2, s, e, device=dev, dtype=torch.float16)
    kg = torch.randn(bs, h // 2, s, e, device=dev, dtype=torch.float16)
    b.put("sdpa_gqa_flash_fp16_causal_tflops", 2 * bs * (8 * h // 2) * s * s * e, 1e12,
          lambda: F.scaled_dot_product_attention(qg, kg, kg, is_causal=True, enable_gqa=True))
    n = b.size(8192, 256)
    for tag, dt, prec in (("fp32_highest", torch.float32, "highest"), ("tf32", torch.float32, "high"),
                          ("fp16", torch.float16, "highest"), ("bf16", torch.bfloat16, "highest")):
        torch.backends.cuda.matmul.allow_tf32 = prec == "high"
        torch.set_float32_matmul_precision(prec)
        x, y = torch.randn(n, n, device=dev, dtype=dt), torch.randn(n, n, device=dev, dtype=dt)
        b.put(f"matmul_{tag}_tflops", 2 * n ** 3, 1e12, lambda: x @ y, iters=5)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    qa = torch.randint(-127, 128, (n, n), device=dev, dtype=torch.int8)
    qb = torch.randint(-127, 128, (n, n), device=dev, dtype=torch.int8)
    b.put("int8_mm_tops", 2 * n ** 3, 1e12, lambda: torch._int_mm(qa, qb), iters=5)
    a8, b8 = qa.to(torch.float8_e4m3fn), qb.t().contiguous().to(torch.float8_e4m3fn).t()
    one = torch.ones((), device=dev)
    b.put("fp8_e4m3_scaled_mm_tflops", 2 * n ** 3, 1e12, lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16), iters=5)
    wl = torch.randn(n, n, device=dev)
    xl = torch.randn(n, n, device=dev)
    for tag, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        def run(dt=dt):
            with torch.autocast("cuda", dtype=dt):
                return F.linear(xl, wl)
        b.put(f"autocast_{tag}_linear_tflops", 2 * n ** 3, 1e12, run, iters=5)
