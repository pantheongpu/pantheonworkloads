"""lib-composite-blocks: small random-initialised blocks from real model families, one forward (or one denoise
step) each: a ViT encoder block, a diffusion-style UNet step (ResBlock + attention), a DLRM-style recommender
forward. No pretrained weights: seeded random init, so they are a functional check of the kernel mix, not a model."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import libkernels as lk

SUITE = lk.Suite("lib-composite-blocks")
op = SUITE.op
D = lk.Data(20250105)


# ---------------------------------------------------------------------------------------------- ViT
class ViT(nn.Module):
    """Patch embedding, class token, learned positions, `depth` pre-LayerNorm encoder blocks, head."""

    def __init__(self, dim=64, heads=4, depth=2, patch=4, image=16, classes=10):
        super().__init__()
        self.heads = heads
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        n = (image // patch) ** 2 + 1
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.5)
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.5)
        self.ln1 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.qkv = nn.ModuleList(nn.Linear(dim, 3 * dim) for _ in range(depth))
        self.proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.ln2 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.fc1 = nn.ModuleList(nn.Linear(dim, 4 * dim) for _ in range(depth))
        self.fc2 = nn.ModuleList(nn.Linear(4 * dim, dim) for _ in range(depth))
        self.ln_f, self.head = nn.LayerNorm(dim), nn.Linear(dim, classes)

    def forward(self, img):
        x = self.patch(img).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1).to(x.dtype), x], 1) + self.pos.to(x.dtype)
        for i in range(len(self.qkv)):
            b, t, e = x.shape
            q, k, v = self.qkv[i](self.ln1[i](x)).view(b, t, 3, self.heads, e // self.heads).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, e)
            x = x + self.proj[i](a)
            x = x + self.fc2[i](F.gelu(self.fc1[i](self.ln2[i](x))))
        x = self.ln_f(x)
        return self.head(x[:, 0]), x


VIT = lk.seeded(31, ViT)
IMG = D.randn(4, 3, 16, 16)


@op("vit_block_forward_fp32", bound=2e-4)
def _(c):
    return c.module(VIT).eval()(c(IMG))


@op("vit_block_forward_fp16", dtype=torch.float16, bound=3e-2)
def _(c):
    return c.module(VIT).eval()(c(IMG))


@op("vit_block_forward_bf16", dtype=torch.bfloat16, bound=1.5e-1)
def _(c):
    return c.module(VIT).eval()(c(IMG))


@op("vit_block_backward_fp32", bound=2e-3)
def _(c):
    m = c.module(VIT)
    logits, _ = m(c(IMG))
    F.cross_entropy(logits, c.raw(torch.tensor([1, 3, 5, 7]))).backward()
    return m.qkv[0].weight.grad, m.patch.weight.grad, m.pos.grad


# ---------------------------------------------------------------------------------------------- UNet step
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x))) + self.t(F.silu(temb))[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n, self.qkv, self.proj, self.heads = nn.GroupNorm(8, ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1), heads

    def forward(self, x):
        b, ch, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).view(b, 3, self.heads, ch // self.heads, h * w).transpose(-1, -2).unbind(1)
        a = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, ch, h, w)
        return x + self.proj(a)


class TinyUNet(nn.Module):
    def __init__(self, ch=32, tdim=64):
        super().__init__()
        self.tdim = tdim
        self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.res1, self.down = ResBlock(ch, ch, tdim), nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.res2, self.attn = ResBlock(ch * 2, ch * 2, tdim), AttnBlock(ch * 2)
        self.up = nn.Conv2d(ch * 2, ch, 3, padding=1)
        self.res3 = ResBlock(ch * 2, ch, tdim)
        self.n_out, self.conv_out = nn.GroupNorm(8, ch), nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x, t):
        te = self.temb(timestep_embedding(t, self.tdim).to(x.dtype))
        h1 = self.res1(self.conv_in(x), te)
        h = self.attn(self.res2(self.down(h1), te))
        h = self.up(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.res3(torch.cat([h, h1], 1), te)
        return self.conv_out(F.silu(self.n_out(h)))


UNET = lk.seeded(32, TinyUNet)
XT = D.randn(2, 3, 16, 16)
NOISE = D.randn(2, 3, 16, 16)
TSTEP = torch.tensor([400, 700])
BETAS = torch.linspace(1e-4, 0.02, 1000)
ALPHAS = 1 - BETAS
ABAR = torch.cumprod(ALPHAS, 0)


def ddpm_step(c, dtype_note=None):
    m = c.module(UNET).eval()
    x = c(XT)
    eps = m(x, c.raw(TSTEP))
    a, ab, beta = (c(v[TSTEP].view(-1, 1, 1, 1).float()) for v in (ALPHAS, ABAR, BETAS))
    mean = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
    return eps, mean + beta.sqrt() * c(NOISE)         # x_{t-1} = mean + sigma z (sigma^2 = beta)


@op("unet_denoise_step_fp32", bound=5e-4)
def _(c):
    with torch.no_grad():
        return ddpm_step(c)


@op("unet_denoise_step_fp16", dtype=torch.float16, bound=5e-2)
def _(c):
    with torch.no_grad():
        return ddpm_step(c)


@op("unet_denoise_step_channels_last_bf16", dtype=torch.bfloat16, bound=2e-1)
def _(c):
    with torch.no_grad():
        m = c.module(UNET).eval().to(memory_format=torch.channels_last)
        x = c(XT).contiguous(memory_format=torch.channels_last)
        return m(x, c.raw(TSTEP)).contiguous()


@op("unet_training_gradients_fp32", bound=5e-3)
def _(c):
    m = c.module(UNET)
    F.mse_loss(m(c(XT), c.raw(TSTEP)), c(NOISE)).backward()
    return m.conv_in.weight.grad, m.attn.qkv.weight.grad, m.temb[0].weight.grad


# ---------------------------------------------------------------------------------------------- DLRM
TABLES, EMB, DENSE = (700, 1200, 300, 2000, 900, 50, 1500, 400), 16, 13


class DLRM(nn.Module):
    """Bottom MLP on dense features, one embedding bag per sparse feature, pairwise dot interaction, top MLP."""

    def __init__(self):
        super().__init__()
        self.bottom = nn.Sequential(nn.Linear(DENSE, 32), nn.ReLU(), nn.Linear(32, EMB), nn.ReLU())
        self.tables = nn.ModuleList(nn.EmbeddingBag(n, EMB, mode="sum") for n in TABLES)
        f = len(TABLES) + 1
        self.register_buffer("tri", torch.tril_indices(f, f, offset=-1))
        self.top = nn.Sequential(nn.Linear(EMB + f * (f - 1) // 2, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, dense, idxs, offs):
        d = self.bottom(dense)
        feats = torch.stack([d] + [t(i, o) for t, i, o in zip(self.tables, idxs, offs)], 1)       # (B, F, EMB)
        z = torch.bmm(feats, feats.transpose(1, 2))
        flat = z[:, self.tri[0], self.tri[1]]
        return torch.sigmoid(self.top(torch.cat([d, flat], 1))).squeeze(1)


DLRM_M = lk.seeded(33, DLRM)
for _t in DLRM_M.tables:
    nn.init.normal_(_t.weight, std=0.5)          # (the default init is N(0,1) and drowns the dense path)
BATCH = 24
DENSE_X = D.rand(BATCH, DENSE) * 3
LENS = [D.randint(1, 6, (BATCH,)) for _ in TABLES]
OFFS = [torch.cat([torch.zeros(1, dtype=torch.long), n.cumsum(0)[:-1]]) for n in LENS]
IDXS = [D.randint(0, rows, (int(n.sum()),)) for rows, n in zip(TABLES, LENS)]
CLICKS = (D.rand(BATCH) < 0.3).float()


@op("dlrm_forward_fp32", bound=2e-4)
def _(c):
    with torch.no_grad():
        return c.module(DLRM_M)(c(DENSE_X), [c.raw(i) for i in IDXS], [c.raw(o) for o in OFFS])


@op("dlrm_forward_fp16", dtype=torch.float16, bound=3e-2)
def _(c):
    with torch.no_grad():
        return c.module(DLRM_M)(c(DENSE_X), [c.raw(i) for i in IDXS], [c.raw(o) for o in OFFS])


@op("dlrm_training_gradients_fp32", bound=2e-3)
def _(c):
    m = c.module(DLRM_M)
    p = m(c(DENSE_X), [c.raw(i) for i in IDXS], [c.raw(o) for o in OFFS])
    F.binary_cross_entropy(p, c(CLICKS)).backward()
    return m.tables[0].weight.grad, m.tables[3].weight.grad, m.bottom[0].weight.grad, m.top[0].weight.grad


def bench(b):
    dev = lk.DEVICE
    dt = torch.float16
    n = b.size(256, 4)
    vit = ViT(dim=b.size(768, 64), heads=b.size(12, 4), depth=b.size(12, 2), patch=16, image=b.size(224, 32)).to(dev, dt).eval()
    img = torch.randn(n, 3, b.size(224, 32), b.size(224, 32), device=dev, dtype=dt)
    with torch.no_grad():
        b.put("vit_base_forward_fp16_images_s", n, 1, lambda: vit(img), iters=3)
    ch = b.size(128, 32)
    unet = TinyUNet(ch=ch).to(dev, dt).eval()
    x = torch.randn(b.size(16, 2), 3, b.size(128, 16), b.size(128, 16), device=dev, dtype=dt)
    t = torch.randint(0, 1000, (x.shape[0],), device=dev)
    with torch.no_grad():
        b.put("unet_denoise_step_fp16_images_s", x.shape[0], 1, lambda: unet(x, t), iters=3)
    bs = b.size(8192, 24)
    m = DLRM().to(dev).eval()
    dense = torch.rand(bs, DENSE, device=dev)
    lens = [torch.randint(1, 6, (bs,), device=dev) for _ in TABLES]
    offs = [torch.cat([torch.zeros(1, dtype=torch.long, device=dev), ln.cumsum(0)[:-1]]) for ln in lens]
    idxs = [torch.randint(0, rows, (int(ln.sum()),), device=dev) for rows, ln in zip(TABLES, lens)]
    with torch.no_grad():
        b.put("dlrm_forward_fp32_samples_s", bs, 1, lambda: m(dense, idxs, offs), iters=5)
