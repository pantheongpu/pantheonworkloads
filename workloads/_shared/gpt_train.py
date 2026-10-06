#!/usr/bin/env python3
"""A few AdamW steps on a tiny character-level GPT: the training-step workloads' engine.

Not a copy of nanoGPT, but the same design (github.com/karpathy/nanoGPT, MIT, read at commit
3adf61e154c3fe3fca428ad6bc3818b27a3b8291): pre-LayerNorm blocks, causal self-attention through
torch's scaled_dot_product_attention, GELU MLP, weight-tied embedding and head, AdamW with
decoupled weight decay on matrices only. It exercises what inference does not: backward kernels
(matmul, attention, layer norm, embedding, cross entropy), a real optimizer, and, with
--dtype bf16, torch autocast (bf16 compute, fp32 master weights).

Everything is deterministic given the seed: the model is initialised and the batches are drawn on
the CPU, whatever the device, so every target starts from the same weights and sees the same
data. The text is checked in (data/tinytext.txt); nothing is downloaded.

    gpt_train.py --mode functional --dtype fp32|bf16          losses of a few steps
    gpt_train.py --mode bench      --dtype fp32,bf16          steps/s and tokens/s

Environment (set by the workloads' run.sh): PW_TARGET (cpu | gpu | sim:...).
Prints one JSON object on its last stdout line (docs/workload-contract.md); exits 77 and says why
when the target cannot run the request (no GPU, or no bf16 on this GPU).
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

SKIP = 77
HERE = pathlib.Path(__file__).resolve().parent
PRESETS = {
    # functional: small enough that a CPU-hosted simulator finishes in minutes
    "tiny": dict(n_layer=2, n_head=2, n_embd=32, block_size=32, batch_size=8, steps=20, lr=3e-3),
    # bench: nanoGPT's "baby GPT" for shakespeare-char (6 layers, 6 heads, 384 wide, 10.7M parameters
    # with a 65-char vocabulary), big enough to keep a GPU busy
    "baby": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=32, steps=30, lr=3e-4),
}
SEED = 1337


def skip(why):
    print(f"SKIP: {why}")
    sys.exit(SKIP)


def build(cfg, vocab, torch):
    nn, F = torch.nn, torch.nn.functional

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            e = cfg["n_embd"]
            self.ln1, self.ln2 = nn.LayerNorm(e), nn.LayerNorm(e)
            self.qkv, self.proj = nn.Linear(e, 3 * e), nn.Linear(e, e)
            self.fc, self.out = nn.Linear(e, 4 * e), nn.Linear(4 * e, e)

        def forward(self, x):
            b, t, e = x.shape
            h = cfg["n_head"]
            q, k, v = self.qkv(self.ln1(x)).split(e, dim=2)
            q, k, v = (z.view(b, t, h, e // h).transpose(1, 2) for z in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(y.transpose(1, 2).reshape(b, t, e))
            return x + self.out(F.gelu(self.fc(self.ln2(x))))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            e = cfg["n_embd"]
            self.wte, self.wpe = nn.Embedding(vocab, e), nn.Embedding(cfg["block_size"], e)
            self.blocks = nn.ModuleList(Block() for _ in range(cfg["n_layer"]))
            self.ln_f = nn.LayerNorm(e)
            self.head = nn.Linear(e, vocab, bias=False)
            self.head.weight = self.wte.weight          # weight tying, as nanoGPT
            self.apply(self._init)
            for n, p in self.named_parameters():        # GPT-2's scaled init of the residual projections
                if n.endswith("proj.weight") or n.endswith("out.weight"):
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

        @staticmethod
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets):
            pos = torch.arange(idx.shape[1], device=idx.device)
            x = self.wte(idx) + self.wpe(pos)
            for blk in self.blocks:
                x = blk(x)
            logits = self.head(self.ln_f(x))
            return F.cross_entropy(logits.view(-1, logits.shape[-1]).float(), targets.view(-1))

    return GPT()


def batches(text, cfg, torch, steps):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    g = torch.Generator().manual_seed(SEED)
    out = []
    for _ in range(steps):
        ix = torch.randint(len(data) - cfg["block_size"] - 1, (cfg["batch_size"],), generator=g)
        x = torch.stack([data[i:i + cfg["block_size"]] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + cfg["block_size"]] for i in ix])
        out.append((x, y))
    return len(chars), out


def device_for(target, torch):
    if target == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        skip(f"target {target}: torch {torch.__version__} sees no GPU (cuda.is_available() is False)")
    return "cuda"


def bf16_ok(device, torch):
    if device == "cpu":
        return True            # autocast runs bf16 on any CPU (slowly where there is no bf16 hardware)
    try:                   # native support only: newer torch otherwise counts slow emulation on pre-Ampere GPUs
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())


def train(dtype, device, cfg, text, torch, steps, timed=False, warmup=0):
    """Run `steps` AdamW steps (after `warmup` untimed ones); return (losses, seconds of the timed ones)."""
    torch.manual_seed(SEED)
    vocab, data = batches(text, cfg, torch, steps + warmup)
    model = build(cfg, vocab, torch).to(device)          # initialised on the CPU, then moved
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95))
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    losses, t0 = [], None
    for i, (x, y) in enumerate(data):
        if i == warmup:
            sync()
            t0 = time.perf_counter()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(dtype == "bf16")):
            loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i >= warmup:
            losses.append(loss.item())                   # .item() syncs, so the timing below is honest
    sync()
    return losses, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["functional", "bench"], required=True)
    ap.add_argument("--dtype", required=True, help="fp32, bf16 or fp32,bf16")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--warmup", type=int, default=0)
    a = ap.parse_args(argv)
    preset = a.preset or ("tiny" if a.mode == "functional" else "baby")
    cfg = dict(PRESETS[preset])
    if a.steps:
        cfg["steps"] = a.steps
    dtypes = a.dtype.split(",")

    try:
        import torch
    except ImportError as e:
        skip(f"torch is not installed ({e})")
    target = os.environ.get("PW_TARGET", "cpu")
    device = device_for(target, torch)
    # Real fp32: no TF32, no reduced-precision reductions behind the user's back.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    text = (HERE / "data" / "tinytext.txt").read_text()
    where = device if device == "cpu" else torch.cuda.get_device_name(0)
    info = f"{preset} GPT on {where}, torch {torch.__version__}"

    if "bf16" in dtypes and not bf16_ok(device, torch):
        if a.mode == "functional":
            skip(f"bf16 is not supported on {where}")
        print(f"note: bf16 is not supported on {where}; only {[d for d in dtypes if d != 'bf16']} is measured", file=sys.stderr)
        dtypes = [d for d in dtypes if d != "bf16"]
        if not dtypes:
            skip(f"bf16 is not supported on {where}")

    n_tokens = cfg["batch_size"] * cfg["block_size"]
    if a.mode == "functional":
        (dtype,) = dtypes
        losses, _ = train(dtype, device, cfg, text, torch, cfg["steps"])
        if not all(math.isfinite(v) for v in losses):
            sys.exit(f"non-finite loss: {losses}")
        if not losses[-1] < losses[0]:
            sys.exit(f"the loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(json.dumps({"output": [round(v, 6) for v in losses],
                          "detail": f"{dtype} {cfg['steps']} steps, {info}, loss {losses[0]:.3f} -> {losses[-1]:.3f}",
                          "metrics": {}}))
        return
    metrics = {}
    for dtype in dtypes:
        losses, seconds = train(dtype, device, cfg, text, torch, cfg["steps"], warmup=max(a.warmup, 1))
        # No learning-rate warm-up, so single steps can spike: judge the end of the run, not its last step.
        tail = losses[-max(1, len(losses) // 4):]
        if not all(math.isfinite(v) for v in losses) or not sum(tail) / len(tail) < losses[0]:
            sys.exit(f"{dtype}: training is broken (loss {losses[0]:.4f} -> {losses[-1]:.4f}); a speed means nothing")
        metrics[f"{dtype}_steps_per_s"] = round(cfg["steps"] / seconds, 3)
        metrics[f"{dtype}_tokens_per_s"] = round(cfg["steps"] * n_tokens / seconds, 1)
    print(json.dumps({"output": f"{'+'.join(dtypes)} {cfg['steps']} steps of {n_tokens} tokens",
                      "detail": info + f", {cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}E batch {cfg['batch_size']}x{cfg['block_size']}",
                      "metrics": metrics}))


if __name__ == "__main__":
    main()
