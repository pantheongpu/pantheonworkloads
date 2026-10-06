#!/usr/bin/env python3
"""Architecture-coverage engine: tiny seeded-random models of open architectures, built from
Hugging Face `transformers` config classes. NO weights are downloaded and none are pretrained.

These workloads test an architecture's CODE PATHS and kernels (grouped-query attention, rotary
embeddings, RMSNorm, sliding-window masks, MoE routing, state-space scans, cross attention, ...)
on a CPU, a GPU or a simulated GPU. They say NOTHING about the quality of any pretrained model
whose name the architecture shares (Llama, Mistral, Mixtral, Qwen2, Gemma, Phi-3, GPT-NeoX,
Falcon, Mamba, ViT, CLIP, Whisper, T5). Only transformers' Apache-2.0 modelling code is used.

    arch_cases.py --mode functional --group llama-family|moe|ssm|vision|encdec
    arch_cases.py --mode bench      --arch llama --size small|medium|large     (real GPU only)

Per case: build the model with a seeded generator (weights re-drawn from N(0, 1/fan_in) so the
logits have a real spread), run a forward pass, then greedy decode steps with the KV / state
cache where the architecture has one. Output (per case, in one JSON object):
  ids    exact string: argmax at every prefill position, the greedy tokens, top-5 ids of the last
         position (and, for MoE, the experts each token is routed to) -- compared exactly
  stats  a few float logit statistics -- compared within the manifest's tolerance
Every run fails (it never prints a reference to record) when
  * an exact id could be a near-tie: gap between consecutive top-6 logits (or router scores) < MARGIN
  * cached decoding disagrees with recomputing the whole sequence without a cache
  * the target's result is further from the same model in float64 on the CPU than FP64_TOL
    (ids must be equal, stats within tolerance)
Environment: PW_TARGET (cpu | gpu | sim:...), PW_ARCH_ATTN (sdpa | eager, default sdpa),
PW_ARCH_ONLY (comma list of case names to run), PW_ARCH_SEED_OFFSET (testing only).
Exit 77 + a reason when torch, transformers or a GPU is missing.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

SKIP = 77
SEED = 20260506
MARGIN = 1e-3          # fp32 reordering noise is ~1e-6 on these sizes; 1000x headroom
FP64_TOL = 2e-3        # target vs float64 CPU: |diff| <= FP64_TOL * (1 + rms of the logits)
CACHE_TOL = 1e-3       # cached decode vs full recompute, same scaling
DECODE_STEPS = 6


def skip(why):
    print(f"SKIP: {why}")
    sys.exit(SKIP)


# --------------------------------------------------------------------------- case definitions
# Each case: family (how the driver feeds it), config class + kwargs, notes on the code path.
# vocab is small (512) and the dims tiny so a CPU-executed simulator finishes in seconds.
def cases(group):
    V = 512
    d = {}
    if group == "llama-family":
        # Llama: GQA 4:1 (8 heads, 2 kv heads), head_dim 32, RoPE, RMSNorm, SwiGLU with a non-power-of-two width
        d["llama-gqa4"] = ("causal", "llama", "LlamaConfig", dict(
            vocab_size=V, hidden_size=256, intermediate_size=704, num_hidden_layers=2,
            num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=128,
            tie_word_embeddings=False), "GQA 4:1, RoPE, RMSNorm, SwiGLU, untied head")
        # Llama with head_dim 128 and multi-query attention (1 kv head)
        d["llama-mqa-hd128"] = ("causal", "llama", "LlamaConfig", dict(
            vocab_size=V, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=128, max_position_embeddings=128,
            tie_word_embeddings=True), "MQA, head_dim 128, tied embeddings")
        # Mistral: sliding window of 5 is shorter than the 11-token prompt, so the window mask
        # (and the cache's rolling window during decoding) decides the result
        d["mistral-swa"] = ("causal", "mistral", "MistralConfig", dict(
            vocab_size=V, hidden_size=256, intermediate_size=704, num_hidden_layers=2,
            num_attention_heads=8, num_key_value_heads=2, sliding_window=5, max_position_embeddings=128),
            "sliding-window attention (window 5 < prompt 11), GQA 4:1")
        # Qwen2: biases on q/k/v only, GQA 3:1, head_dim 64, tied embeddings
        d["qwen2-gqa3"] = ("causal", "qwen2", "Qwen2Config", dict(
            vocab_size=V, hidden_size=384, intermediate_size=1024, num_hidden_layers=2,
            num_attention_heads=6, num_key_value_heads=2, max_position_embeddings=128,
            tie_word_embeddings=True), "q/k/v bias, GQA 3:1, head_dim 64, tied embeddings")
        # Gemma: head_dim independent of hidden/heads (64 vs 128/2... here 4 heads x 64 on 192 wide),
        # (1 + w) RMSNorm, GeGLU tanh-approx, scaled embeddings, MQA
        d["gemma-mqa"] = ("causal", "gemma", "GemmaConfig", dict(
            vocab_size=V, hidden_size=192, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=1, head_dim=64, max_position_embeddings=128,
            hidden_activation="gelu_pytorch_tanh"), "head_dim != hidden/heads, (1+w) RMSNorm, GeGLU, scaled embeddings, MQA")
        # Phi-3: fused qkv_proj and gate_up_proj
        d["phi3-fused"] = ("causal", "phi3", "Phi3Config", dict(
            vocab_size=V, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
            original_max_position_embeddings=128, pad_token_id=0, eos_token_id=2, bos_token_id=1), "fused qkv_proj and gate_up_proj, head_dim 64")
        # GPT-NeoX: parallel attention+MLP residual, LayerNorm, rotary on 25% of head_dim
        d["gpt-neox-parallel"] = ("causal", "gpt_neox", "GPTNeoXConfig", dict(
            vocab_size=V, hidden_size=256, intermediate_size=1024, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=128, rotary_pct=0.25,
            use_parallel_residual=True), "parallel attention + MLP, LayerNorm, partial RoPE (25%)")
        # Falcon-40B style: new decoder architecture, parallel attention, GQA (2 kv heads), two layer norms
        d["falcon-new-gqa"] = ("causal", "falcon", "FalconConfig", dict(
            vocab_size=V, hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
            num_kv_heads=2, new_decoder_architecture=True, parallel_attn=True, alibi=False,
            bias=False, max_position_embeddings=128), "Falcon-40B style: parallel attention, GQA 2:1, RoPE")
        # Falcon-7B style: multi-query attention, parallel attention with one shared layer norm
        d["falcon-mqa"] = ("causal", "falcon", "FalconConfig", dict(
            vocab_size=V, hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
            multi_query=True, new_decoder_architecture=False, parallel_attn=True, alibi=False,
            bias=False, max_position_embeddings=128), "Falcon-7B style: multi-query attention, parallel attention")
    elif group == "moe":
        # Mixtral: 8 experts, top-2 routing, softmax over the selected scores, GQA, no sliding window
        d["mixtral-top2of8"] = ("causal", "mixtral", "MixtralConfig", dict(
            vocab_size=V, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, num_local_experts=8,
            num_experts_per_tok=2, max_position_embeddings=128, sliding_window=None,
            router_jitter_noise=0.0), "top-2 of 8 routing, expert gather/scatter, GQA 2:1")
        # Odd expert count, top-3: not a power of two, and k > 2
        d["mixtral-top3of5"] = ("causal", "mixtral", "MixtralConfig", dict(
            vocab_size=V, hidden_size=128, intermediate_size=192, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, num_local_experts=5,
            num_experts_per_tok=3, max_position_embeddings=128, sliding_window=None,
            router_jitter_noise=0.0), "top-3 of 5 routing (odd expert count), MQA")
    elif group == "ssm":
        d["mamba1"] = ("mamba", "mamba", "MambaConfig", dict(
            vocab_size=V, hidden_size=128, state_size=16, num_hidden_layers=2, expand=2,
            conv_kernel=4, time_step_rank=8), "selective scan (S6), causal depthwise conv, recurrent state cache")
        d["mamba2"] = ("mamba", "mamba2", "Mamba2Config", dict(
            vocab_size=V, hidden_size=128, state_size=16, num_hidden_layers=2, expand=2,
            conv_kernel=4, num_heads=8, head_dim=32, n_groups=2, chunk_size=8),
            "Mamba-2 SSD (chunked, chunk 8 < sequence 11), grouped B/C, recurrent state cache")
    elif group == "vision":
        d["vit-p8"] = ("vit", "vit", "ViTConfig", dict(
            image_size=32, patch_size=8, num_channels=3, hidden_size=128, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=256, num_labels=10),
            "patch embedding conv, class token, learned positions, pre-LN encoder, 17 tokens (odd)")
        d["clip-pair"] = ("clip", "clip", "CLIPConfig", dict(), "text tower (causal mask, EOS pooling) + vision tower, contrastive logits")
    elif group == "encdec":
        d["whisper-style"] = ("whisper", "whisper", "WhisperConfig", dict(
            vocab_size=V, num_mel_bins=16, d_model=128, encoder_layers=2, decoder_layers=2,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=256,
            decoder_ffn_dim=256, max_source_positions=51, max_target_positions=64,
            decoder_start_token_id=1, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            suppress_tokens=[], begin_suppress_tokens=[]),
            "2x conv1d stem, sinusoidal encoder positions, decoder self + cross attention with cache, 51 frames (odd)")
        d["t5-relu"] = ("t5", "t5", "T5Config", dict(
            vocab_size=V, d_model=128, d_kv=64, d_ff=256, num_layers=2, num_decoder_layers=2,
            num_heads=2, relative_attention_num_buckets=16, relative_attention_max_distance=64,
            feed_forward_proj="relu", decoder_start_token_id=0, pad_token_id=0, eos_token_id=1),
            "T5 v1.0: relative position bias, RMS-style T5LayerNorm, ReLU FFN, tied embeddings, padded encoder input")
        d["t5-gated-gelu"] = ("t5", "t5", "T5Config", dict(
            vocab_size=V, d_model=128, d_kv=64, d_ff=256, num_layers=2, num_decoder_layers=2,
            num_heads=2, relative_attention_num_buckets=16, relative_attention_max_distance=64,
            feed_forward_proj="gated-gelu", tie_word_embeddings=False, decoder_start_token_id=0,
            pad_token_id=0, eos_token_id=1), "T5 v1.1: gated GELU FFN, untied output head")
    else:
        raise SystemExit(f"unknown group {group}")
    only = os.environ.get("PW_ARCH_ONLY")
    if only:
        d = {k: v for k, v in d.items() if k in only.split(",")}
    return d


GROUPS = ["llama-family", "moe", "ssm", "vision", "encdec"]


# --------------------------------------------------------------------------- model construction
def make_config(transformers, cls, kwargs, kind):
    if kind == "clip":
        text = transformers.CLIPTextConfig(
            vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=2, max_position_embeddings=16, projection_dim=64,
            eos_token_id=511, bos_token_id=510, pad_token_id=0)
        vis = transformers.CLIPVisionConfig(
            hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
            image_size=32, patch_size=8, projection_dim=64)
        return transformers.CLIPConfig(text_config=text.to_dict(), vision_config=vis.to_dict(), projection_dim=64)
    return getattr(transformers, cls)(**kwargs)


def reinit(model, seed, torch):
    """Redraw weights from a seeded CPU generator: matrices N(0, 1/fan_in) (embeddings the same with fan_in = width),
    norm scales 1 + 0.1 N, other vectors 0.02 N. Keeps what the architecture sets up itself
    (Mamba's A_log, D, dt bias; non-trainable positional tables)."""
    g = torch.Generator().manual_seed(seed)
    keep = ("A_log", "D", "dt_bias", "dt_proj.bias", "relative_attention_bias_unused")
    emb_types = (torch.nn.Embedding,)
    norm_ids = {id(m.weight) for m in model.modules() if "norm" in type(m).__name__.lower() and getattr(m, "weight", None) is not None}
    emb_ids = {id(m.weight) for m in model.modules() if isinstance(m, emb_types)}
    for name, p in model.named_parameters():
        if not p.requires_grad or name.split(".")[-1] in keep or name.endswith(tuple(keep)):
            continue
        with torch.no_grad():
            if p.dim() == 0:
                continue
            if p.dim() == 1:
                if id(p) in norm_ids or ("norm" in name.lower() and name.endswith("weight")):
                    v = 1.0 + 0.1 * torch.randn(p.shape, generator=g)
                else:
                    v = 0.02 * torch.randn(p.shape, generator=g)
            elif "relative_attention_bias" in name:
                v = torch.randn(p.shape, generator=g)
            elif id(p) in emb_ids or "position" in name or "shared" in name:
                v = torch.randn(p.shape, generator=g) / math.sqrt(p.shape[-1])   # small, so tied heads do not just echo the input token
            else:
                fan_in = p.numel() // p.shape[0] if ("conv" in name and p.dim() == 3) else p.shape[-1]
                v = torch.randn(p.shape, generator=g) / math.sqrt(fan_in)
            p.copy_(v.to(p.dtype))
    return model


def build(transformers, torch, spec, attn, seed):
    kind, _, cls, kwargs, _ = spec
    cfg = make_config(transformers, cls, kwargs, kind)
    used = attn
    model = None
    for impl in ([attn, "eager"] if attn != "eager" else ["eager"]):
        try:
            torch.manual_seed(seed)
            c = copy.deepcopy(cfg)
            c._attn_implementation = impl
            model = _model_class(transformers, kind, cls)(c) if kind != "clip" else transformers.CLIPModel(c)
            used = impl
            break
        except (ValueError, NotImplementedError, AttributeError, KeyError):
            if impl == "eager":
                raise
    model.eval()
    reinit(model, seed, torch)
    return model, used


def _model_class(transformers, kind, cls):
    base = cls[:-len("Config")]
    if kind == "causal" or kind == "mamba":
        return getattr(transformers, base + "ForCausalLM")
    if kind == "vit":
        return transformers.ViTForImageClassification
    if kind in ("whisper", "t5"):
        return getattr(transformers, base + "ForConditionalGeneration")
    raise KeyError(kind)


# --------------------------------------------------------------------------- numeric helpers
class Margins:
    """Tracks the smallest gap between the winner and the runner-up wherever an exact id is recorded."""
    def __init__(self):
        self.min = float("inf")
        self.where = ""

    def top(self, logits, label, depth=1):
        """logits (..., V): gaps between consecutive ranks 1..depth+1."""
        s = logits.double().flatten(0, -2).topk(depth + 1, dim=-1).values
        gap = (s[:, :-1] - s[:, 1:]).min().item()
        if gap < self.min:
            self.min, self.where = gap, label


def ids_str(t):
    return " ".join(str(int(x)) for x in t.flatten().tolist())


def r6(x):
    return round(float(x), 6)


def logit_stats(prefill_logits, step_top1):
    """[mean, rms, mean of last-position top-1 logit over batch, std of last-position logits (row 0),
    then the top-1 logit of row 0 at each decode step]"""
    last = prefill_logits[:, -1].double()
    out = [prefill_logits.double().mean().item(), prefill_logits.double().pow(2).mean().sqrt().item(),
           last.max(-1).values.mean().item(), last[0].std().item()]
    return [r6(v) for v in out + step_top1]


# --------------------------------------------------------------------------- drivers per family
def gen(torch, seed, *shape, vocab=None, normal=False):
    g = torch.Generator().manual_seed(seed)
    if vocab:
        return torch.randint(3, vocab, shape, generator=g)
    return torch.randn(*shape, generator=g)


def run_causal(model, cfg, dev, dt, torch, seed, mg, extra):
    B, T = 2, 11                                  # odd prompt length, batch of 2
    ids = gen(torch, seed, B, T, vocab=cfg.vocab_size).to(dev)
    is_moe = hasattr(cfg, "num_local_experts")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, **({"output_router_logits": True} if is_moe else {}))
        prefill = out.logits
        mg.top(prefill, "prefill positions", 1)
        mg.top(prefill[:, -1], "prefill top-5", 5)
        routed = ""
        if is_moe:
            routed, rg = routing(out.router_logits, cfg.num_experts_per_tok, torch)
            mg.min = min(mg.min, rg)
            mg.where = mg.where if mg.min != rg else "router scores"
        pkv, last = out.past_key_values, prefill[:, -1]
        toks, step_logits, step_top1 = [], [], []
        for s in range(DECODE_STEPS):
            mg.top(last, f"decode step {s}", 1)
            nxt = last.argmax(-1)
            toks.append(nxt)
            step_top1.append(last[0].max().item())
            o = model(input_ids=nxt[:, None], past_key_values=pkv, use_cache=True)
            pkv, last = o.past_key_values, o.logits[:, -1]
            step_logits.append(o.logits[:, -1])
        gen_ids = torch.stack(toks, 1)
        # cache check: recompute everything without a cache
        full = model(input_ids=torch.cat([ids, gen_ids], 1), use_cache=False).logits
        ref = full[:, T:T + DECODE_STEPS - 1]
        got = torch.stack(step_logits[:-1], 1)
        cache_err = (ref - got).abs().max().item() if DECODE_STEPS > 1 else 0.0
        cache_err = max(cache_err, (full[:, :T] - prefill).abs().max().item())
    top5 = prefill[:, -1].topk(5, dim=-1).indices
    s = {"ids": f"prefill {ids_str(prefill.argmax(-1))} | greedy {ids_str(gen_ids)} | top5 {ids_str(top5)}"
               + (f" | routed {routed}" if routed else ""),
         "stats": logit_stats(prefill, step_top1)}
    return s, prefill, cache_err


def routing(router_logits, k, torch):
    """Experts chosen per token and layer (exact), and the smallest gap between the k-th and (k+1)-th score."""
    parts, gap = [], float("inf")
    for layer in router_logits:
        sc = layer.double()
        v = sc.topk(min(k + 1, sc.shape[-1]), dim=-1)
        if sc.shape[-1] > k:
            gap = min(gap, (v.values[:, k - 1] - v.values[:, k]).min().item())
        # order inside the chosen set does not matter: sort the ids
        parts.append(",".join(ids_str(r.sort().values).replace(" ", "") for r in v.indices[:, :k]))
    return " / ".join(parts), gap


def run_mamba(model, cfg, dev, dt, torch, seed, mg, extra):
    B, T = 2, 11
    ids = gen(torch, seed, B, T, vocab=cfg.vocab_size).to(dev)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        prefill = out.logits
        mg.top(prefill, "prefill positions", 1)
        mg.top(prefill[:, -1], "prefill top-5", 5)
        cache, last = out.cache_params, prefill[:, -1]
        toks, step_logits, step_top1 = [], [], []
        for s in range(DECODE_STEPS):
            mg.top(last, f"decode step {s}", 1)
            nxt = last.argmax(-1)
            toks.append(nxt)
            step_top1.append(last[0].max().item())
            o = model(input_ids=nxt[:, None], cache_params=cache, use_cache=True,
                      cache_position=torch.tensor([T + s], device=dev))
            cache, last = o.cache_params, o.logits[:, -1]
            step_logits.append(last)
        gen_ids = torch.stack(toks, 1)
        full = model(input_ids=torch.cat([ids, gen_ids], 1), use_cache=False).logits
        got = torch.stack(step_logits[:-1], 1)
        cache_err = max((full[:, T:T + DECODE_STEPS - 1] - got).abs().max().item(),
                        (full[:, :T] - prefill).abs().max().item())
    top5 = prefill[:, -1].topk(5, dim=-1).indices
    return ({"ids": f"prefill {ids_str(prefill.argmax(-1))} | greedy {ids_str(gen_ids)} | top5 {ids_str(top5)}",
             "stats": logit_stats(prefill, step_top1)}, prefill, cache_err)


def run_vit(model, cfg, dev, dt, torch, seed, mg, extra):
    x = gen(torch, seed, 3, cfg.num_channels, cfg.image_size, cfg.image_size).to(dev, model.dtype)
    with torch.no_grad():
        out = model(pixel_values=x, output_hidden_states=True)
    lg = out.logits
    mg.top(lg, "class logits", 4)
    h = out.hidden_states[-1].double()
    top5 = lg.topk(5, dim=-1).indices
    st = [lg.double().mean().item(), lg.double().pow(2).mean().sqrt().item(), lg.max(-1).values.mean().item(),
          h.mean().item(), h.pow(2).mean().sqrt().item(), float(h.shape[1])]
    return {"ids": f"top5 {ids_str(top5)}", "stats": [r6(v) for v in st]}, lg, 0.0


def run_clip(model, cfg, dev, dt, torch, seed, mg, extra):
    B, T = 3, 7
    ids = gen(torch, seed, 4, T, vocab=500).to(dev)            # 4 texts of 7 tokens, EOS (511) at varying places
    mask = torch.ones_like(ids)
    for i, n in enumerate([7, 5, 4, 6]):
        ids[i, n - 1] = 511
        ids[i, n:] = 0
        mask[i, n:] = 0
    x = gen(torch, seed + 1, B, 3, 32, 32).to(dev, model.dtype)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, pixel_values=x)
    li, lt = out.logits_per_image, out.logits_per_text
    mg.top(li, "image->text", 2)
    mg.top(lt, "text->image", 2)
    st = [li.double().mean().item(), li.double().pow(2).mean().sqrt().item(), out.text_embeds.double().abs().mean().item(),
          out.image_embeds.double().abs().mean().item()]
    ids_s = f"img2txt {ids_str(li.argmax(-1))} | txt2img {ids_str(lt.argmax(-1))} | rank {ids_str(li.argsort(-1, descending=True))}"
    return {"ids": ids_s, "stats": [r6(v) for v in st] + [r6(v) for v in li[0].tolist()]}, li, 0.0


def run_whisper(model, cfg, dev, dt, torch, seed, mg, extra):
    B = 2
    feats = gen(torch, seed, B, cfg.num_mel_bins, cfg.max_source_positions * 2).to(dev, model.dtype)
    start = torch.full((B, 1), cfg.decoder_start_token_id, device=dev)
    with torch.no_grad():
        out = model(input_features=feats, decoder_input_ids=start, use_cache=True)
        enc = out.encoder_last_hidden_state
        # prefill a 3-token forced prefix, so the decoder has more than one position to attend over
        prefix = torch.cat([start, gen(torch, seed + 2, B, 2, vocab=cfg.vocab_size).to(dev)], 1)
        out = model(encoder_outputs=(enc,), decoder_input_ids=prefix, use_cache=True)
        prefill, pkv = out.logits, out.past_key_values
        mg.top(prefill, "prefill positions", 1)
        mg.top(prefill[:, -1], "prefill top-5", 5)
        last, toks, step_logits, step_top1 = prefill[:, -1], [], [], []
        for s in range(DECODE_STEPS):
            mg.top(last, f"decode step {s}", 1)
            nxt = last.argmax(-1)
            toks.append(nxt)
            step_top1.append(last[0].max().item())
            o = model(encoder_outputs=(enc,), decoder_input_ids=nxt[:, None], past_key_values=pkv, use_cache=True)
            pkv, last = o.past_key_values, o.logits[:, -1]
            step_logits.append(last)
        gen_ids = torch.stack(toks, 1)
        full = model(encoder_outputs=(enc,), decoder_input_ids=torch.cat([prefix, gen_ids], 1), use_cache=False).logits
        P = prefix.shape[1]
        got = torch.stack(step_logits[:-1], 1)
        cache_err = max((full[:, P:P + DECODE_STEPS - 1] - got).abs().max().item(), (full[:, :P] - prefill).abs().max().item())
    top5 = prefill[:, -1].topk(5, dim=-1).indices
    st = logit_stats(prefill, step_top1) + [r6(enc.double().pow(2).mean().sqrt().item())]
    return {"ids": f"prefill {ids_str(prefill.argmax(-1))} | greedy {ids_str(gen_ids)} | top5 {ids_str(top5)}", "stats": st}, prefill, cache_err


def run_t5(model, cfg, dev, dt, torch, seed, mg, extra):
    B, T = 2, 9
    ids = gen(torch, seed, B, T, vocab=cfg.vocab_size).to(dev)
    mask = torch.ones_like(ids)
    ids[1, 6:] = cfg.pad_token_id                  # second row is 6 tokens long: padded encoder input
    mask[1, 6:] = 0
    start = torch.full((B, 1), cfg.decoder_start_token_id, device=dev)
    with torch.no_grad():
        enc = model.get_encoder()(input_ids=ids, attention_mask=mask).last_hidden_state
        prefix = torch.cat([start, gen(torch, seed + 2, B, 2, vocab=cfg.vocab_size).to(dev)], 1)
        out = model(encoder_outputs=(enc,), attention_mask=mask, decoder_input_ids=prefix, use_cache=True)
        prefill, pkv = out.logits, out.past_key_values
        mg.top(prefill, "prefill positions", 1)
        mg.top(prefill[:, -1], "prefill top-5", 5)
        last, toks, step_logits, step_top1 = prefill[:, -1], [], [], []
        for s in range(DECODE_STEPS):
            mg.top(last, f"decode step {s}", 1)
            nxt = last.argmax(-1)
            toks.append(nxt)
            step_top1.append(last[0].max().item())
            o = model(encoder_outputs=(enc,), attention_mask=mask, decoder_input_ids=nxt[:, None],
                      past_key_values=pkv, use_cache=True)
            pkv, last = o.past_key_values, o.logits[:, -1]
            step_logits.append(last)
        gen_ids = torch.stack(toks, 1)
        full = model(encoder_outputs=(enc,), attention_mask=mask,
                     decoder_input_ids=torch.cat([prefix, gen_ids], 1), use_cache=False).logits
        P = prefix.shape[1]
        got = torch.stack(step_logits[:-1], 1)
        cache_err = max((full[:, P:P + DECODE_STEPS - 1] - got).abs().max().item(), (full[:, :P] - prefill).abs().max().item())
    top5 = prefill[:, -1].topk(5, dim=-1).indices
    st = logit_stats(prefill, step_top1) + [r6(enc[0].double().pow(2).mean().sqrt().item())]
    return {"ids": f"prefill {ids_str(prefill.argmax(-1))} | greedy {ids_str(gen_ids)} | top5 {ids_str(top5)}", "stats": st}, prefill, cache_err


DRIVERS = {"causal": run_causal, "mamba": run_mamba, "vit": run_vit, "clip": run_clip,
           "whisper": run_whisper, "t5": run_t5}


# --------------------------------------------------------------------------- functional mode
def device_for(target, torch):
    if target == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        skip(f"target {target}: torch sees no GPU")
    return "cuda"


SEARCH_MARGIN = 3e-3   # a case's seed is the first salt whose float64 CPU run has every exact-id gap above this


def run_case(name, spec, transformers, torch, dev, attn, offset):
    kind = spec[0]
    # Random weights make near-ties likely somewhere in ~70 recorded ids. Find, on the float64 CPU twin
    # (deterministic to ~1e-12 on any machine), the first salt whose smallest gap is well above MARGIN;
    # the target then has to reproduce those ids. The salt is printed in the detail.
    for salt in range(128):
        seed = SEED + offset + 1000 * salt
        model, used = build(transformers, torch, spec, attn, seed)
        ref_model = copy.deepcopy(model).double()        # float64 CPU twin: same weights, no reordering noise
        if hasattr(ref_model, "set_experts_implementation") and hasattr(model.config, "num_local_experts"):
            ref_model.set_experts_implementation("eager")      # grouped_mm has no float64 kernel
        mg64 = Margins()
        res64, logits64, _ = DRIVERS[kind](ref_model, ref_model.config, "cpu", None, torch, seed, mg64, {})
        if mg64.min >= SEARCH_MARGIN:
            break
    else:
        raise SystemExit(f"{name}: no seed with every id gap >= {SEARCH_MARGIN:g} in 128 tries")
    cfg = model.config
    experts = None
    if hasattr(ref_model, "set_experts_implementation") and hasattr(cfg, "num_local_experts"):
        experts = os.environ.get("PW_ARCH_EXPERTS")        # target: transformers' default (grouped_mm where it can dispatch)
        if experts:
            model.set_experts_implementation(experts)
        experts = model.config._experts_implementation
    model = model.to(dev)
    mg = Margins()
    t0 = time.perf_counter()
    res, logits, cache_err = DRIVERS[kind](model, cfg, dev, None, torch, seed, mg, {})
    seconds = time.perf_counter() - t0
    scale = 1.0 + logits.double().pow(2).mean().sqrt().item()
    errs = {}
    errs["cache"] = cache_err / scale
    errs["vs_fp64_logits"] = (logits.double().cpu() - logits64.double()).abs().max().item() / scale
    errs["vs_fp64_stats"] = max(abs(a - b) for a, b in zip(res["stats"], res64["stats"])) if len(res["stats"]) == len(res64["stats"]) else float("inf")
    problems = []
    if not all(math.isfinite(v) for v in res["stats"]):
        problems.append(f"non-finite statistics {res['stats']}")
    if mg.min < MARGIN:
        problems.append(f"near-tie in an exact id ({mg.where}: gap {mg.min:.2e} < {MARGIN:g}); change the case's seed")
    if errs["cache"] > CACHE_TOL:
        problems.append(f"cached decoding differs from full recompute by {errs['cache']:.2e} (relative) > {CACHE_TOL:g}")
    if res["ids"] != res64["ids"]:
        problems.append(f"ids differ from the float64 CPU run: {res['ids']!r} vs {res64['ids']!r}")
    if errs["vs_fp64_logits"] > FP64_TOL:
        problems.append(f"logits differ from float64 CPU by {errs['vs_fp64_logits']:.2e} (relative) > {FP64_TOL:g}")
    return res, problems, {"experts": experts, "salt": salt, "attn": used, "margin": mg.min, "errs": errs, "seconds": seconds,
                           "params": sum(p.numel() for p in model.parameters())}


def functional(group):
    try:
        import torch
    except ImportError as e:
        skip(f"torch is not installed ({e})")
    try:
        import transformers
    except ImportError as e:
        skip(f"transformers is not installed ({e}); tools/torch-cpu-env.sh makes a Python with both")
    target = os.environ.get("PW_TARGET", "cpu")
    dev = device_for(target, torch)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    attn = os.environ.get("PW_ARCH_ATTN", "sdpa")
    offset = int(os.environ.get("PW_ARCH_SEED_OFFSET", "0"))
    output, notes, problems = {}, [], []
    for name, spec in cases(group).items():
        res, probs, info = run_case(name, spec, transformers, torch, dev, attn, offset)
        output[name] = res
        problems += [f"{name}: {p}" for p in probs]
        e = info["errs"]
        notes.append(f"{name} {info['params'] / 1e6:.2f}M salt={info['salt']} attn={info['attn']}" + (f" experts={info['experts']}" if info["experts"] else "") + f" margin={info['margin']:.1e} "
                     f"fp64logits={e['vs_fp64_logits']:.1e} fp64stats={e['vs_fp64_stats']:.1e} cache={e['cache']:.1e}")
        print(notes[-1], file=sys.stderr)
    if problems:
        sys.exit("; ".join(problems))
    where = dev if dev == "cpu" else torch.cuda.get_device_name(0)
    print(json.dumps({"output": output, "metrics": {},
                      "detail": f"{group}: {len(output)} random-weight architectures on {where}, torch {torch.__version__}, "
                                f"transformers {transformers.__version__}: " + "; ".join(notes)}))


# --------------------------------------------------------------------------- benchmark mode
# Real-GPU throughput of the same architectures at a larger size. Random weights, so only the speed
# (and finiteness) means anything. PW_ARCH_BENCH_SIZE: small | medium | large.
BENCH_SIZES = {
    #          layers hidden heads kv  inter   batch prompt  new tokens
    "small":  (4,  512,  8,  2,  1408, 8,  256, 64),
    "medium": (16, 2048, 16, 4,  5632, 8,  512, 128),
    "large":  (32, 4096, 32, 8, 14336, 4, 1024, 128),     # 7B-class shape, ~6.5 B parameters
}
BENCH_ARCHS = {"llama": "LlamaConfig", "mistral": "MistralConfig", "qwen2": "Qwen2Config",
               "gemma": "GemmaConfig", "phi3": "Phi3Config", "mixtral": "MixtralConfig"}


def bench(arch, size):
    try:
        import torch
        import transformers
    except ImportError as e:
        skip(f"torch or transformers is missing ({e})")
    target = os.environ.get("PW_TARGET", "cpu")
    if target.startswith("sim:"):
        skip("benchmark numbers are for real targets only")
    if arch not in BENCH_ARCHS:
        sys.exit(f"PW_ARCH_BENCH_ARCH {arch!r}: one of {sorted(BENCH_ARCHS)}")
    if size not in BENCH_SIZES:
        sys.exit(f"PW_ARCH_BENCH_SIZE {size!r}: one of {sorted(BENCH_SIZES)}")
    dev = device_for(target, torch)
    L, H, nh, nkv, inter, B, P, N = BENCH_SIZES[size]
    if dev == "cpu":                        # cpu only tests the workload
        B, P, N = 1, min(P, 32), 8
    kw = dict(vocab_size=32000, hidden_size=H, intermediate_size=inter, num_hidden_layers=L,
              num_attention_heads=nh, num_key_value_heads=nkv, max_position_embeddings=P + N + 8)
    if arch == "mixtral":
        kw.update(num_local_experts=8, num_experts_per_tok=2, intermediate_size=inter // 2, sliding_window=None)
    if arch == "mistral":
        kw.update(sliding_window=max(P // 2, 16))
    if arch == "gemma":
        kw.update(head_dim=H // nh, hidden_activation="gelu_pytorch_tanh")
    if arch == "phi3":
        kw.update(num_key_value_heads=nh, original_max_position_embeddings=P + N + 8, pad_token_id=0, eos_token_id=2, bos_token_id=1)
    cfg = getattr(transformers, BENCH_ARCHS[arch])(**kw)
    cfg._attn_implementation = os.environ.get("PW_ARCH_ATTN", "sdpa")
    dtype = torch.float32
    if dev == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    cls = getattr(transformers, BENCH_ARCHS[arch][:-6] + "ForCausalLM")
    torch.manual_seed(SEED)
    with torch.device(dev):
        model = cls(cfg).to(dtype).eval()
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    ids = torch.randint(3, 32000, (B, P), device=dev)
    reps = int(os.environ.get("PW_ARCH_BENCH_REPS", "3"))

    def once():
        with torch.no_grad():
            sync(); t0 = time.perf_counter()
            out = model(input_ids=ids, use_cache=True)
            sync(); t1 = time.perf_counter()
            pkv, last = out.past_key_values, out.logits[:, -1]
            for _ in range(N):
                o = model(input_ids=last.argmax(-1, keepdim=True), past_key_values=pkv, use_cache=True)
                pkv, last = o.past_key_values, o.logits[:, -1]
            sync(); t2 = time.perf_counter()
        if not torch.isfinite(last.float()).all():
            sys.exit("non-finite logits: a speed means nothing")
        return t1 - t0, t2 - t1

    once()                                  # warm-up
    runs = [once() for _ in range(reps)]
    pre = sorted(r[0] for r in runs)[len(runs) // 2]
    dec = sorted(r[1] for r in runs)[len(runs) // 2]
    m = {f"{arch}_{size}_prefill_tokens_per_s": round(B * P / pre, 1),
         f"{arch}_{size}_decode_tokens_per_s": round(B * N / dec, 1)}
    params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"output": f"{arch} {size} batch {B} prompt {P} new {N} {str(dtype).split('.')[-1]}",
                      "detail": f"{arch}-style {params / 1e9:.2f}B random-weight parameters, {L}L/{H}H, batch {B}, prompt {P}, "
                                f"{N} greedy tokens with KV cache, median of {reps}, torch {torch.__version__}, "
                                f"transformers {transformers.__version__}",
                      "metrics": m}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["functional", "bench"], required=True)
    ap.add_argument("--group", choices=GROUPS)
    ap.add_argument("--arch", default=os.environ.get("PW_ARCH_BENCH_ARCH", "llama"))
    ap.add_argument("--size", default=os.environ.get("PW_ARCH_BENCH_SIZE", "small"))
    a = ap.parse_args(argv)
    if a.mode == "functional":
        if not a.group:
            ap.error("--group is required")
        functional(a.group)
    else:
        bench(a.arch, a.size)


if __name__ == "__main__":
    main()
