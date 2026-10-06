# Architecture-coverage workloads (`arch-*`)

**These workloads test the code paths and kernels of model architectures, not the quality of any pretrained
model.** Every model is built from a tiny `transformers` config with **seeded random weights**. No checkpoint is
downloaded or implied, so no model licence applies; "Llama", "Mistral", "Mixtral", "Qwen2", "Gemma", "Phi-3",
"GPT-NeoX", "Falcon", "Mamba", "ViT", "CLIP", "Whisper" and "T5" only name which modelling code and config class is
exercised. The only third-party code used is `transformers` (Apache-2.0) and PyTorch (BSD-3-Clause).

Why: a simulated GPU has to execute the kernels real models use (grouped-query attention, rotary embeddings,
RMSNorm, sliding-window masks, expert gather/scatter, selective scans, cross attention). Small random models hit
those kernels in seconds on a CPU-executed simulator and need no weights, accounts or licence terms.

## Workloads

| Workload | Kind | Targets | What it runs |
| --- | --- | --- | --- |
| `arch-llama-family` | functional | cpu, gpu, sim:nvidia/\*, sim:amd/\* | 9 decoder-only models |
| `arch-moe` | functional | same | 2 Mixtral-style MoE models |
| `arch-ssm` | functional | same | Mamba-1 and Mamba-2 |
| `arch-vision` | functional | same | ViT, CLIP text+vision pair |
| `arch-encdec` | functional | same | Whisper-style, T5 v1.0 and v1.1 style |
| `arch-bench` | benchmark | gpu (cpu only to test it) | prefill and decode tokens/s at a larger size |

## Coverage table

State `ran-on-cpu`: run and recorded (`bin/pw record --target cpu`) here with CPU-only PyTorch 2.13.0 and
transformers 5.18.0 from conda-forge, 2 threads on a shared 4-core machine. `not run`: written, never executed on
that target. **Nothing has run on a GPU or a simulator.**

| Architecture (case) | Code path it exercises | Workload | State |
| --- | --- | --- | --- |
| Llama, GQA 4:1 (`llama-gqa4`) | grouped-query attention (8 heads, 2 kv heads, head_dim 32), RoPE, RMSNorm, SwiGLU with width 704, untied head | `arch-llama-family` | ran-on-cpu |
| Llama, MQA head_dim 128 (`llama-mqa-hd128`) | one kv head, head_dim 128, tied embeddings | `arch-llama-family` | ran-on-cpu |
| Mistral (`mistral-swa`) | sliding-window mask (window 5 < prompt 11) and rolling window in the KV cache | `arch-llama-family` | ran-on-cpu |
| Qwen2 (`qwen2-gqa3`) | q/k/v bias, GQA 3:1, head_dim 64, tied embeddings | `arch-llama-family` | ran-on-cpu |
| Gemma (`gemma-mqa`) | head_dim (64) independent of hidden/heads, (1+w) RMSNorm, GeGLU tanh, scaled embeddings, MQA | `arch-llama-family` | ran-on-cpu |
| Phi-3 (`phi3-fused`) | fused `qkv_proj` and `gate_up_proj` | `arch-llama-family` | ran-on-cpu |
| GPT-NeoX (`gpt-neox-parallel`) | parallel attention + MLP residual, LayerNorm, partial RoPE (25%) | `arch-llama-family` | ran-on-cpu |
| Falcon-40B style (`falcon-new-gqa`) | new decoder architecture, parallel attention, GQA 2:1 | `arch-llama-family` | ran-on-cpu |
| Falcon-7B style (`falcon-mqa`) | multi-query attention, parallel attention, shared layer norm | `arch-llama-family` | ran-on-cpu |
| Mixtral top-2 of 8 (`mixtral-top2of8`) | router, top-k, per-expert gather/scatter (`grouped_mm` expert kernel), GQA 2:1 | `arch-moe` | ran-on-cpu |
| Mixtral top-3 of 5 (`mixtral-top3of5`) | odd expert count, k = 3, MQA | `arch-moe` | ran-on-cpu |
| Mamba (`mamba1`) | selective scan, causal depthwise conv, recurrent conv+ssm state cache | `arch-ssm` | ran-on-cpu |
| Mamba-2 (`mamba2`) | chunked SSD scan (chunk 8 < sequence 11), grouped B/C, state cache | `arch-ssm` | ran-on-cpu |
| ViT (`vit-p8`) | patch-embedding conv, class token, learned positions, 17 tokens (odd) | `arch-vision` | ran-on-cpu |
| CLIP (`clip-pair`) | text tower (causal + padding mask, EOS pooling), vision tower, projections, contrastive logits | `arch-vision` | ran-on-cpu |
| Whisper-style (`whisper-style`) | 2x conv1d stem, 51 frames (odd), sinusoidal positions, decoder self + cross attention caches | `arch-encdec` | ran-on-cpu |
| T5 v1.0 (`t5-relu`) | relative position bias, T5LayerNorm, ReLU FFN, padded encoder input | `arch-encdec` | ran-on-cpu |
| T5 v1.1 (`t5-gated-gelu`) | gated-GELU FFN, untied head | `arch-encdec` | ran-on-cpu |
| Llama/Mistral/Qwen2/Gemma/Phi-3/Mixtral at 0.04 to 6.5 B (`arch-bench`) | prefill and cached decode throughput | `arch-bench` | CPU smoke test of the workload only (numbers meaningless, nothing recorded); not run on a GPU |
| any case on `gpu`, `sim:nvidia/*`, `sim:amd/*` | | all | not run |

Not covered: sliding-window layers mixed with full attention (Gemma 2/3), MLA, rope scaling variants (Llama 3.1
long-context, YaRN), MoE shared experts (Qwen-MoE, DeepSeek), vision-language models, quantised weights, half-precision
functional runs (the functional workloads are fp32 only; `arch-bench` uses bf16/fp16 on a GPU), Mamba with the
`mamba_ssm`/`causal_conv1d` CUDA kernels (transformers falls back to its PyTorch scan when they are absent, which is what
ran here).

## What a case does

1. Build the config (tiny: 0.3 to 3.4 M parameters, vocab 512), construct the model on the CPU under a seed,
   then redraw weights from a seeded CPU generator (matrices N(0, 1/fan_in), embeddings the same with fan_in = width,
   norm scales 1 + 0.1 N, other vectors 0.02 N; Mamba's `A_log`, `D` and `dt` bias keep the architecture's own init).
   transformers' stock init gives logits of ~1e-2, which makes every greedy choice a near-tie.
2. Prefill a batch of 2 prompts of 11 tokens (T5: 9 tokens with the second row padded to 6; Whisper: 51 frames),
   then 6 greedy decode steps with the KV cache (state cache for Mamba, self + cross caches for the encoder-decoders).
3. Output per case: `ids`, an exact string (argmax at every prefill position, greedy ids, top-5 ids of the last position;
   MoE adds the experts each token is routed to per layer; CLIP and ViT give ranks and top-5), and `stats`, about 10
   floats (logit mean, rms, last-position top-1 and spread, top-1 logit at each decode step, an encoder or hidden-state rms).
4. Self-checks that fail the run: (a) any exact id with a gap below 1e-3 to its runner-up (a near-tie would make
   the reference flaky), (b) cached decoding differs from recomputing the whole sequence with no cache by more than 1e-3 of the
   logit scale, (c) the target's ids differ from the same model in float64 on the CPU, or its logits differ by more than 2e-3 (relative).
   Because random weights produce near-ties, each case's seed salt is the first whose float64 CPU run has every gap above
   3e-3; the salt is printed in `detail`.

## Tolerance policy and evidence

`compare: tolerance`, `abs 1e-3, rel 1e-3` for the whole object; strings (`ids`) are compared exactly.
Measured on the CPU (torch 2.13.0), all 18 cases:

| Comparison | Largest difference |
| --- | --- |
| 1 thread vs 4 threads (statistics) | 4e-6 (encdec); ids identical |
| sdpa attention + `grouped_mm` experts vs eager attention + eager experts | within the same 4e-6 |
| float32 vs float64 CPU (statistics, which are printed to 6 decimals) | 1e-5 (Gemma); logits 1e-5 relative at most |
| cached decode vs full recompute | 8e-6 relative |

So 1e-3 is about 100x above reordering noise. A GPU's fp32 kernels (TF32 disabled) may differ more; if they do,
record the evidence in the manifests. A wrong kernel moves logits by 1e-1 or more and changes the ids.
Reference caveat: ids depend on `transformers`' module structure and parameter order (weights are drawn per parameter
in `named_parameters()` order), so a transformers upgrade can change the references; the manifests pin 5.18.0.

## Running

```bash
PW_PYTHON=$(tools/torch-cpu-env.sh) bin/pw run arch-llama-family arch-moe arch-ssm arch-vision arch-encdec --target cpu
# or let the run.sh build it: PW_ARCH_AUTO_ENV=1 bin/pw run arch-moe --target cpu
PW_ARCH_BENCH_ARCH=mixtral PW_ARCH_BENCH_SIZE=medium bin/pw run arch-bench --target gpu --bench --device "<GPU name>"
```

`tools/torch-cpu-env.sh` is the route to a CPU PyTorch when `download.pytorch.org` is blocked and PyPI's torch wheel is the
CUDA build: it downloads the static `micromamba` binary from conda-forge (sha256 checked), then creates an environment
from conda-forge only (`pytorch=2.13.0=cpu_*`, `transformers`, `numpy`, `pyyaml`; about 1 GB, 2 minutes). `PW_TORCH_CPU_PREFIX`,
`PW_MAMBA_ROOT` and `PW_TORCH_SPEC` override its prefix, package cache and specs; it exits 77 when it cannot build.

Knobs: `PW_ARCH_ATTN=eager` (default `sdpa`), `PW_ARCH_EXPERTS=eager|batched_mm|grouped_mm` (MoE; transformers' default is
`grouped_mm`, which may lack a kernel on a simulated GPU), `PW_ARCH_ONLY=<case,case>`. Simulated targets use the
existing `workloads/_shared/torch_env.sh` (pantheonsim's PyTorch environments); the Python there needs `transformers`
installed or the workload exits 77 saying so. Not tried here: no pantheonsim build was available.

## Licences (verified)

- `transformers` 5.18.0: Apache-2.0. Read from the installed package's `LICENSE` ("Apache License Version 2.0", "Copyright
  2018- The Hugging Face team"), its `METADATA` (`License: Apache 2.0 License`) and the conda-forge package record
  (`license: Apache-2.0`).
- PyTorch 2.13.0 (conda-forge CPU build): BSD-3-Clause per the conda-forge package record; the shipped `LICENSE` is PyTorch's.
- `micromamba` 2.0.5: conda-forge package metadata says `BSD-3-Clause AND MIT AND OpenSSL` (read from the tarball's `info/index.json`); only a download helper, nothing is redistributed.
- No model weights, tokenizers, datasets or images are used or committed; all inputs are seeded random tensors.
