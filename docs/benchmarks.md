# Recording benchmark numbers

Benchmark numbers are for **real GPUs only**. The simulator runs on a CPU, so `--bench` is refused
for `sim:` targets, and a workload that reports `metrics` on a simulated target fails.

```bash
bin/pw run smollm2-135m-ollama --target gpu --bench --repeat 5 [--device "NVIDIA H100 80GB HBM3"]
bin/pw bench-table            # markdown table of everything recorded under bench/
```

`--bench` runs the workload `--repeat` times (default 3) and writes
`bench/<workload>/<UTC time>-<device>.json` only when **every** run passed its functional check.
A number from a run that produced the wrong answer is never recorded.

## Record format (`pw-bench/0`)

```json
{
  "schema": "pw-bench/0",
  "workload": "smollm2-135m-ollama",
  "date": "2026-10-06T12:00:00Z",
  "environment": {
    "target": "gpu",
    "device": "NVIDIA H100 80GB HBM3",
    "driver": "…",
    "runtime": "ollama",
    "runtime_version": "ollama version is …",
    "runtime_versions": {"torch": "2.14.1+cu130"},
    "model": {"id": "smollm2:135m", "revision": "…", "licence": "…", "source_url": "…"},
    "host_os": "Linux-…",
    "repo_commit": "…",
    "repo_dirty": false
  },
  "repeats": 5,
  "metrics": {"decode_tokens_per_s": {"median": 0.0, "values": [0.0, 0.0, 0.0, 0.0, 0.0]}}
}
```

(The numbers above are placeholders, not measurements.)

`runtime_versions` is optional: a workload may add a `versions` object to its result line (the PyTorch workloads
do, for torch, torchvision, transformers, diffusers and numpy, read from the installed packages by
`workloads/_pytorch/env.sh`). When present, `bin/pw` copies it into the record and sets `runtime_version` from it,
and refuses to record repeats that ran with different versions. Records made before this field existed carry the
manifest's free-text `runtime_version` only; the A10G records of 2026-10-08 used torch 2.10.0, not 2.14.1.

## Rules for numbers worth keeping

- **The device must be known.** NVIDIA cards are detected with `nvidia-smi`; for AMD and anything
  else pass `--device`. Without a device the run is refused, because an unattributed number is useless.
- **Pin the model.** A manifest with `model.revision: null` records `null`; pin it first or the
  number cannot be reproduced.
- **A dirty tree is recorded** (`repo_dirty`). Do not commit results from one.
- **One machine per record.** Compare cards by comparing records, never by averaging them.
- Record clocks, power limits and the like by hand in the commit message when they were changed
  from the defaults; the runner does not read them yet.

## What has been recorded

First real-GPU run, 2026-10-08: one NVIDIA A10G (24 GB, sm_86, ECC on, 300 W, default clocks), AWS g5.2xlarge, Ubuntu 22.04
Deep Learning AMI, driver 595.91.07. 40 records under `bench/`, 5 repeats each, from a clean tree at commit `7efdb83`
(`repo_dirty: false`), recorded with `PW_BENCH_DIR` outside the checkout so earlier records do not make later ones dirty.
`bin/pw bench-table` prints them. Runtime versions: torch 2.10.0+cu130, transformers 5.19.0, onnxruntime-gpu 1.30.0 (CUDA EP, TF32
off), spaCy 3.8.16 with cupy-cuda12x, llama.cpp b11447 (CUDA, sm_86, `-ngl 99`), whisper.cpp v1.9.5 (CUDA, sm_86), Ollama 0.40.1,
vLLM 0.30.0 (torch 2.13.0+cu130, `VLLM_USE_FLASHINFER_SAMPLER=0`). Full account, setup traps and every failure: `docs/first-gpu-run-a10g.md`.

`onnx-zoo-bertsquad-int8-bench` was recorded on 2026-10-09 (g5.xlarge, A10G, driver 595.91.07, 5 repeats, 242.99 ms per question, onnxruntime-gpu 1.30.0 with `session.x64quantprecision=1`; `docs/int8-and-nondeterminism.md`); its check failed on 2026-10-08 on the AVX2 host. The kokoro and moonshine records of 2026-10-08 predate that session option (which can change the CPU-side int8 kernels) and were not re-recorded. `llamacpp-smollm2-135m` and `whisper-cpp-tiny-en` print no metrics, so `--bench`
refuses them; the benchmark twins carry the numbers. `pytorch-microsuite`, `gpt2-small-pytorch`, `bert-base-uncased-pytorch`,
`resnet18-randinit-pytorch` and `smollm2-135m-ollama` are functional workloads that also report metrics, and were recorded with `--bench`.

Second rig, same day (another g5.2xlarge, A10G, 1710 MHz, 300 W, default clocks; `repo_dirty: false`, 5 repeats, `-ngl 99`, pp512/tg128, llama.cpp b11447 CUDA sm_86):
`llamacpp-bench-qwen25-0p5b` (official Q8_0 GGUF) pp512 27159.84, tg128 455.45 tokens/s (medians; tg spread 455.35 to 455.74);
`llamacpp-bench-mistral-7b-v03` (Q8_0, 7.7 GB) pp512 4014.08, tg128 61.02 tokens/s (tg spread 61.01 to 61.02).

Caveats that apply to the numbers:

- The ONNX workloads run with the CUDA execution provider and the CPU provider as fallback. Nodes without a CUDA kernel (probably in the int8
  models: Kokoro, Moonshine; not checked per node) would run on the host CPU inside the same session, so those numbers may measure a mixed placement
  (Kokoro ran at 0.74x real time, which suggests it).
- The llama.cpp `llamacpp-synth-bench-*` models are tiny random-weight GGUFs; their tokens/s say nothing about real model sizes.
- One host, one card, one run day. The A10G is a cloud card; compare cards only by comparing records.
- Not collected yet: memory, time to first token, AMD devices (not auto-detected).

## Tier 1 models (2026-10-08, third g5.2xlarge, NVIDIA A10G)

Same card and software as above (driver 595.91.07; llama.cpp b11447 CUDA sm_86 `-ngl 99`, whisper.cpp v1.9.5 CUDA, torch 2.10.0+cu130, transformers 5.19.0,
diffusers 0.41.0), 5 repeats each, recorded from a clean checkout of commit `978153e` (`repo_dirty: false`) with `PW_BENCH_DIR` outside it.
Medians (the records under `bench/` hold all five values; llama-bench's own mean of 3 sits inside each value):

| Workload | Metric | Median |
| --- | --- | ---: |
| `llamacpp-bench-llama32-1b-instruct` (Q8_0) | pp512 / tg128 tokens/s | 18391.93 / 299.64 |
| `llamacpp-bench-llama32-3b-instruct` (Q8_0) | pp512 / tg128 tokens/s | 7646.12 / 123.27 |
| `llamacpp-bench-gemma4-e4b-it` (QAT Q4_0) | pp512 / tg128 tokens/s | 5459.27 / 114.45 |
| `llamacpp-bench-gpt-oss-20b` (MXFP4) | pp512 / tg128 tokens/s | 4388.35 / 143.39 |
| `llamacpp-bench-deepseek-r1-distill-qwen-1p5b` (Q8_0) | pp512 / tg128 tokens/s | 12804.82 / 219.05 |
| `llamacpp-bench-deepseek-r1-distill-qwen-7b` (Q8_0) | pp512 / tg128 tokens/s | 4229.72 / 60.37 |
| `whisper-cpp-bench-large-v3` | encode / decode ms per run (lower is better) | 96.46 / 7.72 |
| `sdxl-base-bench` (fp16, 1024x1024, 25 steps, batch 1) | images/s; UNet steps/s | 0.1268; 3.169 |
| `blip2-opt-2p7b-bench` (fp16) | captions/s at batch 8; decode tokens/s at batch 1 | 17.207; 45.46 |

Caveats: the Llama 3.2, SDXL and BLIP-2 workloads are `restricted: true`; the recorded runtime version for the PyTorch and whisper records is `null` as for
the earlier PyTorch records. `sdxl-base-bench` excludes model load and one warm-up image; BLIP-2's decode figure includes the vision tower and Q-Former once per caption.
The functional unit-test run on the same host (233 tests) had two failures that are not caused by these workloads: `lib-attention-precision` against its CPU
reference (the known fp8 conversion difference of `docs/first-gpu-run-a10g.md`) and a vLLM glue test that assumes no `vllm-greedy-smollm2-135m` reference exists.

Setup traps seen on this run: `uv pip install torch==2.10.0 --index-url .../cu130 --extra-index-url pypi` silently picks PyPI's `+cu128` build, which
`workloads/_pytorch/env.sh` rejects for the `cu130` flavour; add `--index-strategy unsafe-best-match` (or drop the extra index for torch). The first load of the 7 GB SDXL
files from a fresh EBS volume took minutes; later loads 2 to 18 s. whisper.cpp's CUDA build took about 40 min on 8 vCPUs when competing with other jobs.

## torch 2.14.1 re-run (2026-10-08, fourth g5.2xlarge, NVIDIA A10G)

The PyTorch workloads were re-run on the pinned **torch 2.14.1+cu130 / torchvision 0.29.1+cu130** (transformers 5.19.0, diffusers 0.41.0,
numpy 2.5.3; `torch.__version__` read on the rig, not inferred), one on-demand g5.2xlarge in us-east-1b, the same AMI family and driver
(595.91.07) as the 2.10.0 records above. Each bench twin was re-recorded with `--bench --repeat 5` from a clean tree (`repo_dirty: false`,
commit `479c1c7` when run; the commit hash in the records is the pre-rebase one) with `PW_BENCH_DIR` outside the checkout. The new records sit beside the
old ones in `bench/<name>/` and carry `environment.runtime_versions`; the old ones have no such field and ran torch 2.10.0+cu130.
`mem 99`: these workloads have no memory-fraction option, so none was set.

The 2.10.0 records were not touched. Medians; change is 2.14.1 against the 2026-10-08 2.10.0 record; `**` marks a change above 5%.
The last column is a control: the same checkout re-run with torch 2.10.0+cu130 on this same instance, to tell a torch effect from a
different-instance effect (5 repeats; run only for `arch-bench`, `lib-kernels-bench` and `blip2-opt-2p7b-bench`, `-` = not run). The control (arch-bench, lib-kernels-bench, blip2-opt-2p7b-bench) agrees with the old 2.10.0 records to within 1.4% for
arch-bench and 0.7% for BLIP-2, and within 2.9% for every one of the 56 lib-kernels-bench metrics; the slightly lower matmul numbers
in the two last rows (-2.5% / -2.4%) are already in the control (65.44), so they are an instance difference, not a torch effect.

| Workload | Metric | 2.10.0 (2026-10-08) | 2.14.1 | Change | 2.10.0 same rig |
| --- | --- | ---: | ---: | ---: | ---: |
| `arch-bench` | `llama_small_decode_tokens_per_s` | 1381 | 2086 | +51.1% ** | 1400 |
| `arch-bench` | `llama_small_prefill_tokens_per_s` | 2.885e+05 | 3.904e+05 | +35.3% ** | 2.888e+05 |
| `blip2-opt-2p7b-bench` | `captions_per_s_b8` | 17.21 | 18.07 | +5.0% ** (spread 6.4%) | 17.09 |
| `blip2-opt-2p7b-bench` | `decode_tokens_per_s_b1` | 45.46 | 57.62 | +26.7% ** | 45.47 |
| `gpt-train-bench` | `bf16_steps_per_s` | 35.22 | 35.29 | +0.2% | - |
| `gpt-train-bench` | `bf16_tokens_per_s` | 2.885e+05 | 2.891e+05 | +0.2% | - |
| `gpt-train-bench` | `fp32_steps_per_s` | 19.41 | 19.3 | -0.6% | - |
| `gpt-train-bench` | `fp32_tokens_per_s` | 1.59e+05 | 1.581e+05 | -0.6% | - |
| `gpt2-small-pytorch` | `decode_tokens_per_s` | 103.7 | 104.6 | +0.9% | - |
| `pytorch-microsuite` | `conv2d_fp16_tflops` | 43.28 | 43.49 | +0.5% | - |
| `pytorch-microsuite` | `copy_gb_s` | 482.2 | 482.1 | -0.0% | - |
| `pytorch-microsuite` | `matmul_bf16_tflops` | 62.32 | 62.39 | +0.1% | - |
| `pytorch-microsuite` | `matmul_fp16_tflops` | 62.32 | 62.39 | +0.1% | - |
| `pytorch-microsuite` | `matmul_fp32_tflops` | 23.11 | 22.48 | -2.8% | - |
| `pytorch-microsuite` | `matmul_tf32_tflops` | 30.91 | 30.86 | -0.2% | - |
| `pytorch-microsuite` | `sdpa_causal_fp16_tflops` | 54.96 | 54.91 | -0.1% | - |
| `resnet18-randinit-pytorch` | `images_per_s_b64_224` | 3233 | 3175 | -1.8% | - |
| `sdxl-base-bench` | `images_per_s_1024_25steps` | 0.1268 | 0.1273 | +0.4% | - |
| `sdxl-base-bench` | `unet_steps_per_s` | 3.169 | 3.183 | +0.4% | - |
| `lib-kernels-bench` | `dlrm_forward_fp32_samples_s` | 7.529e+06 | 1.551e+07 | +106.0% ** | 7.513e+06 |
| `lib-kernels-bench` | `index_add_gb_s` | 148.9 | 156.4 | +5.0% ** | 148.9 |
| `lib-kernels-bench` | `lu_solve_batched_32x32_systems_s` | 7.2e+06 | 9.093e+06 | +26.3% ** | 7.103e+06 |
| `lib-kernels-bench` | `matmul_bf16_tflops` | 67.13 | 65.48 | -2.5% | 65.44 |
| `lib-kernels-bench` | `matmul_fp16_tflops` | 65.91 | 64.32 | -2.4% | 64.76 |

`lib-kernels-bench` has 56 metrics; the rows listed for it are the five outside 2%
(51 of 56 are within 2%, worst slowdown -2.5%). There is no slowdown above 5% anywhere.

Reading it: everything limited by the GPU's arithmetic or memory bandwidth (matmul, convolution, FFT, linear algebra, attention,
the SDXL UNet, the whole `gpt-train-bench`) did not move beyond noise. The large speedups are all in workloads that are bound by the
host launching many small kernels: a 4-layer, 512-wide model's prefill and decode (`arch-bench`), batch-1 BLIP-2 decoding, a DLRM
forward pass, and batches of 32x32 solves. With torch 2.10.0 on the same instance these came out at the old values (1400 vs 1381,
45.47 vs 45.46, 7.51e6 vs 7.53e6), so the change belongs to torch 2.14.1 (probably lower per-op CPU dispatch cost; the cause inside
torch was not investigated), not to the instance. `index_add` +5.0% is a real small gain (tight spread, reproduced against the control).
Do not compare a record made with torch 2.10.0 with one made with 2.14.1 for these small-kernel workloads.

## Vision-language models (2026-10-09, fifth g5.2xlarge, NVIDIA A10G)

One on-demand g5.2xlarge in us-east-1d, the same AMI family and driver (595.91.07) as above, torch 2.14.1+cu130 / torchvision 0.29.1+cu130 /
transformers 5.19.0 / numpy 2.5.3 (read from the installed packages; recorded in `environment.runtime_versions`). The three `-bench` twins were run with
`--bench --repeat 5` from a clean clone of commit `94554d9` (`repo_dirty: false`) with `PW_BENCH_DIR` outside the checkout. All three are fp16, batch 1, greedy, SDPA attention,
one 512x512 image (scikit-image `astronaut.png`), model load and image preprocessing excluded. Medians of the 5 repeats (every repeat is itself the median of 5 timed pairs):

| Workload | `decode_tokens_per_s_b1` | `prefill_images_per_s_b1` | `peak_gpu_memory_gb` | Prompt tokens | Spread of the 5 decode values |
| --- | ---: | ---: | ---: | ---: | --- |
| `qwen25-vl-7b-bench` | 28.76 | 4.699 | 16.72 | 352 | 28.75 to 28.77 |
| `qwen3-vl-4b-bench` | 23.64 | 9.828 | 9.03 | 273 | 23.57 to 23.77 |
| `smolvlm2-2p2b-bench` | 47.99 | 2.727 | 5.24 | 1439 | 45.39 to 48.68 |

Definitions: `decode_tokens_per_s_b1` = 127 / (time of a forced 128-token generation minus time of a 1-token generation), so the vision tower and the prompt
prefill cancel out; `prefill_images_per_s_b1` = 1 / (time of a 1-token generation), i.e. the image through the vision tower plus the prompt through the language
model plus one decoding step; `peak_gpu_memory_gb` = `torch.cuda.max_memory_allocated()` over the whole process, including the 128-token KV cache.

Reading it: these decode rates are far below the memory-bandwidth bound of the card (roughly 600 GB/s over 8 to 16 GB of weights would allow 40 to 75 tokens/s), and the
4B model decodes slower than the 7B (the 4B Qwen3 language model has 36 layers against 28, and it adds deep-stack vision features), which is the signature of a batch-1 eager
decode limited by the host launching many small kernels, the same effect as in the torch 2.14.1 section above. That explanation was not verified with a profile. The SmolVLM2 prefill is
the slowest of the three because its image processor splits the image into tiles (1439 prompt tokens against 273 and 352); its decode spread (6%) is the widest. Do not
compare these with the llama.cpp numbers (different runtime, quantised weights) or with the BLIP-2 captions/s (different task and batch).

## More llama.cpp models (2026-10-09, sixth g5.2xlarge, NVIDIA A10G)

One on-demand g5.2xlarge (us-east-1a), driver 595.91.07, llama.cpp b11447 CUDA sm_86 (built with `PW_CUDA_ARCHS=86`, plus `llama-embedding`), `-ngl 99`, `llama-bench`
pp512/tg128 with `-r 3` inside each repeat, 5 repeats per workload through `bin/pw run --bench --repeat 5`, `PW_BENCH_DIR` outside the checkout. The records under `bench/`
carry `repo_commit` of the commit that added the workloads (the rig had no `.git`; the tree was identical). Medians of the 5 repeats (range of the 5 in brackets):

| Workload | Quantisation | pp512 tokens/s | tg128 tokens/s |
| --- | --- | ---: | ---: |
| `llamacpp-bench-qwen3-30b-a3b` (MoE, 18.6 GB) | Q4_K_M | 3429.22 (3417.76 to 3429.97) | 155.85 (155.65 to 155.88) |
| `llamacpp-bench-qwen3-8b` | Q6_K | 3555.48 (3496.55 to 3561.72) | 69.15 (69.07 to 69.17) |
| `llamacpp-bench-phi4-14b` | Q4_K | 2412.72 (2405.73 to 2413.73) | 51.23 (51.22 to 51.23) |
| `llamacpp-bench-granite40-h-small` (hybrid Mamba-2 MoE, 19.5 GB) | Q4_K_M | 1754.60 (1754.00 to 1754.93) | 66.46 (66.44 to 66.48) |
| `llamacpp-bench-mistral-small-32-24b` (24B, 14.3 GB) | Q4_K_M | 1513.79 (1513.29 to 1516.21) | 32.83 (32.82 to 32.83) |
| `llamacpp-bench-olmo2-7b-instruct` | Q8_0 | 3910.10 (3899.31 to 3911.22) | 61.59 (61.57 to 61.59) |
| `llamacpp-bench-qwen3-embedding-4b` | Q8_0 | 6427.55 (6417.91 to 6433.08) | 96.52 (96.48 to 96.53) |

Reading it: the MoE with about 3B active parameters decodes 2.25x faster than the dense 8B Q6_K (it reads far fewer weight bytes per token), and the 24B dense model at 14.3 GB
decodes at 32.8 tokens/s, close to what the card's memory bandwidth allows for that file size (roughly 600 GB/s over 14.3 GB, an upper bound of about 42 tokens/s; not profiled). The tg128 of the Qwen3-Embedding-4B
is a generation benchmark of an embedding model and says nothing about embedding throughput (the functional workload does not time anything). Do not compare these with the
Q8_0 rows of the Tier 1 table without the quantisation in mind. All five repeats ran on one card in one session: they bound run-to-run noise on this host, not card-to-card variation.
