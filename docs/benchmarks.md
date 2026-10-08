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

Not recorded: `onnx-zoo-bertsquad-int8-bench` (its functional check fails on this host, see that file). `llamacpp-smollm2-135m` and `whisper-cpp-tiny-en` print no metrics, so `--bench`
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
