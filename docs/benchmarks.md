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
