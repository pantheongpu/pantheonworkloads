# pantheonworkloads

Open-source AI workloads (LLM inference and more) with pinned models and reproducible benchmarks, run on real GPUs and on pantheonsim's simulated NVIDIA and AMD GPUs to check functional correctness.

## Two kinds of numbers, kept apart

| | Where it runs | What it records | What it is for |
| --- | --- | --- | --- |
| **Functional** | [pantheonsim](https://github.com/pantheongpu/pantheonsim) profile, or a real GPU | pass / fail against a reference output, within a stated tolerance | "does the simulated GPU model behave like the real one for this workload?" |
| **Benchmark** | real GPU only | tokens/s (more to come), with device, driver, runtime and model revision | "how fast is this workload on this card?" This repo is where those numbers are kept. |

The simulator executes on a CPU, so its wall-clock time says nothing about GPU speed.
Functional results therefore never carry performance numbers, and benchmark files are
refused for simulated targets.

## Layout

```
workloads/<name>/manifest.yaml   what the workload is: source, model pointer, licence, tolerance
workloads/<name>/run.sh          runs it; prints one JSON object on its last line (docs/workload-contract.md)
workloads/<name>/reference.json  expected output, recorded on a trusted target (absent until recorded)
bin/pw                           the runner: pw list | validate | run | record | matrix | bench-table | coverage
bench/<name>/*.json              recorded benchmark numbers from real GPUs (committed)
results/                         functional run output (git-ignored)
docs/                            manifest, workload contract, benchmarks, trace schema (draft)
```

## Quick start

```bash
bin/pw validate                                   # check every manifest
bin/pw run selftest --target cpu                  # no GPU, no model: exercises the runner
bin/pw run smollm2-135m-ollama --target sim:nvidia/h100   # needs ollama and a pantheonsim build
bin/pw matrix results/                            # one table from every run's results.tsv
bin/pw run smollm2-135m-ollama --target gpu --bench --repeat 5   # real GPU: record benchmark numbers
bin/pw bench-table                                # everything recorded under bench/
bin/pw coverage                                   # the table under Coverage below, from the repo data
```

`--target` is `cpu`, `gpu` (whatever the host has) or `sim:<vendor>/<profile>`
(for example `sim:amd/mi300x`). For `sim:` targets `PANTHEONSIM_DIR` must point at a
built pantheonsim checkout (its `build/vgpu`).

Results go to `results/<run-id>/`: `results.tsv` (workload, result, seconds, detail, the same
columns pantheonsim's workload matrix reads), `machine.txt`, and one `<workload>.json` per workload.

## Models and licences

The repository stores **pointers** to models (id, revision, checksum), never weights, and each
manifest records the model's licence. Models are fetched by the workload's own script.
Check a model's licence before adding it: the terms differ a lot between model families and
change between versions.

## Benchmark results

Real-GPU numbers only (the simulator never carries performance numbers). Generated from `bench/`; do not edit by hand.

<!-- bench:start -->

131 records from real GPUs (newest per workload and device; all metrics, every record and the software versions: [`docs/results.md`](docs/results.md)).

| Workload | Device | Metrics (median of N runs) | Software | Date |
| --- | --- | --- | --- | --- |
| arch-bench | NVIDIA A10G | llama_small_decode_tokens_per_s 2,086; llama_small_prefill_tokens_per_s 390,386 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| bert-base-uncased-pytorch | NVIDIA A10G | forward_sequences_per_s_b32_s128 640.5 (n=5) | not recorded | 2026-10-08 |
| bge-m3-bench | NVIDIA A10G | forward_passes_per_s 67.86; peak_gpu_memory_gb 1.15 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| bge-reranker-large-bench | NVIDIA A10G | forward_passes_per_s 69.31; peak_gpu_memory_gb 1.13 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| bge-reranker-v2-m3-bench | NVIDIA A10G | forward_passes_per_s 68.74; peak_gpu_memory_gb 1.15 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| blip2-opt-2p7b-bench | NVIDIA A10G | captions_per_s_b8 18.07; decode_tokens_per_s_b1 57.62 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| canary-1b-v2-bench | NVIDIA A10G | audio_seconds_per_s 29.79 (n=5) | torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| chatterbox-bench | NVIDIA A10G | audio_seconds_per_s 1.22 (n=5) | diffusers 0.29.0, torch 2.6.0, transformers 5.2.0 | 2026-10-10 |
| chronos-2-bench | NVIDIA A10G | forecasts_per_s_b1 36.08 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| chronos-bolt-base-bench | NVIDIA A10G | forecasts_per_s_b1 31.09 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| chronos-t5-large-bench | NVIDIA A10G | forecasts_per_s_b1 1.17 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| cogvideox-2b-bench | NVIDIA L40S | peak_gpu_memory_gb 29.54; seconds_per_generation 49.35; steps_per_s 0.608 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| cogvideox-5b-bench | NVIDIA L40S | peak_gpu_memory_gb 36.98; seconds_per_generation 142.2; steps_per_s 0.211 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| crnn-text-recognition-bench | NVIDIA A10G | images_per_s 659.8; latency_ms 1.516 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| depth-anything-v2-large-bench | NVIDIA A10G | images_per_s_b1 8.93; peak_gpu_memory_gb 1.68 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| depth-anything-v2-small-bench | NVIDIA A10G | images_per_s_b1 52.41; peak_gpu_memory_gb 0.22 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| dinov2-giant-bench | NVIDIA A10G | images_per_s_b1 10.81; peak_gpu_memory_gb 4.59 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| f5-tts-bench | NVIDIA A10G | audio_seconds_per_s 2.21 (n=5) | torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| flux1-schnell-bench | NVIDIA L40S | peak_gpu_memory_gb 36.3; seconds_per_generation 2.252; steps_per_s 1.776 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| flux2-klein-4b-bench | NVIDIA A10G | peak_gpu_memory_gb 18.6; seconds_per_generation 3.224; steps_per_s 1.241 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| glm-ocr-bench | NVIDIA A10G | decode_tokens_per_s_b1 51.9; peak_gpu_memory_gb 2.29; prefill_images_per_s_b1 14.85 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| glove-wiki-gigaword-50-knn-bench | NVIDIA A10G | batch_latency_ms 2.725; queries_per_s 93,958 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| gpt-train-bench | NVIDIA A10G | bf16_steps_per_s 35.29; bf16_tokens_per_s 289,064; fp32_steps_per_s 19.3; +1 more (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| gpt2-small-pytorch | NVIDIA A10G | decode_tokens_per_s 104.6 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| gtcrn-enhance-onnx-bench | NVIDIA A10G | frames_per_s 152.3; realtime_factor 2.435 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| hunyuanvideo-15-720p-t2v-bench | NVIDIA L40S | peak_gpu_memory_gb 37.93; seconds_per_generation 175.5; steps_per_s 0.171 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| kokoro-tts-int8-onnx-bench | NVIDIA A10G | latency_ms_per_sentence 5,112; realtime_factor 0.6749 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| kws-zipformer-gigaspeech-onnx-bench | NVIDIA A10G | latency_ms_per_clip 617.1; realtime_factor 18.91 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| lib-kernels-bench | NVIDIA A10G | autocast_bf16_linear_tflops 61.8; autocast_fp16_linear_tflops 61.55; batch_norm_train_fp32_gb_s 306.7; +53 more (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| llamacpp-bench-codestral-22b-v01 | NVIDIA A10G | pp512_tokens_per_s 1,388; tg128_tokens_per_s 34.16 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-command-r-08-2024 | NVIDIA L40S | pp512_tokens_per_s 2,582; tg128_tokens_per_s 34.59 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-deepseek-r1-distill-llama-8b | NVIDIA A10G | pp512_tokens_per_s 3,987; tg128_tokens_per_s 58.13 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-deepseek-r1-distill-qwen-14b | NVIDIA A10G | pp512_tokens_per_s 2,184; tg128_tokens_per_s 50.32 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-deepseek-r1-distill-qwen-1p5b | NVIDIA A10G | pp512_tokens_per_s 12,805; tg128_tokens_per_s 219.1 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-deepseek-r1-distill-qwen-32b | NVIDIA L40S | pp512_tokens_per_s 2,464; tg128_tokens_per_s 33.63 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-deepseek-r1-distill-qwen-7b | NVIDIA A10G | pp512_tokens_per_s 4,230; tg128_tokens_per_s 60.37 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-devstral-small-2-24b-2512 | NVIDIA A10G | pp512_tokens_per_s 1,503; tg128_tokens_per_s 31.88 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-exaone-4p5-33b | NVIDIA L40S | pp512_tokens_per_s 2,484; tg128_tokens_per_s 33.88 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-falcon-h1-34b-instruct | NVIDIA L40S | pp512_tokens_per_s 2,184; tg128_tokens_per_s 29.32 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-falcon-h1r-7b | NVIDIA A10G | pp512_tokens_per_s 3,043; tg128_tokens_per_s 50.78 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-falcon3-10b-instruct | NVIDIA A10G | pp512_tokens_per_s 3,064; tg128_tokens_per_s 43.77 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-gemma4-12b-it | NVIDIA A10G | pp512_tokens_per_s 2,631; tg128_tokens_per_s 59.44 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-gemma4-26b-a4b-it | NVIDIA A10G | pp512_tokens_per_s 3,712; tg128_tokens_per_s 123 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-gemma4-31b-it | NVIDIA A10G | pp512_tokens_per_s 1,083; tg128_tokens_per_s 25.31 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-gemma4-e2b-it | NVIDIA A10G | pp512_tokens_per_s 8,560; tg128_tokens_per_s 193.3 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-gemma4-e4b-it | NVIDIA A10G | pp512_tokens_per_s 5,459; tg128_tokens_per_s 114.5 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-glm47-flash | NVIDIA A10G | pp512_tokens_per_s 2,729; tg128_tokens_per_s 111.4 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-gpt-oss-20b | NVIDIA A10G | pp512_tokens_per_s 4,388; tg128_tokens_per_s 143.4 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-granite40-h-small | NVIDIA A10G | pp512_tokens_per_s 1,755; tg128_tokens_per_s 66.46 (n=5) | llama.cpp b11447 | 2026-10-09 |
| llamacpp-bench-kimi-linear-48b-a3b-instruct | NVIDIA L40S | pp512_tokens_per_s 5,155; tg128_tokens_per_s 162.6 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-llama31-8b-instruct | NVIDIA A10G | pp512_tokens_per_s 3,984; tg128_tokens_per_s 58.13 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-llama32-1b-instruct | NVIDIA A10G | pp512_tokens_per_s 18,392; tg128_tokens_per_s 299.6 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-llama32-3b-instruct | NVIDIA A10G | pp512_tokens_per_s 7,646; tg128_tokens_per_s 123.3 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-magistral-small-2509 | NVIDIA A10G | pp512_tokens_per_s 1,514; tg128_tokens_per_s 32.83 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-ministral-3-14b-instruct-2512 | NVIDIA A10G | pp512_tokens_per_s 2,484; tg128_tokens_per_s 33.63 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-ministral-3-14b-reasoning-2512 | NVIDIA A10G | pp512_tokens_per_s 2,483; tg128_tokens_per_s 33.64 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-mistral-7b-v03 | NVIDIA A10G | pp512_tokens_per_s 4,014; tg128_tokens_per_s 61.02 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-mistral-small-32-24b | NVIDIA A10G | pp512_tokens_per_s 1,514; tg128_tokens_per_s 32.83 (n=5) | llama.cpp b11447 | 2026-10-09 |
| llamacpp-bench-nemotron-3-nano-30b-a3b | NVIDIA L40S | pp512_tokens_per_s 7,256; tg128_tokens_per_s 168 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-nemotron-3p5-lightning-30b-a3b | NVIDIA L40S | pp512_tokens_per_s 7,057; tg128_tokens_per_s 177.7 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-nemotron-nano-9b-v2 | NVIDIA A10G | pp512_tokens_per_s 2,878; tg128_tokens_per_s 49.15 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-north-mini-code-1p0 | NVIDIA A10G | pp512_tokens_per_s 3,201; tg128_tokens_per_s 143.5 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-olmo2-7b-instruct | NVIDIA A10G | pp512_tokens_per_s 3,910; tg128_tokens_per_s 61.59 (n=5) | llama.cpp b11447 | 2026-10-09 |
| llamacpp-bench-orpheus-3b-0p1-ft | NVIDIA A10G | pp512_tokens_per_s 7,617; tg128_tokens_per_s 120.5 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-phi4-14b | NVIDIA A10G | pp512_tokens_per_s 2,413; tg128_tokens_per_s 51.23 (n=5) | llama.cpp b11447 | 2026-10-09 |
| llamacpp-bench-phi4-mini-instruct | NVIDIA A10G | pp512_tokens_per_s 7,420; tg128_tokens_per_s 105.8 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-phi4-reasoning-plus | NVIDIA A10G | pp512_tokens_per_s 2,410; tg128_tokens_per_s 51.22 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen25-0p5b | NVIDIA A10G | pp512_tokens_per_s 27,160; tg128_tokens_per_s 455.4 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-bench-qwen25-coder-32b-instruct | NVIDIA L40S | pp512_tokens_per_s 2,465; tg128_tokens_per_s 33.63 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen3-30b-a3b | NVIDIA A10G | pp512_tokens_per_s 3,429; tg128_tokens_per_s 155.8 (n=5) | llama.cpp b11447 | 2026-10-09 |
| llamacpp-bench-qwen3-32b | NVIDIA L40S | pp512_tokens_per_s 2,453; tg128_tokens_per_s 33.73 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen3-8b | NVIDIA A10G | pp512_tokens_per_s 3,555; tg128_tokens_per_s 69.15 (n=5) | llama.cpp b11447 | 2026-10-09 |
| llamacpp-bench-qwen3-coder-30b-a3b-instruct | NVIDIA A10G | pp512_tokens_per_s 3,379; tg128_tokens_per_s 155.7 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen3-embedding-4b | NVIDIA A10G | pp512_tokens_per_s 6,428; tg128_tokens_per_s 96.52 (n=5) | llama.cpp b11447 | 2026-10-09 |
| llamacpp-bench-qwen3-embedding-8b | NVIDIA A10G | pp512_tokens_per_s 3,877; tg128_tokens_per_s 56.86 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen35-9b | NVIDIA A10G | pp512_tokens_per_s 3,267; tg128_tokens_per_s 53.08 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen36-27b | NVIDIA L40S | pp512_tokens_per_s 2,590; tg128_tokens_per_s 33.57 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen36-35b-a3b | NVIDIA L40S | pp512_tokens_per_s 6,235; tg128_tokens_per_s 146.6 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-qwen38-27b | NVIDIA A10G | pp512_tokens_per_s 1,097; tg128_tokens_per_s 24.2 (n=5) | llama.cpp b11447 | 2026-10-10 |
| llamacpp-bench-smollm2-135m | NVIDIA A10G | pp512_tokens_per_s 36,022; tg128_tokens_per_s 654.9 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-synth-bench-gemma | NVIDIA A10G | pp512_tokens_per_s 48,279; tg128_tokens_per_s 1,182 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-synth-bench-llama | NVIDIA A10G | pp512_tokens_per_s 47,737; tg128_tokens_per_s 1,164 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-synth-bench-mistral | NVIDIA A10G | pp512_tokens_per_s 51,574; tg128_tokens_per_s 1,195 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-synth-bench-mixtral | NVIDIA A10G | pp512_tokens_per_s 40,886; tg128_tokens_per_s 1,053 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-synth-bench-phi3 | NVIDIA A10G | pp512_tokens_per_s 54,902; tg128_tokens_per_s 1,280 (n=5) | llama.cpp b11447 | 2026-10-08 |
| llamacpp-synth-bench-qwen2 | NVIDIA A10G | pp512_tokens_per_s 50,341; tg128_tokens_per_s 1,178 (n=5) | llama.cpp b11447 | 2026-10-08 |
| ltx-video-bench | NVIDIA L40S | peak_gpu_memory_gb 15.96; seconds_per_generation 4.684; steps_per_s 6.404 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| minilm-l6-v2-onnx-bench | NVIDIA A10G | batch_latency_ms 4.003; sentences_per_s 8,994 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| moonshine-tiny-en-onnx-bench | NVIDIA A10G | latency_ms_per_clip 530.3; realtime_factor 22.01 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| nanodet-object-detection-bench | NVIDIA A10G | images_per_s 111.4; latency_ms 8.974 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| olmocr-2-7b-bench | NVIDIA A10G | decode_tokens_per_s_b1 28.8; peak_gpu_memory_gb 16.68; prefill_images_per_s_b1 4.971 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| onnx-zoo-bertsquad-int8-bench | NVIDIA A10G | latency_ms 242.5; questions_per_s 4.124 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| onnx-zoo-efficientnet-lite4-bench | NVIDIA A10G | batch_latency_ms 2.877; images_per_s 347.6 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| onnx-zoo-mobilenetv2-bench | NVIDIA A10G | batch_latency_ms 13.8; images_per_s 2,318 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| onnx-zoo-shufflenet-v2-bench | NVIDIA A10G | batch_latency_ms 1.35; images_per_s 740.5 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| onnx-zoo-ssd-mobilenetv1-bench | NVIDIA A10G | images_per_s 86.16; latency_ms 11.61 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| paddleocr-vl-bench | NVIDIA A10G | decode_tokens_per_s_b1 19.15; peak_gpu_memory_gb 1.92; prefill_images_per_s_b1 16.91 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| parakeet-tdt-0p6b-v3-bench | NVIDIA A10G | audio_seconds_per_s 111.3 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| pixtral-12b-2409-bench | NVIDIA L40S | decode_tokens_per_s_b1 27; peak_gpu_memory_gb 25.95; prefill_images_per_s_b1 5.932 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| pphumanseg-person-segmentation-bench | NVIDIA A10G | images_per_s 319.2; latency_ms 3.132 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| ppocr-rapidocr-bench | NVIDIA A10G | images_per_s 7.259; latency_ms 137.8 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| pytorch-microsuite | NVIDIA A10G | conv2d_fp16_tflops 43.49; copy_gb_s 482.1; matmul_bf16_tflops 62.39; +4 more (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| qwen-image-2p1-bench | NVIDIA L40S | peak_gpu_memory_gb 39.53; seconds_per_generation 12.07; steps_per_s 2.486 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| qwen25-vl-7b-bench | NVIDIA A10G | decode_tokens_per_s_b1 28.76; peak_gpu_memory_gb 16.72; prefill_images_per_s_b1 4.699 (n=5) | torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-09 |
| qwen3-asr-1p7b-bench | NVIDIA A10G | audio_seconds_per_s 10.88 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| qwen3-reranker-0p6b-bench | NVIDIA A10G | forward_passes_per_s 25.44; peak_gpu_memory_gb 1.32 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| qwen3-reranker-4b-bench | NVIDIA A10G | forward_passes_per_s 12.59; peak_gpu_memory_gb 8.18 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| qwen3-reranker-8b-bench | NVIDIA A10G | forward_passes_per_s 7.3; peak_gpu_memory_gb 16.52 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| qwen3-vl-4b-bench | NVIDIA A10G | decode_tokens_per_s_b1 23.64; peak_gpu_memory_gb 9.03; prefill_images_per_s_b1 9.828 (n=5) | torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-09 |
| resnet18-randinit-pytorch | NVIDIA A10G | images_per_s_b64_224 3,175 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| sam2p1-hiera-large-bench | NVIDIA A10G | images_per_s_b1 4.84; peak_gpu_memory_gb 1.42 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| sdxl-base-bench | NVIDIA A10G | images_per_s_1024_25steps 0.1273; unet_steps_per_s 3.183 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-08 |
| sface-face-embedding-bench | NVIDIA A10G | images_per_s 1,194; latency_ms 0.8377 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| siglip2-giant-opt-patch16-384-bench | NVIDIA A10G | images_per_s_b1 8.64; peak_gpu_memory_gb 7.54 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| siglip2-so400m-patch16-512-bench | NVIDIA A10G | images_per_s_b1 10.14; peak_gpu_memory_gb 4.61 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| silero-vad-onnx-bench | NVIDIA A10G | realtime_factor 59.61; windows_per_s 1,863 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| smollm2-135m-ollama | NVIDIA A10G | decode_tokens_per_s 399.7 (n=5) | Warning: could not connect to a running Ollama instance | 2026-10-08 |
| smolvlm2-2p2b-bench | NVIDIA A10G | decode_tokens_per_s_b1 47.99; peak_gpu_memory_gb 5.24; prefill_images_per_s_b1 2.727 (n=5) | torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-09 |
| spacy-en-core-web-md-bench | NVIDIA A10G | docs_per_s 998; words_per_s 15,303 (n=5) | spacy 3.8.16 | 2026-10-09 |
| spacy-en-core-web-sm-bench | NVIDIA A10G | docs_per_s 962.4; words_per_s 16,362 (n=5) | spacy 3.8.16 | 2026-10-09 |
| spacy-multilingual-sm-bench | NVIDIA A10G | nb_core_news_sm_words_per_s 13,039; ru_core_news_sm_words_per_s 13,045; uk_core_news_sm_words_per_s 11,673; +1 more (n=5) | spacy 3.8.16 | 2026-10-09 |
| vllm-bench-throughput | NVIDIA A10G | output_tokens_per_s 32,545; requests_per_s 254.3; total_tokens_per_s 65,090 (n=5) | not recorded | 2026-10-08 |
| wan21-t2v-1p3b-bench | NVIDIA L40S | peak_gpu_memory_gb 19.09; seconds_per_generation 37.38; steps_per_s 0.803 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| wan22-ti2v-5b-bench | NVIDIA L40S | peak_gpu_memory_gb 27.65; seconds_per_generation 21.91; steps_per_s 1.369 (n=5) | diffusers 0.41.0, torch 2.14.1+cu130, transformers 5.19.0 | 2026-10-10 |
| wespeaker-resnet34-onnx-bench | NVIDIA A10G | embeddings_per_s 14.22; realtime_factor 141.1 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| whisper-cpp-bench-large-v3 | NVIDIA A10G | decode_ms_per_run 7.72; encode_ms_per_run 96.46 (n=5) | not recorded | 2026-10-08 |
| whisper-cpp-bench-large-v3-turbo | NVIDIA A10G | decode_ms_per_run 1.31; encode_ms_per_run 85.45 (n=5) | not recorded | 2026-10-10 |
| whisper-cpp-bench-tiny-en | NVIDIA A10G | decode_ms_per_run 0.82; encode_ms_per_run 3.07 (n=5) | not recorded | 2026-10-08 |
| yolox-object-detection-bench | NVIDIA A10G | images_per_s 71.29; latency_ms 14.03 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| yunet-face-detection-bench | NVIDIA A10G | images_per_s 212.4; latency_ms 4.707 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |
| zipformer-audio-tagging-onnx-bench | NVIDIA A10G | clips_per_s 21.17; realtime_factor 174.3 (n=5) | onnxruntime-gpu 1.30.0 | 2026-10-09 |

<!-- bench:end -->

## Coverage

The table below is generated by `bin/pw coverage` from the manifests, `reference.json` files, `model.sha256`
files, `bench/` and the results tables in `docs/*validation*.md`. Nothing in it is typed by hand, and
`tests/test_coverage.py` fails when it drifts from the repository; after changing a manifest, a reference or a
validation table, run `bin/pw coverage --write`.

How to read it:

- **Reference** is the output a run is compared with. `n/a` marks the `-bench` twins (benchmarks record
  metrics, not a reference output). `yes (cpu)` is read from the reference file's `recorded_on`: every
  reference so far was recorded on the CPU, none on a GPU.
- **Sim-validated** counts the simulated cards whose run passed in [`docs/sim-validation.md`](docs/sim-validation.md).
  Only the PyTorch workloads (and the runner self-check) have been run on pantheonsim so far; the onnxruntime,
  spaCy and llama.cpp workloads have not, and nothing has run on `sim:amd` beyond `selftest` (no ROCm PyTorch
  was obtainable there).
- **GPU-validated** is `bench` when `bench/<name>/*.json` exists and `functional` when a validation table
  records a passing run on a real GPU. `no` means no real GPU has run it yet.
- **Model pinned** is `yes` when a sha256 of the weights (or of the generated files) is recorded,
  `revision only` when only a revision such as a sha1 id is, `no` when the model is unpinned, and `n/a`
  when the workload has no weights.

Honest caveats:

- References were recorded on the CPU, so a real GPU may disagree for reasons that are not bugs. Tolerances are
  reasoned or measured against the CPU and the simulator, and are **unverified on any physical GPU** until a
  GPU run says otherwise. The first GPU runs are the test of those tolerances.
- The `arch-*`, `lib-*`, `gpt-train-*`, `pytorch-microsuite`, `resnet18-randinit-pytorch` and `llamacpp-synth-*`
  workloads use seeded random weights or inputs. They test kernels, graphs and code paths, **not** the quality
  of any named pretrained model.
- `UNVERIFIED` licences (the SmolLM2, GPT-2, BERT and Mistral workloads on Hugging Face, Ollama and vLLM) mean
  the model card could not be read when the manifest was written. Those workloads are also unpinned, have no
  reference, and have not run. Check the licence before relying on them.
- The MiniLM model is **second-hand**: its Apache-2.0 licence is stated by a third-party npm repackaging, not
  by the original repository. The two Zipformer models (keyword spotting, audio tagging) have their licence only
  from the model card inside the release archive; the terms of their training data were not found.
- `whisper-cpp-tiny-en` and `whisper-cpp-bench-tiny-en` are pinned by a sha1 only, and the model could not be
  downloaded where they were written, so no transcript, reference or benchmark exists for them.
- `loadgen-plumbing-check` runs MLCommons LoadGen against a toy CPU system under test. It is **not an MLPerf
  result** ([`docs/mlperf.md`](docs/mlperf.md)).
- Trace capture is designed only ([`docs/trace-schema.md`](docs/trace-schema.md) is a draft).
- The **Model catalog** group ([`docs/model-registry.md`](docs/model-registry.md)) holds workloads for open-weight models of any size
  (DeepSeek V4, Kimi K3, Qwen3.8, GLM-5.3, Llama, Mistral, FLUX, Wan, Whisper, ...), written from Hugging Face metadata on 2026-10-09:
  **none has been run**, none has a reference or a bench record, and each declares in `requires:` the GPUs it needs (up to 8 x 256 GiB), so
  `bin/pw` skips it on a smaller host. Gated repositories are listed there as blocked, with the access route.

Further reading: [`docs/model-registry.md`](docs/model-registry.md) (every tracked model, its pin, licence, runtime and hardware need),
[`docs/models.md`](docs/models.md) (what ran and what has not),
[`docs/pretrained-reachable.md`](docs/pretrained-reachable.md) (licence evidence, pins, what was rejected),
[`docs/arch-coverage.md`](docs/arch-coverage.md), [`docs/lib-coverage.md`](docs/lib-coverage.md),
[`docs/llamacpp.md`](docs/llamacpp.md), [`docs/llamacpp-synth.md`](docs/llamacpp-synth.md),
[`docs/benchmarks.md`](docs/benchmarks.md) and [`docs/sim-validation.md`](docs/sim-validation.md).

<!-- coverage:start -->

#### llama.cpp (synthetic GGUF, pretrained GGUF and llama-bench): 51

| Workload | Kind | Runtime | Targets | Needs | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llamacpp-bench-deepseek-r1-distill-qwen-1p5b | benchmark | llama.cpp | gpu | any | n/a | MIT | yes | no | bench |
| llamacpp-bench-deepseek-r1-distill-qwen-7b | benchmark | llama.cpp | gpu | any | n/a | MIT | yes | no | bench |
| llamacpp-bench-gemma4-e4b-it | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-gpt-oss-20b | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-granite40-h-small | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-llama32-1b-instruct | benchmark | llama.cpp | gpu | any | n/a | Llama 3.2 Community Licence (restricted) | yes | no | bench |
| llamacpp-bench-llama32-3b-instruct | benchmark | llama.cpp | gpu | any | n/a | Llama 3.2 Community Licence (restricted) | yes | no | bench |
| llamacpp-bench-mistral-7b-v03 | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 (per card) | yes | no | bench |
| llamacpp-bench-mistral-small-32-24b | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-olmo2-7b-instruct | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-phi4-14b | benchmark | llama.cpp | gpu | any | n/a | MIT | yes | no | bench |
| llamacpp-bench-qwen25-0p5b | benchmark | llama.cpp | cpu, gpu | any | n/a | Apache-2.0 (per card) | yes | no | bench |
| llamacpp-bench-qwen3-30b-a3b | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen3-8b | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen3-embedding-4b | benchmark | llama.cpp | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-smollm2-135m | benchmark | llama.cpp | cpu, gpu | any | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-deepseek-r1-distill-qwen-1p5b | functional | llama.cpp | gpu | any | yes (gpu) | MIT | yes | no | no |
| llamacpp-deepseek-r1-distill-qwen-7b | functional | llama.cpp | gpu | any | yes (gpu) | MIT | yes | no | no |
| llamacpp-gemma4-e4b-it | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-gpt-oss-20b | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-granite40-h-small | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-llama32-1b-instruct | functional | llama.cpp | gpu | any | yes (gpu) | Llama 3.2 Community Licence (restricted) | yes | no | no |
| llamacpp-llama32-3b-instruct | functional | llama.cpp | gpu | any | yes (gpu) | Llama 3.2 Community Licence (restricted) | yes | no | no |
| llamacpp-mistral-7b-v03 | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 (per card) | yes | no | no |
| llamacpp-mistral-small-32-24b | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-olmo2-7b-instruct | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-phi4-14b | functional | llama.cpp | gpu | any | yes (gpu) | MIT | yes | no | no |
| llamacpp-qwen25-0p5b | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (gpu) | Apache-2.0 (per card) | yes | no | no |
| llamacpp-qwen3-30b-a3b | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-8b | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-embedding-4b | functional | llama.cpp | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-smollm2-135m | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-synth-bench-gemma | benchmark | llama.cpp | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| llamacpp-synth-bench-llama | benchmark | llama.cpp | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| llamacpp-synth-bench-mistral | benchmark | llama.cpp | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| llamacpp-synth-bench-mixtral | benchmark | llama.cpp | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| llamacpp-synth-bench-phi3 | benchmark | llama.cpp | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| llamacpp-synth-bench-qwen2 | benchmark | llama.cpp | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| llamacpp-synth-gemma | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-llama | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-mistral | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-mixtral | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-phi3 | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-quant-float | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-quant-iquant | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-quant-iquant-imatrix | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-quant-kquant | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-quant-legacy | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-quant-lowbit | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-quant-moe | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |
| llamacpp-synth-qwen2 | functional | llama.cpp | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | yes | no | no |

#### Pretrained models on onnxruntime, spaCy, numpy and sentencepiece: 52

| Workload | Kind | Runtime | Targets | Needs | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crnn-text-recognition | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| crnn-text-recognition-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| glove-wiki-gigaword-50-knn | functional | onnxruntime | cpu, gpu | any | yes (cpu) | PDDL-1.0 | yes | no | no |
| glove-wiki-gigaword-50-knn-bench | benchmark | onnxruntime | gpu | any | n/a | PDDL-1.0 | yes | no | bench |
| gtcrn-enhance-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| gtcrn-enhance-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | MIT | yes | no | bench |
| kokoro-tts-int8-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| kokoro-tts-int8-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| kws-zipformer-gigaspeech-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 (per card) | yes | no | no |
| kws-zipformer-gigaspeech-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 (per card) | yes | no | bench |
| minilm-l6-v2-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 (second-hand) | yes | no | no |
| minilm-l6-v2-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 (second-hand) | yes | no | bench |
| moonshine-tiny-en-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| moonshine-tiny-en-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | MIT | yes | no | bench |
| nanodet-object-detection | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| nanodet-object-detection-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| onnx-zoo-bertsquad-int8 | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| onnx-zoo-bertsquad-int8-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| onnx-zoo-bidaf | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| onnx-zoo-efficientnet-lite4 | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| onnx-zoo-efficientnet-lite4-bench | benchmark | onnxruntime | gpu | any | n/a | MIT | yes | no | bench |
| onnx-zoo-mnist | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| onnx-zoo-mobilenetv2 | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| onnx-zoo-mobilenetv2-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| onnx-zoo-shufflenet-v2 | functional | onnxruntime | cpu, gpu | any | yes (cpu) | BSD-3-Clause | yes | no | no |
| onnx-zoo-shufflenet-v2-bench | benchmark | onnxruntime | gpu | any | n/a | BSD-3-Clause | yes | no | bench |
| onnx-zoo-ssd-mobilenetv1 | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| onnx-zoo-ssd-mobilenetv1-bench | benchmark | onnxruntime | gpu | any | n/a | MIT | yes | no | bench |
| pphumanseg-person-segmentation | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| pphumanseg-person-segmentation-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| ppocr-rapidocr | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| ppocr-rapidocr-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| py3langid-wheel | functional | numpy | cpu | any | yes (cpu) | BSD-3-Clause | yes | no | no |
| sentencepiece-test-model | functional | sentencepiece | cpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| sface-face-embedding | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| sface-face-embedding-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| silero-vad-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| silero-vad-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | MIT | yes | no | bench |
| spacy-en-core-web-md | functional | spacy | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| spacy-en-core-web-md-bench | benchmark | spacy | gpu | any | n/a | MIT | yes | no | bench |
| spacy-en-core-web-sm | functional | spacy | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| spacy-en-core-web-sm-bench | benchmark | spacy | gpu | any | n/a | MIT | yes | no | bench |
| spacy-multilingual-sm | functional | spacy | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| spacy-multilingual-sm-bench | benchmark | spacy | gpu | any | n/a | MIT | yes | no | bench |
| wespeaker-resnet34-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | CC-BY-4.0 | yes | no | no |
| wespeaker-resnet34-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | CC-BY-4.0 | yes | no | bench |
| yolox-object-detection | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 | yes | no | no |
| yolox-object-detection-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| yunet-face-detection | functional | onnxruntime | cpu, gpu | any | yes (cpu) | MIT | yes | no | no |
| yunet-face-detection-bench | benchmark | onnxruntime | gpu | any | n/a | MIT | yes | no | bench |
| zipformer-audio-tagging-onnx | functional | onnxruntime | cpu, gpu | any | yes (cpu) | Apache-2.0 (per card) | yes | no | no |
| zipformer-audio-tagging-onnx-bench | benchmark | onnxruntime | gpu | any | n/a | Apache-2.0 (per card) | yes | no | bench |

#### Architecture coverage (random-weight PyTorch): 6

| Workload | Kind | Runtime | Targets | Needs | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arch-bench | benchmark | pytorch | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| arch-encdec | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| arch-llama-family | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| arch-moe | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| arch-ssm | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| arch-vision | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |

#### GPU-library coverage (random inputs, PyTorch): 6

| Workload | Kind | Runtime | Targets | Needs | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lib-attention-precision | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| lib-composite-blocks | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| lib-fft-linalg | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| lib-kernels-bench | benchmark | pytorch | gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| lib-rnn-conv | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |
| lib-sparse-embedding | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x4 | no |

#### Training: 3

| Workload | Kind | Runtime | Targets | Needs | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-train-bench | benchmark | pytorch | cpu, gpu | any | n/a | n/a (no weights) | n/a | no | bench |
| gpt-train-bf16 | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x5 | no |
| gpt-train-fp32 | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x6 | no |

#### Other (Hugging Face PyTorch, whisper.cpp, vLLM, Ollama, MLPerf LoadGen, runner self-check): 23

| Workload | Kind | Runtime | Targets | Needs | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert-base-uncased-pytorch | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (gpu) | Apache-2.0 | yes | no | bench |
| blip2-opt-2p7b-bench | benchmark | pytorch | gpu | any | n/a | OPT-2.7b licence (restricted) | yes | no | bench |
| blip2-opt-2p7b-pytorch | functional | pytorch | gpu | any | yes (gpu) | OPT-2.7b licence (restricted) | yes | no | no |
| gpt2-small-pytorch | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (gpu) | MIT (per card) | yes | no | bench |
| loadgen-plumbing-check | functional | none | cpu | any | yes (cpu) | n/a (no weights) | n/a | no | no |
| pytorch-microsuite | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x8 | bench |
| qwen25-vl-7b-bench | benchmark | pytorch | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| qwen25-vl-7b-pytorch | functional | pytorch | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| qwen3-vl-4b-bench | benchmark | pytorch | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| qwen3-vl-4b-pytorch | functional | pytorch | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| resnet18-randinit-pytorch | functional | pytorch | cpu, gpu, sim:nvidia, sim:amd | any | yes (cpu) | n/a (no weights) | n/a | nvidia x8 | bench |
| sdxl-base-bench | benchmark | pytorch | gpu | any | n/a | OpenRAIL++-M (restricted) | yes | no | bench |
| sdxl-base-diffusers | functional | pytorch | gpu | any | yes (gpu) | OpenRAIL++-M (restricted) | yes | no | no |
| selftest | functional | none | cpu, gpu, sim:* | any | yes (cpu) | n/a (no weights) | n/a | amd x1, nvidia x1 | no |
| smollm2-135m-ollama | functional | ollama | cpu, gpu, sim:nvidia | any | yes (gpu) | Apache-2.0 | yes | no | bench |
| smolvlm2-2p2b-bench | benchmark | pytorch | gpu | any | n/a | Apache-2.0 | yes | no | bench |
| smolvlm2-2p2b-pytorch | functional | pytorch | gpu | any | yes (gpu) | Apache-2.0 | yes | no | no |
| vllm-bench-throughput | benchmark | vllm | gpu | any | n/a | Apache-2.0 (per card) | yes | no | bench |
| vllm-greedy-smollm2-135m | functional | vllm | gpu, sim:amd | any | yes (gpu) | Apache-2.0 (per card) | yes | no | no |
| whisper-cpp-bench-large-v3 | benchmark | whisper.cpp | gpu | any | n/a | MIT | yes | no | bench |
| whisper-cpp-bench-tiny-en | benchmark | whisper.cpp | gpu | any | n/a | MIT | yes | no | bench |
| whisper-cpp-large-v3 | functional | whisper.cpp | gpu | any | yes (gpu) | MIT | yes | no | no |
| whisper-cpp-tiny-en | functional | whisper.cpp | cpu, gpu | any | yes (gpu) | MIT | yes | no | no |

#### Model catalog: written from Hub metadata (docs/model-registry.md); models of any size, each gated by `requires`. A row with a reference recorded and a GPU-validated entry has had its first real run (see the registry for the state of each); the others were never run: 258

| Workload | Kind | Runtime | Targets | Needs | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bge-m3-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | MIT | yes | no | bench |
| bge-m3-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | MIT | yes | no | no |
| bge-reranker-large-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | MIT | yes | no | bench |
| bge-reranker-large-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | MIT | yes | no | no |
| bge-reranker-v2-m3-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| bge-reranker-v2-m3-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| canary-1b-v2-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | CC-BY-4.0 | yes | no | bench |
| canary-1b-v2-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | CC-BY-4.0 | yes | no | no |
| chatterbox-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | MIT | yes | no | bench |
| chatterbox-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | MIT | yes | no | no |
| chronos-2-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| chronos-2-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| chronos-bolt-base-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| chronos-bolt-base-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| chronos-t5-large-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| chronos-t5-large-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| cogvideox-2b-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | Apache-2.0 | yes | no | bench |
| cogvideox-2b-diffusers | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| cogvideox-5b-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | custom licence (restricted) | yes | no | bench |
| cogvideox-5b-diffusers | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | custom licence (restricted) | yes | no | no |
| cogvideox15-5b-bench | benchmark | pytorch | gpu | 1 x 48 GiB | n/a | custom licence (restricted) | yes | no | no |
| cogvideox15-5b-diffusers | functional | pytorch | gpu | 1 x 48 GiB | no | custom licence (restricted) | yes | no | no |
| depth-anything-v2-large-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | CC-BY-NC-4.0 (restricted) | yes | no | bench |
| depth-anything-v2-large-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | CC-BY-NC-4.0 (restricted) | yes | no | no |
| depth-anything-v2-small-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| depth-anything-v2-small-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| dinov2-giant-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| dinov2-giant-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| f5-tts-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | CC-BY-NC-4.0 (restricted) | yes | no | bench |
| f5-tts-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | CC-BY-NC-4.0 (restricted) | yes | no | no |
| flux1-schnell-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | Apache-2.0 | yes | no | bench |
| flux1-schnell-diffusers | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| flux2-klein-4b-bench | benchmark | pytorch | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| flux2-klein-4b-diffusers | functional | pytorch | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| glm-ocr-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | MIT | yes | no | bench |
| glm-ocr-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | MIT | yes | no | no |
| glm46v-bench | benchmark | pytorch | gpu | 4 x 80 GiB | n/a | MIT | yes | no | no |
| glm46v-pytorch | functional | pytorch | gpu | 4 x 80 GiB | no | MIT | yes | no | no |
| hunyuanvideo-15-720p-t2v-bench | benchmark | pytorch | gpu | 1 x 44 GiB | n/a | Tencent Hunyuan Community License (restricted) | yes | no | bench |
| hunyuanvideo-15-720p-t2v-diffusers | functional | pytorch | gpu | 1 x 44 GiB | yes (gpu) | Tencent Hunyuan Community License (restricted) | yes | no | no |
| hunyuanvideo-bench | benchmark | pytorch | gpu | 1 x 80 GiB | n/a | Tencent Hunyuan Community License (restricted) | yes | no | no |
| hunyuanvideo-diffusers | functional | pytorch | gpu | 1 x 80 GiB | no | Tencent Hunyuan Community License (restricted) | yes | no | no |
| llamacpp-bench-codestral-22b-v01 | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Mistral AI Non-Production License (restricted) | yes | no | bench |
| llamacpp-bench-command-a-03-2025 | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | CC-BY-NC-4.0 (restricted) | yes | no | no |
| llamacpp-bench-command-a-plus-05-2026 | benchmark | llama.cpp | gpu | 2 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-command-r-08-2024 | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | CC-BY-NC-4.0 (restricted) | yes | no | bench |
| llamacpp-bench-command-r-plus-08-2024 | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | CC-BY-NC-4.0 (restricted) | yes | no | no |
| llamacpp-bench-deepseek-r1-0528 | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-deepseek-r1-distill-llama-70b | benchmark | llama.cpp | gpu | 1 x 48 GiB | n/a | Llama 3.3 Community Licence (restricted) | yes | no | no |
| llamacpp-bench-deepseek-r1-distill-llama-8b | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | Llama 3.1 Community Licence (restricted) | yes | no | bench |
| llamacpp-bench-deepseek-r1-distill-qwen-14b | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | MIT | yes | no | bench |
| llamacpp-bench-deepseek-r1-distill-qwen-32b | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | MIT | yes | no | bench |
| llamacpp-bench-deepseek-v3p1-terminus | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-deepseek-v3p2 | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-deepseek-v4-flash-0731 | benchmark | llama.cpp | gpu | 4 x 44 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-deepseek-v4-pro-0813 | benchmark | llama.cpp | gpu | 4 x 256 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-devstral-2-123b-2512 | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | custom licence (restricted) | yes | no | no |
| llamacpp-bench-devstral-small-2-24b-2512 | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-exaone-4p5-33b | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | EXAONE AI Model License Agreement (restricted) | yes | no | bench |
| llamacpp-bench-falcon-h1-34b-instruct | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | Falcon LLM License (restricted) | yes | no | bench |
| llamacpp-bench-falcon-h1r-7b | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | Falcon LLM License (restricted) | yes | no | bench |
| llamacpp-bench-falcon3-10b-instruct | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | Falcon LLM License (restricted) | yes | no | bench |
| llamacpp-bench-gemma4-12b-it | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-gemma4-26b-a4b-it | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-gemma4-31b-it | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-gemma4-e2b-it | benchmark | llama.cpp | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-glm47 | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-glm47-flash | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | MIT | yes | no | bench |
| llamacpp-bench-glm52 | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-glm53 | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | GLM-5.3 License (restricted) | yes | no | no |
| llamacpp-bench-glm53-flash | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | MIT | yes | no | no |
| llamacpp-bench-gpt-oss-120b | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-hy3 | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-kimi-k2-thinking | benchmark | llama.cpp | gpu | 4 x 180 GiB | n/a | Modified MIT (restricted) | yes | no | no |
| llamacpp-bench-kimi-k26 | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | Modified MIT (restricted) | yes | no | no |
| llamacpp-bench-kimi-k27-code | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | Modified MIT (restricted) | yes | no | no |
| llamacpp-bench-kimi-k3 | benchmark | llama.cpp | gpu | 8 x 192 GiB | n/a | Kimi K3 License (restricted) | yes | no | no |
| llamacpp-bench-kimi-linear-48b-a3b-instruct | benchmark | llama.cpp | gpu | 1 x 40 GiB | n/a | MIT | yes | no | bench |
| llamacpp-bench-llama31-405b-instruct | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | Llama 3.1 Community Licence (restricted) | yes | no | no |
| llamacpp-bench-llama31-70b-instruct | benchmark | llama.cpp | gpu | 1 x 48 GiB | n/a | Llama 3.1 Community Licence (restricted) | yes | no | no |
| llamacpp-bench-llama31-8b-instruct | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | Llama 3.1 Community Licence (restricted) | yes | no | bench |
| llamacpp-bench-llama33-70b-instruct | benchmark | llama.cpp | gpu | 1 x 48 GiB | n/a | Llama 3.3 Community Licence (restricted) | yes | no | no |
| llamacpp-bench-llama4-maverick-17b-128e-instruct | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | Llama 4 Community License (restricted) | yes | no | no |
| llamacpp-bench-llama4-scout-17b-16e-instruct | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | Llama 4 Community License (restricted) | yes | no | no |
| llamacpp-bench-magistral-small-2509 | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-minimax-m2p5 | benchmark | llama.cpp | gpu | 2 x 80 GiB | n/a | Modified MIT (restricted) | yes | no | no |
| llamacpp-bench-minimax-m2p7 | benchmark | llama.cpp | gpu | 2 x 80 GiB | n/a | custom licence (restricted) | yes | no | no |
| llamacpp-bench-minimax-m3 | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | minimax-community (restricted) | yes | no | no |
| llamacpp-bench-ministral-3-14b-instruct-2512 | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-ministral-3-14b-reasoning-2512 | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-mistral-large-3-675b-instruct-2512 | benchmark | llama.cpp | gpu | 8 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-mistral-large-instruct-2411 | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | Mistral Research License (restricted) | yes | no | no |
| llamacpp-bench-mistral-medium-3p5-128b | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | custom licence (restricted) | yes | no | no |
| llamacpp-bench-mistral-small-4-119b-2603 | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-mixtral-8x22b-instruct-v01 | benchmark | llama.cpp | gpu | 2 x 48 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-nemotron-3-nano-30b-a3b | benchmark | llama.cpp | gpu | 1 x 40 GiB | n/a | NVIDIA Nemotron Open Model License (restricted) | yes | no | bench |
| llamacpp-bench-nemotron-3-super-120b-a12b | benchmark | llama.cpp | gpu | 2 x 48 GiB | n/a | NVIDIA Nemotron Open Model License (restricted) | yes | no | no |
| llamacpp-bench-nemotron-3-ultra-550b-a55b | benchmark | llama.cpp | gpu | 8 x 48 GiB | n/a | OpenMDW-1.1 (restricted) | yes | no | no |
| llamacpp-bench-nemotron-3p5-lightning-30b-a3b | benchmark | llama.cpp | gpu | 1 x 40 GiB | n/a | OpenMDW-1.1 (restricted) | yes | no | bench |
| llamacpp-bench-nemotron-nano-9b-v2 | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | NVIDIA Open Model License (restricted) | yes | no | bench |
| llamacpp-bench-north-mini-code-1p0 | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-orpheus-3b-0p1-ft | benchmark | llama.cpp | gpu | 1 x 10 GiB | n/a | Apache-2.0 (restricted) | yes | no | bench |
| llamacpp-bench-phi4-mini-instruct | benchmark | llama.cpp | gpu | 1 x 10 GiB | n/a | MIT | yes | no | bench |
| llamacpp-bench-phi4-reasoning-plus | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | MIT | yes | no | bench |
| llamacpp-bench-qwen25-72b-instruct | benchmark | llama.cpp | gpu | 1 x 48 GiB | n/a | Qwen License (restricted) | yes | no | no |
| llamacpp-bench-qwen25-coder-32b-instruct | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen3-235b-a22b-instruct-2507 | benchmark | llama.cpp | gpu | 2 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-qwen3-235b-a22b-thinking-2507 | benchmark | llama.cpp | gpu | 2 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-qwen3-32b | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen3-coder-30b-a3b-instruct | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen3-coder-480b-a35b-instruct | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-qwen3-coder-next | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-qwen3-embedding-8b | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen3-next-80b-a3b-instruct | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-qwen35-122b-a10b | benchmark | llama.cpp | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-qwen35-397b-a17b | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-qwen35-9b | benchmark | llama.cpp | gpu | 1 x 15 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen36-27b | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen36-35b-a3b | benchmark | llama.cpp | gpu | 1 x 24 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen38-27b | benchmark | llama.cpp | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| llamacpp-bench-qwen38-2p4t-a95b | benchmark | llama.cpp | gpu | 8 x 180 GiB | n/a | qwen3.8-max (restricted) | yes | no | no |
| llamacpp-bench-qwen38-flash-next | benchmark | llama.cpp | gpu | 2 x 80 GiB | n/a | Qwen Community License 1.0 (restricted) | yes | no | no |
| llamacpp-bench-step-3p7-flash | benchmark | llama.cpp | gpu | 2 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| llamacpp-bench-trinity-large-thinking | benchmark | llama.cpp | gpu | 4 x 80 GiB | n/a | OpenMDW-1.1 (restricted) | yes | no | no |
| llamacpp-codestral-22b-v01 | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Mistral AI Non-Production License (restricted) | yes | no | no |
| llamacpp-command-a-03-2025 | functional | llama.cpp | gpu | 1 x 80 GiB | no | CC-BY-NC-4.0 (restricted) | yes | no | no |
| llamacpp-command-a-plus-05-2026 | functional | llama.cpp | gpu | 2 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-command-r-08-2024 | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | CC-BY-NC-4.0 (restricted) | yes | no | no |
| llamacpp-command-r-plus-08-2024 | functional | llama.cpp | gpu | 1 x 80 GiB | no | CC-BY-NC-4.0 (restricted) | yes | no | no |
| llamacpp-deepseek-r1-0528 | functional | llama.cpp | gpu | 8 x 80 GiB | no | MIT | yes | no | no |
| llamacpp-deepseek-r1-distill-llama-70b | functional | llama.cpp | gpu | 1 x 48 GiB | no | Llama 3.3 Community Licence (restricted) | yes | no | no |
| llamacpp-deepseek-r1-distill-llama-8b | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | Llama 3.1 Community Licence (restricted) | yes | no | no |
| llamacpp-deepseek-r1-distill-qwen-14b | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | MIT | yes | no | no |
| llamacpp-deepseek-r1-distill-qwen-32b | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | MIT | yes | no | no |
| llamacpp-deepseek-v3p1-terminus | functional | llama.cpp | gpu | 8 x 80 GiB | no | MIT | yes | no | no |
| llamacpp-deepseek-v3p2 | functional | llama.cpp | gpu | 8 x 80 GiB | no | MIT | yes | no | no |
| llamacpp-deepseek-v4-flash-0731 | functional | llama.cpp | gpu | 4 x 44 GiB | no | MIT | yes | no | no |
| llamacpp-deepseek-v4-pro-0813 | functional | llama.cpp | gpu | 4 x 256 GiB | no | MIT | yes | no | no |
| llamacpp-devstral-2-123b-2512 | functional | llama.cpp | gpu | 1 x 80 GiB | no | custom licence (restricted) | yes | no | no |
| llamacpp-devstral-small-2-24b-2512 | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-exaone-4p5-33b | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | EXAONE AI Model License Agreement (restricted) | yes | no | no |
| llamacpp-falcon-h1-34b-instruct | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | Falcon LLM License (restricted) | yes | no | no |
| llamacpp-falcon-h1r-7b | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | Falcon LLM License (restricted) | yes | no | no |
| llamacpp-falcon3-10b-instruct | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | Falcon LLM License (restricted) | yes | no | no |
| llamacpp-gemma4-12b-it | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-gemma4-26b-a4b-it | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-gemma4-31b-it | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-gemma4-e2b-it | functional | llama.cpp | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-glm47 | functional | llama.cpp | gpu | 4 x 80 GiB | no | MIT | yes | no | no |
| llamacpp-glm47-flash | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | MIT | yes | no | no |
| llamacpp-glm52 | functional | llama.cpp | gpu | 8 x 80 GiB | no | MIT | yes | no | no |
| llamacpp-glm53 | functional | llama.cpp | gpu | 8 x 80 GiB | no | GLM-5.3 License (restricted) | yes | no | no |
| llamacpp-glm53-flash | functional | llama.cpp | gpu | 4 x 80 GiB | no | MIT | yes | no | no |
| llamacpp-gpt-oss-120b | functional | llama.cpp | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-hy3 | functional | llama.cpp | gpu | 4 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-kimi-k2-thinking | functional | llama.cpp | gpu | 4 x 180 GiB | no | Modified MIT (restricted) | yes | no | no |
| llamacpp-kimi-k26 | functional | llama.cpp | gpu | 8 x 80 GiB | no | Modified MIT (restricted) | yes | no | no |
| llamacpp-kimi-k27-code | functional | llama.cpp | gpu | 8 x 80 GiB | no | Modified MIT (restricted) | yes | no | no |
| llamacpp-kimi-k3 | functional | llama.cpp | gpu | 8 x 192 GiB | no | Kimi K3 License (restricted) | yes | no | no |
| llamacpp-kimi-linear-48b-a3b-instruct | functional | llama.cpp | gpu | 1 x 40 GiB | yes (gpu) | MIT | yes | no | no |
| llamacpp-llama31-405b-instruct | functional | llama.cpp | gpu | 4 x 80 GiB | no | Llama 3.1 Community Licence (restricted) | yes | no | no |
| llamacpp-llama31-70b-instruct | functional | llama.cpp | gpu | 1 x 48 GiB | no | Llama 3.1 Community Licence (restricted) | yes | no | no |
| llamacpp-llama31-8b-instruct | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | Llama 3.1 Community Licence (restricted) | yes | no | no |
| llamacpp-llama33-70b-instruct | functional | llama.cpp | gpu | 1 x 48 GiB | no | Llama 3.3 Community Licence (restricted) | yes | no | no |
| llamacpp-llama4-maverick-17b-128e-instruct | functional | llama.cpp | gpu | 4 x 80 GiB | no | Llama 4 Community License (restricted) | yes | no | no |
| llamacpp-llama4-scout-17b-16e-instruct | functional | llama.cpp | gpu | 1 x 80 GiB | no | Llama 4 Community License (restricted) | yes | no | no |
| llamacpp-magistral-small-2509 | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-minimax-m2p5 | functional | llama.cpp | gpu | 2 x 80 GiB | no | Modified MIT (restricted) | yes | no | no |
| llamacpp-minimax-m2p7 | functional | llama.cpp | gpu | 2 x 80 GiB | no | custom licence (restricted) | yes | no | no |
| llamacpp-minimax-m3 | functional | llama.cpp | gpu | 4 x 80 GiB | no | minimax-community (restricted) | yes | no | no |
| llamacpp-ministral-3-14b-instruct-2512 | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-ministral-3-14b-reasoning-2512 | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-mistral-large-3-675b-instruct-2512 | functional | llama.cpp | gpu | 8 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-mistral-large-instruct-2411 | functional | llama.cpp | gpu | 1 x 80 GiB | no | Mistral Research License (restricted) | yes | no | no |
| llamacpp-mistral-medium-3p5-128b | functional | llama.cpp | gpu | 1 x 80 GiB | no | custom licence (restricted) | yes | no | no |
| llamacpp-mistral-small-4-119b-2603 | functional | llama.cpp | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-mixtral-8x22b-instruct-v01 | functional | llama.cpp | gpu | 2 x 48 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-nemotron-3-nano-30b-a3b | functional | llama.cpp | gpu | 1 x 40 GiB | yes (gpu) | NVIDIA Nemotron Open Model License (restricted) | yes | no | no |
| llamacpp-nemotron-3-super-120b-a12b | functional | llama.cpp | gpu | 2 x 48 GiB | no | NVIDIA Nemotron Open Model License (restricted) | yes | no | no |
| llamacpp-nemotron-3-ultra-550b-a55b | functional | llama.cpp | gpu | 8 x 48 GiB | no | OpenMDW-1.1 (restricted) | yes | no | no |
| llamacpp-nemotron-3p5-lightning-30b-a3b | functional | llama.cpp | gpu | 1 x 40 GiB | yes (gpu) | OpenMDW-1.1 (restricted) | yes | no | no |
| llamacpp-nemotron-nano-9b-v2 | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | NVIDIA Open Model License (restricted) | yes | no | no |
| llamacpp-north-mini-code-1p0 | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-orpheus-3b-0p1-ft | functional | llama.cpp | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 (restricted) | yes | no | no |
| llamacpp-phi4-mini-instruct | functional | llama.cpp | gpu | 1 x 10 GiB | yes (gpu) | MIT | yes | no | no |
| llamacpp-phi4-reasoning-plus | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | MIT | yes | no | no |
| llamacpp-qwen25-72b-instruct | functional | llama.cpp | gpu | 1 x 48 GiB | no | Qwen License (restricted) | yes | no | no |
| llamacpp-qwen25-coder-32b-instruct | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-235b-a22b-instruct-2507 | functional | llama.cpp | gpu | 2 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-235b-a22b-thinking-2507 | functional | llama.cpp | gpu | 2 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-32b | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-coder-30b-a3b-instruct | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-coder-480b-a35b-instruct | functional | llama.cpp | gpu | 4 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-coder-next | functional | llama.cpp | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-embedding-8b | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen3-next-80b-a3b-instruct | functional | llama.cpp | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-qwen35-122b-a10b | functional | llama.cpp | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-qwen35-397b-a17b | functional | llama.cpp | gpu | 4 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-qwen35-9b | functional | llama.cpp | gpu | 1 x 15 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen36-27b | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen36-35b-a3b | functional | llama.cpp | gpu | 1 x 24 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen38-27b | functional | llama.cpp | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| llamacpp-qwen38-2p4t-a95b | functional | llama.cpp | gpu | 8 x 180 GiB | no | qwen3.8-max (restricted) | yes | no | no |
| llamacpp-qwen38-flash-next | functional | llama.cpp | gpu | 2 x 80 GiB | no | Qwen Community License 1.0 (restricted) | yes | no | no |
| llamacpp-step-3p7-flash | functional | llama.cpp | gpu | 2 x 80 GiB | no | Apache-2.0 | yes | no | no |
| llamacpp-trinity-large-thinking | functional | llama.cpp | gpu | 4 x 80 GiB | no | OpenMDW-1.1 (restricted) | yes | no | no |
| ltx-video-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | custom licence (restricted) | yes | no | bench |
| ltx-video-diffusers | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | custom licence (restricted) | yes | no | no |
| mochi-1-preview-bench | benchmark | pytorch | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| mochi-1-preview-diffusers | functional | pytorch | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| olmocr-2-7b-bench | benchmark | pytorch | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| olmocr-2-7b-pytorch | functional | pytorch | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| paddleocr-vl-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| paddleocr-vl-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| parakeet-tdt-0p6b-v3-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | CC-BY-4.0 | yes | no | bench |
| parakeet-tdt-0p6b-v3-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | CC-BY-4.0 | yes | no | no |
| pixtral-12b-2409-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | Apache-2.0 | yes | no | bench |
| pixtral-12b-2409-pytorch | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| qwen-image-2p1-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | Qwen Research License (restricted) | yes | no | bench |
| qwen-image-2p1-diffusers | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | Qwen Research License (restricted) | yes | no | no |
| qwen25-vl-72b-bench | benchmark | pytorch | gpu | 4 x 44 GiB | n/a | Qwen License (restricted) | yes | no | no |
| qwen25-vl-72b-pytorch | functional | pytorch | gpu | 4 x 44 GiB | no | Qwen License (restricted) | yes | no | no |
| qwen3-asr-1p7b-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| qwen3-asr-1p7b-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| qwen3-reranker-0p6b-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| qwen3-reranker-0p6b-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| qwen3-reranker-4b-bench | benchmark | pytorch | gpu | 1 x 15 GiB | n/a | Apache-2.0 | yes | no | bench |
| qwen3-reranker-4b-pytorch | functional | pytorch | gpu | 1 x 15 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| qwen3-reranker-8b-bench | benchmark | pytorch | gpu | 1 x 22 GiB | n/a | Apache-2.0 | yes | no | bench |
| qwen3-reranker-8b-pytorch | functional | pytorch | gpu | 1 x 22 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| qwen3-vl-235b-a22b-bench | benchmark | pytorch | gpu | 8 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| qwen3-vl-235b-a22b-pytorch | functional | pytorch | gpu | 8 x 80 GiB | no | Apache-2.0 | yes | no | no |
| qwen3-vl-32b-bench | benchmark | pytorch | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| qwen3-vl-32b-pytorch | functional | pytorch | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| sam2p1-hiera-large-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| sam2p1-hiera-large-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| siglip2-giant-opt-patch16-384-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| siglip2-giant-opt-patch16-384-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| siglip2-so400m-patch16-512-bench | benchmark | pytorch | gpu | 1 x 10 GiB | n/a | Apache-2.0 | yes | no | bench |
| siglip2-so400m-patch16-512-pytorch | functional | pytorch | gpu | 1 x 10 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| vllm-bench-deepseek-v4p1-flash | benchmark | vllm | gpu | 8 x 80 GiB | n/a | MIT | yes | no | no |
| vllm-bench-gpt-oss-120b-vllm | benchmark | vllm | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| vllm-bench-hy4-preview | benchmark | vllm | gpu | 8 x 256 GiB | n/a | Apache-2.0 | yes | no | no |
| vllm-bench-pixtral-large-instruct-2411 | benchmark | vllm | gpu | 4 x 80 GiB | n/a | Mistral Research License (restricted) | yes | no | no |
| vllm-deepseek-v4p1-flash | functional | vllm | gpu | 8 x 80 GiB | no | MIT | yes | no | no |
| vllm-gpt-oss-120b-vllm | functional | vllm | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| vllm-hy4-preview | functional | vllm | gpu | 8 x 256 GiB | no | Apache-2.0 | yes | no | no |
| vllm-pixtral-large-instruct-2411 | functional | vllm | gpu | 4 x 80 GiB | no | Mistral Research License (restricted) | yes | no | no |
| wan21-t2v-14b-bench | benchmark | pytorch | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| wan21-t2v-14b-diffusers | functional | pytorch | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| wan21-t2v-1p3b-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | Apache-2.0 | yes | no | bench |
| wan21-t2v-1p3b-diffusers | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| wan22-t2v-a14b-bench | benchmark | pytorch | gpu | 1 x 80 GiB | n/a | Apache-2.0 | yes | no | no |
| wan22-t2v-a14b-diffusers | functional | pytorch | gpu | 1 x 80 GiB | no | Apache-2.0 | yes | no | no |
| wan22-ti2v-5b-bench | benchmark | pytorch | gpu | 1 x 40 GiB | n/a | Apache-2.0 | yes | no | bench |
| wan22-ti2v-5b-diffusers | functional | pytorch | gpu | 1 x 40 GiB | yes (gpu) | Apache-2.0 | yes | no | no |
| whisper-cpp-bench-large-v3-turbo | benchmark | whisper.cpp | gpu | 1 x 10 GiB | n/a | MIT | yes | no | bench |
| whisper-cpp-large-v3-turbo | functional | whisper.cpp | gpu | 1 x 10 GiB | yes (gpu) | MIT | yes | no | no |

#### Totals

| Family | Workloads | Functional | Benchmark | Reference recorded | Model sha256-pinned | Sim-validated | GPU-validated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llamacpp | 51 | 29 | 22 | 29 | 45 | 0 | 22 |
| pretrained | 52 | 28 | 24 | 28 | 52 | 0 | 24 |
| arch | 6 | 5 | 1 | 5 | 0 | 5 | 1 |
| lib | 6 | 5 | 1 | 5 | 0 | 5 | 1 |
| train | 3 | 2 | 1 | 2 | 0 | 2 | 1 |
| other | 23 | 15 | 8 | 15 | 19 | 3 | 13 |
| catalog | 258 | 129 | 129 | 69 | 258 | 0 | 69 |
| **all** | 399 | 213 | 186 | 153 | 374 | 15 | 131 |

<!-- coverage:end -->


