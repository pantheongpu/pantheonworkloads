# Models

Weights are never committed; manifests point at them. A licence is recorded only after reading the model's
own card or repository. When that was not possible the manifest says `UNVERIFIED` in `model.licence` and
`bin/pw validate` warns.

**Status of this list (updated 2026-10-08):** the first real-GPU run (an AWS g5.2xlarge with an NVIDIA A10G; `docs/first-gpu-run-a10g.md`)
had open internet, so the models that were blocked before are now resolved: model cards and LICENSE files were read from the
repositories, revisions pinned to commit hashes, weights checksummed (sha256 in each workload's `model.sha256` or manifest), and the
references recorded **on that GPU** (not the CPU; each commit message says so). Resolved: `gpt2-small-pytorch`
(MIT), `bert-base-uncased-pytorch` (Apache-2.0), `whisper-cpp-tiny-en` and its bench twin (MIT), `smollm2-135m-ollama`,
`vllm-greedy-smollm2-135m`, `vllm-bench-throughput`, `llamacpp-smollm2-135m` and its bench twin (all Apache-2.0). Added later the same day on a second g5.2xlarge (A10G): `llamacpp-qwen25-0p5b` (+ bench) and `llamacpp-mistral-7b-v03` (+ bench), all Apache-2.0 as read from the cards. Still open: the sections below that still say "unverified" for text
older than this note, and everything on the simulator or AMD. GitHub and PyPI were always reachable; Hugging Face was not from the host that wrote the first version.

The Tier 1 models added on a third g5.2xlarge (Llama 3.2 1B/3B, Gemma 4 E4B, gpt-oss-20b, DeepSeek-R1-Distill-Qwen 1.5B/7B, Whisper large-v3, SDXL base, BLIP-2) are in "Tier 1 models" below, followed by the vision-language models added on 2026-10-09 (Qwen2.5-VL-7B, Qwen3-VL-4B, SmolVLM2-2.2B), with the ones that are blocked or too big for one A10G. Seven more llama.cpp models (Qwen3-30B-A3B, Qwen3-8B, Phi-4, Granite 4.0 H-Small, Mistral-Small-3.2-24B, OLMo-2-7B, Qwen3-Embedding-4B) were added on 2026-10-09 in "More llama.cpp architectures" below.

Each section below is self-contained and covers one workload family.

## llama.cpp workloads

| Model | Workload | Licence | State |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct (GGUF, `bartowski/SmolLM2-135M-Instruct-GGUF` Q8_0) | `llamacpp-smollm2-135m`, `llamacpp-bench-smollm2-135m` | Apache-2.0 (GGUF repo card and base model card, 2026-10-08) | pinned: HF commit `09816acd...`, file sha256 `5a139571...`; reference recorded on an A10G. A third-party conversion: no official GGUF repo of that name exists |
| Qwen2.5-0.5B-Instruct (official GGUF, `Qwen/Qwen2.5-0.5B-Instruct-GGUF` Q8_0) | `llamacpp-qwen25-0p5b`, `llamacpp-bench-qwen25-0p5b` | Apache-2.0, read 2026-10-08 from the GGUF repo card and its LICENSE file (Apache License 2.0 text), and from the base card `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760...` and its LICENSE file; not gated | pinned: GGUF repo commit `9217f5db...`, file sha256 `ca59ca7f...`; reference recorded on an A10G, identical in 4 consecutive GPU runs |
| TinyLlama-1.1B, SmolLM2-360M | not added | not read | candidates (expected Apache-2.0); add after the cards are read |
| Mistral-7B-v0.3 (GGUF, `mradermacher/Mistral-7B-v0.3-GGUF` Q8_0, 7.7 GB) | `llamacpp-mistral-7b-v03` (GPU only), `llamacpp-bench-mistral-7b-v03` | Apache-2.0, read 2026-10-08 from the front matter of the base card `mistralai/Mistral-7B-v0.3` @ `caa1feb0...` and of the GGUF repo card (the official repo has no LICENSE file; neither repo is gated, although the official card carries an `extra_gated_description`) | pinned: GGUF repo commit `76804246...`, file sha256 `1a69d4e4...`; there is no official GGUF, so this is a third-party conversion (static quants, `base_model: mistralai/Mistral-7B-v0.3`). Reference recorded on an A10G, identical in 4 consecutive GPU runs. Unverified: how the conversion was produced beyond its card |
| Llama family (Meta) | `llamacpp-llama32-1b-instruct`, `llamacpp-llama32-3b-instruct` (+ bench twins), `restricted: true` | Llama 3.2 Community Licence (official repos gated) | see "Tier 1 models" below: community conversion whose card carries the licence text; Llama 3.2 Vision 11B blocked |

To pin a model: read its card, put the licence and the commit hash in the manifest, download the file,
`sha256sum` it into `workloads/<name>/model.sha256`, and use the commit-pinned `resolve/<hash>/...` URL.

## PyTorch workloads (`pytorch-microsuite`, `gpt2-small-pytorch`, `bert-base-uncased-pytorch`, `resnet18-randinit-pytorch`)

Status (2026-10-08): **run on an NVIDIA A10G**; references for GPT-2 and BERT were recorded there with pinned revisions and licences read from the repositories (see `docs/first-gpu-run-a10g.md`). The paragraph below describes the state before that run. The machine they were written on could reach PyPI but not
`download.pytorch.org` or `huggingface.co` (the egress proxy answered 403), and PyPI's Linux torch is the
~870 MB CUDA build with several GB of NVIDIA dependencies, so PyTorch was not installed, nothing was
executed, no `reference.json` exists, and every run reports SKIP until one is recorded on a trusted target
(`bin/pw record <workload> --target cpu`). The scripts are syntax-checked only. The tolerances are reasoned from
floating-point behaviour, not measured: check them against the first cpu-versus-gpu recording.

| Workload | What it runs | Model, licence, revision |
| --- | --- | --- |
| `pytorch-microsuite` | 13 ops (fp32/fp16/bf16 matmul, bmm, linear+GELU, conv2d x2, SDPA, hand-written attention, softmax, layernorm, reductions, cumsum) plus topk/argsort/argmax ids, on seeded CPU-generated inputs | none |
| `gpt2-small-pytorch` | GPT-2 124M, 5 greedy tokens, fp32, eager attention, KV cache | `openai-community/gpt2` @ `607a30d7...`; MIT (card); `model.safetensors` sha256 pinned |
| `bert-base-uncased-pytorch` | BERT-base forward, top-5 masked-word prediction, SDPA attention | `google-bert/bert-base-uncased` @ `86b5e093...`; Apache-2.0 (LICENSE file); `model.safetensors` sha256 pinned |
| `resnet18-randinit-pytorch` | torchvision ResNet-18 with seeded random weights, 4 images 64x64 | no weights; code BSD-3-Clause |

Licences:
- torchvision code: BSD-3-Clause, read from `github.com/pytorch/vision/blob/main/LICENSE`. Its pretrained
  ImageNet weights were not checked, so the vision workload uses random weights (a deterministic network is all
  a functional check needs). PyPI metadata lists torch as `Apache-2.0 AND ... BSD-3-Clause ...` and
  transformers as Apache 2.0 (code only).
- GPT-2 and BERT: the model cards could not be read here, so their manifests say **UNVERIFIED** instead of
  guessing. Before relying on them: read both cards, fill in `model.licence`, and pin `model.revision` to the
  commit hash of the first download (until then `PW_HF_REVISION` selects one). Each run's `detail` also prints
  the licence the Hub's card metadata declares at that moment.

How they run (shared code in `workloads/_pytorch/`, which has no manifest so `pw` ignores it):
- `env.sh` finds a Python with the right PyTorch, else creates a venv (`PW_VENV_ROOT`, default
  `~/.cache/pantheonworkloads/venvs/<flavor>`) from `requirements-torch.txt` and `requirements-common.txt`, else
  exits 77. Flavors: `cpu` (cpu target), `cu130` (gpu on NVIDIA, `sim:nvidia/*`), `rocm` (gpu on AMD, `sim:amd/*`);
  `PW_TORCH_FLAVOR`, `PW_TORCH_PYTHON`, `PW_NO_INSTALL=1` and `PW_TORCH_INDEX_{CPU,CU130,ROCM}` override. It
  also reuses pantheonsim's own venvs (`~/.local/share/torch-cu13*`, `torch-rocm*`, `VGPU_TORCH_CUDA_PYTHON`,
  `VGPU_TORCH_PYTHON`) when they import what the workload needs. The pins (torch 2.14.1, torchvision 0.29.1,
  transformers 5.19.0) are existing PyPI releases, but their cu130 and ROCm builds were not checked against the PyTorch
  index, and the ROCm index name (`rocm7.1`) is a guess; pantheonsim's own venvs are the tested route there.
- `sim:nvidia/<gpu>`: `$PANTHEONSIM_DIR/build/vgpu run --gpu nvidia/<gpu> --preload <python> main.py`, as
  pantheonsim's `nvidia/tests/e2e/run_pytorch.sh` does (CUDA 13 wheels).
- `sim:amd/<gpu>`: the tree-of-links trick of `amd/tests/e2e/run_pytorch.sh` (the simulator's `libamdhip64` and
  `librocm_smi64` replace PyTorch's), with `VGPU_GPU`, `VGPU_DEVICE_COUNT=1`, `VGPU_MEMORY_RAM_MB=4096`.
- Simulated runs are capped by `systemd-run -p MemoryMax=10G` where available (`VGPU_TORCH_MEMORY_MAX`), and any
  `VirtualGPU error [` line in the log fails the run, both as pantheonsim does. Triton/Inductor caches are per run.
- Inputs, seeds and model weights come from the CPU, the device only computes; TF32 and cudnn autotune are off.
  ids are compared exactly (strings in `output`), floats by tolerance; runs fail if top-k scores are within 1e-3,
  so a reference is never a near-tie. `bin/pw` now compares object outputs key by key with per-key tolerances.
- Weights go to `$PW_CACHE/hf` (default `~/.cache/pantheonworkloads/hf`), never into the repository. A failed
  download exits 77.
- Metrics appear only for `--target gpu` (never cpu or sim): `matmul_{fp32,tf32,fp16,bf16}_tflops`,
  `conv2d_fp16_tflops`, `sdpa_causal_fp16_tflops`, `copy_gb_s`, `decode_tokens_per_s`,
  `forward_sequences_per_s_b32_s128`, `images_per_s_b64_224`. `PW_NO_BENCH=1` turns them off.
- Not exercised: torch.compile, multi-GPU, half-precision model inference, quantised models.

## vLLM and GPT training (`vllm-*`, `gpt-train-*`)

### What exists

| Workload | Kind | Targets | What it records |
| --- | --- | --- | --- |
| `vllm-greedy-smollm2-135m` | functional | `gpu`, `sim:amd/*` (not the RX 6900 XT) | the text of 8 greedy tokens, exact match |
| `vllm-bench-throughput` | benchmark | `gpu` | requests/s, total tokens/s, output tokens/s from `vllm bench throughput` |
| `gpt-train-fp32` | functional | `cpu`, `gpu`, `sim:nvidia/*`, `sim:amd/*` | 20 training losses, abs tolerance 1e-3 |
| `gpt-train-bf16` | functional | same; exits 77 where bf16 is unsupported | 20 losses under bf16 autocast, abs tolerance 2e-2 |
| `gpt-train-bench` | benchmark | `gpu` (`cpu` only to test the workload) | fp32 and bf16 steps/s and tokens/s |

### What has and has not run

- **Ran on a GPU (2026-10-08, A10G):** `gpt-train-fp32`, `gpt-train-bf16` (CPU references passed at their tolerances), `gpt-train-bench`, `vllm-greedy-smollm2-135m` (reference recorded there, vLLM 0.30.0, `VLLM_USE_FLASHINFER_SAMPLER=0`), `vllm-bench-throughput`. The remark below about vLLM not having run is out of date.
- **Ran earlier:** `gpt-train-fp32` and `gpt-train-bf16` on the `cpu` target (CPU-only PyTorch 2.13.0 from
  conda-forge, 4 shared cores). Their `reference.json` files were recorded there, with `bin/pw record
  --target cpu`; they have not been checked on any GPU or simulator. `gpt-train-bench` ran on `cpu` as a
  smoke test only (6 steps; nothing recorded: CPU numbers from a shared machine mean nothing).
- **Not run:** everything vLLM. vLLM needs a CUDA or ROCm stack and none is available where this was
  written, so the workloads are written against vLLM v0.30.0's source and tested only for their glue
  (`tests/test_workloads.py`, with a fake `torch` and `vllm`: the workloads exit 77 with a reason when
  torch, vLLM, a GPU or pantheonsim is missing, and the benchmark's output parsing works). No vLLM
  reference or benchmark exists. Nothing ran on `gpu`, `sim:nvidia/*` or `sim:amd/*`.

### vLLM

- **Source and licence:** github.com/vllm-project/vllm, tag `v0.30.0` (commit
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`): Apache-2.0, read from that tag's `LICENSE` and
  `pyproject.toml` (`license = "Apache-2.0"`). 0.30.0 is the version pantheonsim's nightly runs
  (`amd/tests/vllm/install.sh`: `vllm==0.30.0+rocm723`); v0.31.0 also exists.
- **Benchmark tool at that tag:** `vllm bench throughput` (`vllm/benchmarks/throughput.py`). The old
  `benchmarks/benchmark_throughput.py` is a stub that only prints the new command. `--output-json`
  gives `elapsed_time`, `num_requests`, `total_num_tokens`, `requests_per_second`,
  `tokens_per_second`; output-token throughput is only on its printed `Throughput:` line, which the
  workload parses. For the random dataset the `--random-input-len/--random-output-len` options win and
  the older `--input-len/--output-len` are ignored: the workload uses the former.
- **Simulated GPUs:** pantheonsim runs vLLM on simulated **AMD** GPUs only (`amd/tests/e2e/run_vllm_amd.sh`,
  ctest `amd_vllm`, nightly): eager mode, one short sequence, fp16, PyTorch's ROCm build linked to the
  simulator's `libamdhip64`, ROCm SMI and AMD SMI. `workloads/_shared/torch_env.sh` repeats that
  setup (`VGPU_VLLM_PYTHON`, `PANTHEONSIM_DIR`). There is no NVIDIA vLLM test there, so the workload does
  not claim `sim:nvidia/*`. vLLM ships no gfx1030 kernels, so the RX 6900 XT is excluded as in pantheonsim.
- **Model:** `HuggingFaceTB/SmolLM2-135M`. **Its licence and revision are unverified**: huggingface.co
  was blocked from the machine this was written on, so the model card could not be read. The manifests
  say `UNVERIFIED: expected Apache-2.0` (the expectation comes from the existing `smollm2-135m-ollama`
  manifest, itself not checked against the card) and `revision: null`; `validate` warns. Verify both
  before recording anything. pantheonsim's own test uses `facebook/opt-125m`; it works through
  `PW_MODEL`, but its licence was not checked either, so it is not the default.
- No weights are stored; vLLM downloads the model on first use.

### GPT training

- **Source:** my own ~200-line implementation (`workloads/_shared/gpt_train.py`, this repository's
  Apache-2.0 licence), following the design of Karpathy's nanoGPT, which is MIT-licensed (`LICENSE`
  of github.com/karpathy/nanoGPT at commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, "Copyright (c)
  2022 Andrej Karpathy"). No nanoGPT code is copied; nanoGPT itself needs tiktoken, config files and
  a dataset download. The `baby` preset is nanoGPT's shakespeare-char shape (6 layers, 6 heads, 384 wide).
- **Data:** `workloads/_shared/data/tinytext.txt`, 1,212 characters of original text written for this
  repository, checked in. Nothing is downloaded.
- **What it exercises that inference does not:** backward through matmul, scaled-dot-product attention,
  layer norm, embedding and cross entropy; gradient clipping; AdamW with decoupled weight decay;
  bf16 autocast with fp32 master weights.
- **Determinism:** the model is initialised, and the batches are drawn, on the CPU with seed 1337 and
  then moved to the device, so every target sees identical weights and data. TF32 is off, so fp32 is
  fp32. `torch.use_deterministic_algorithms(True, warn_only=True)` is set.
- **Tolerances** (measured on the CPU backend over 20 steps): fp32 losses vary by 1.7e-6 between 1
  and 4 threads and 1.2e-6 from a float64 run; bf16 losses vary by 2.3e-4 between 1 and 4 threads and
  up to 6.0e-3 from the fp32 run. The bounds are 1e-3 (fp32) and 2e-2 (bf16), well above reordering
  noise and well below the 0.6 the loss falls by, but they are **unverified on a GPU**: a GPU's
  kernels may need them widened, and if so the evidence should be recorded here.
- **bf16 support:** on a GPU target it is `torch.cuda.is_bf16_supported(including_emulation=False)`; where false, `gpt-train-bf16`
  exits 77 and `gpt-train-bench` measures fp32 only. On the CPU target autocast always runs.
- **Simulators:** `sim:nvidia/*` goes through `vgpu run --gpu <profile> --preload python` with
  PyTorch for CUDA 13 (`VGPU_TORCH_CUDA_PYTHON`), `sim:amd/*` through the ROCm link tree
  (`VGPU_TORCH_PYTHON`), both as pantheonsim's `run_pytorch.sh` scripts do, and its notes say a GPT
  training step is within what its PyTorch suites run. This repository has not run them.
- **Tools needed to run on a GPU:** `PW_PYTHON=/path/to/python` with a GPU build of PyTorch (and vLLM
  0.30.0 for the vLLM workloads).

## Speech and LoadGen workloads (whisper.cpp, MLCommons LoadGen)

Added by the `wl-speech-mlperf` branch. Weights and audio are never committed; they are fetched at run
time by `tools/whisper-assets.sh` and checked against pinned checksums.

| Item | Version pinned | Licence and where it was verified | Checksum |
| --- | --- | --- | --- |
| whisper.cpp (runtime) | tag `v1.9.5`, commit `d1be6fde11ac6e0407606b4e42fe72d34add8037` | MIT, `LICENSE` at that commit ("Copyright (c) 2023-2026 The ggml authors") | commit hash checked by `tools/build-whisper-cpp.sh` |
| Whisper tiny.en weights, ggml format (`ggml-tiny.en.bin`, 75 MiB) | `ggerganov/whisper.cpp` on Hugging Face, commit `5359861c739e955e79d9a303bcbc70fb988958b1` (pinned 2026-10-08; card says `license: mit`; sha256 `921e4cf8...`) | The model is OpenAI's: openai/whisper `README.md` at `86098128c0b4f24f0e2aa2994de830614b474227` says "Whisper's code and model weights are released under the MIT License". The ggml conversion's own model card was **not read** (Hugging Face was unreachable where this was written) | sha1 `c78c86eb1a8faa21b369bcd33207cc90d64ae9df` from whisper.cpp `models/README.md` at v1.9.5. **No sha256 recorded yet**: the file could not be downloaded to compute it |
| Test clip `jfk.wav` (11 s, 16 kHz mono, 352078 bytes) | `samples/jfk.wav` at the same commit, fetched from raw.githubusercontent.com | **Not verified.** The whisper.cpp repo states no provenance or licence for the recording (`samples/README.md` only says the folder holds "audio files used for testing"). The words are from a 1961 US presidential address, but the recording's copyright status is unknown to this repo. Therefore it is fetched, not committed | sha256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| `mlcommons-loadgen` (pip) | `6.0.17` | Apache-2.0: `LICENSE.md` of github.com/mlcommons/inference at `3fbc329939999c13d0a7b5e67fb2092287e06047` (the wheel's own metadata licence field is empty) | cp313 manylinux x86_64 wheel sha256 `79090cf79054bad8142d00b59aa81f40dc6e19cc5ef83ccafca9fce6d8ddaabb` (informational; installs are not hash-pinned because wheels differ per platform) |

MLPerf benchmark models and datasets are **not** used by any workload; see `docs/mlperf.md` for what
was found about their terms (several are member-only or gated, several unverified).

## Tier 1 models: one 24 GB card (added 2026-10-08, g5.2xlarge, NVIDIA A10G)

Scope agreed with the user: open models that fit one A10G. Each row records what was **actually read** on 2026-10-08 (Hugging Face API
metadata, card front matter, LICENSE files, fetched with no Hugging Face credentials), whether the repo is gated, and the pin.
Every file's sha256 is the Hub's LFS oid at the pinned commit and is verified on every run. References were recorded on the A10G
(llama.cpp b11447 with CUDA sm_86, `-ngl 99`; PyTorch 2.10.0+cu130, transformers 5.19.0, diffusers 0.41.0) and checked in 3 further
consecutive runs (all identical). `restricted: true` workloads (docs/manifest.md) are skipped by `bin/pw list --default`,
`bin/pw matrix` and every default selection; name them to run them.

| Model | Workloads | Licence as read | Restricted | Pin |
| --- | --- | --- | --- | --- |
| Llama 3.2 1B-Instruct, Q8_0 (1.3 GB) | `llamacpp-llama32-1b-instruct`, `llamacpp-bench-llama32-1b-instruct` | Llama 3.2 Community Licence. The official `meta-llama/Llama-3.2-1B-Instruct` is **gated (manual)**: only its API metadata (`license: llama3.2`) was readable, its `LICENSE.txt` was not. The text was read from the card front matter (`extra_gated_prompt`, the full agreement incl. the Acceptable Use Policy, the "Built with Llama" attribution and the 700-million-MAU clause) of the community conversion `bartowski/Llama-3.2-1B-Instruct-GGUF` (not gated) | **yes** | GGUF repo `067b946c...`, file sha256 `432f310a...`; third-party conversion |
| Llama 3.2 3B-Instruct, Q8_0 (3.4 GB) | `llamacpp-llama32-3b-instruct`, `llamacpp-bench-llama32-3b-instruct` | as above, `bartowski/Llama-3.2-3B-Instruct-GGUF` card (same text); official repo gated | **yes** | GGUF repo `5ab33fa9...`, file sha256 `b5607b50...` |
| Gemma 4 E4B-it, QAT Q4_0 (5.2 GB) | `llamacpp-gemma4-e4b-it`, `llamacpp-bench-gemma4-e4b-it` | Apache-2.0: front matter of the official GGUF repo `google/gemma-4-E4B-it-qat-q4_0-gguf` (`license: apache-2.0`, `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`, header "License: Apache 2.0") and of `google/gemma-4-E4B-it`; neither gated, neither has a LICENSE file. (Gemma 3 and earlier use the Gemma Terms of Use; Gemma 4 does not.) | no | GGUF repo `4b4a2c1d...`, file sha256 `676c3507...`; Google's own conversion |
| gpt-oss-20b, MXFP4 (12.1 GB) | `llamacpp-gpt-oss-20b`, `llamacpp-bench-gpt-oss-20b` | Apache-2.0, verified: the official `openai/gpt-oss-20b` @ `6cee5e81` has a `LICENSE` file (Apache License 2.0 text) and `license: apache-2.0` front matter; not gated. Its `USAGE_POLICY` is one sentence asking users to follow applicable law. The GGUF repo `ggml-org/gpt-oss-20b-GGUF` says `license: apache-2.0` | no | GGUF repo `ef9b12f2...`, file sha256 `27cd6c43...`; ggml-org automatic conversion. Fits 24 GB with all layers offloaded |
| DeepSeek-R1-Distill-Qwen-7B, Q8_0 (8.1 GB) | `llamacpp-deepseek-r1-distill-qwen-7b`, `llamacpp-bench-...` | MIT: official `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` @ `916b56a4` LICENSE (MIT, "Copyright (c) 2023 DeepSeek") and `license: mit`; the GGUF repo `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF` carries the same LICENSE file | no | GGUF repo `36100415...`, file sha256 `cae89f90...` |
| DeepSeek-R1-Distill-Qwen-1.5B, Q8_0 (1.9 GB) | `llamacpp-deepseek-r1-distill-qwen-1p5b`, `llamacpp-bench-...` | MIT: official repo @ `ad9f0ae0` LICENSE and front matter. The bartowski GGUF repo has **no LICENSE file and no license field**, so its licence rests on the official repo | no | GGUF repo `9cc28b17...`, file sha256 `166baa90...` |
| Whisper large-v3 via whisper.cpp (3.1 GB) | `whisper-cpp-large-v3`, `whisper-cpp-bench-large-v3` | MIT per `ggerganov/whisper.cpp` card front matter. **The two cards disagree**: `openai/whisper-large-v3` @ `06f233fe` says `license: apache-2.0` (no LICENSE file); the `openai/whisper` code repo is MIT. Both permissive | no | `ggerganov/whisper.cpp` `5359861c...` (same pin as tiny.en), `ggml-large-v3.bin` sha256 `64d182b4...` |
| Stable Diffusion XL base 1.0, fp16 | `sdxl-base-diffusers`, `sdxl-base-bench` | CreativeML Open RAIL++-M: card `license: openrail++` and `LICENSE.md` of the official `stabilityai/stable-diffusion-xl-base-1.0` @ `46216598` (use-based restrictions, Attachment A); not gated | **yes** | repo `46216598...`; four fp16 safetensors (unet, text_encoder, text_encoder_2, vae; 6.9 GB) pinned in `model.sha256` |
| BLIP-2 OPT-2.7b, fp16 | `blip2-opt-2p7b-pytorch`, `blip2-opt-2p7b-bench` | Card of `Salesforce/blip2-opt-2.7b` @ `59a1ef6c` says `license: mit` (not gated, no LICENSE file). **But** the checkpoint embeds `facebook/opt-2.7b` @ `905a4b60`, whose card says `license: other` and `commercial: false` (Meta's OPT licence, non-commercial research; its text is not a file in that repo, so only the front matter was read). Treated as non-permissive | **yes** | repo `59a1ef6c...`; two safetensors shards pinned; image: scikit-image `astronaut.png` (NASA, public domain; scikit-image v0.22.0 commit `441fe68b`), sha256 pinned, fetched at run time |
| Qwen2.5-VL-7B-Instruct, fp16 (16.6 GB bf16 on disk, 16.7 GB peak on the card) | `qwen25-vl-7b-pytorch`, `qwen25-vl-7b-bench` | Apache-2.0: front matter (`license: apache-2.0`) of the official `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898`, read 2026-10-09; not gated; the repo has **no LICENSE file**, so the card is the only source. The 3B sibling uses `qwen-research` and was not used | no | repo `cc594898...`; five safetensors shards pinned in `model.sha256` (Hub LFS oids) |
| Qwen3-VL-4B-Instruct, fp16 (8.9 GB bf16 on disk, 9.0 GB peak) | `qwen3-vl-4b-pytorch`, `qwen3-vl-4b-bench` | Apache-2.0: front matter of the official `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec`, read 2026-10-09; not gated; no LICENSE file | no | repo `ebb281ec...`; two shards pinned |
| SmolVLM2-2.2B-Instruct, fp16 (9.0 GB fp32 on disk, 5.0 GB peak) | `smolvlm2-2p2b-pytorch`, `smolvlm2-2p2b-bench` | Apache-2.0: front matter, the card's summary ("License: Apache 2.0") and its License section, official `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb53`, read 2026-10-09; not gated; no LICENSE file | no | repo `482adb53...`; two shards pinned; needs `num2words` (`requirements-vlm.txt`) |

Blocked or deferred (nothing here was run):

| Model | Why |
| --- | --- |
| Stable Diffusion 3 medium | `stabilityai/stable-diffusion-3-medium-diffusers` is gated (`gated: auto`; login and acceptance of Stability's community licence needed; card names `stabilityai-nc-research-community`, the LICENSE file is behind the gate). No Hugging Face credentials were available and no ungated copy whose card carries the licence text was looked for. Blocked |
| Llama 3.2 Vision 11B | The official `meta-llama/Llama-3.2-11B-Vision-Instruct` is gated (manual). The only ungated route found is `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` (about 7.2 GB, would fit in 4-bit), but its card only links the Llama 3.2 Community Licence instead of carrying the text, which fails the rule for community conversions. In fp16 the weights alone are about 21 GB, which does not leave room for the vision activations on one A10G; not tried. Recorded as: deferred, blocked on a readable licence, does not fit one A10G in fp16 |

Out of scope for one A10G (deferred, needs a bigger rig). Nothing larger than one A10G was launched. **Update 2026-10-09:** model size no longer limits what
the repository holds. These models, and many more, are in [`docs/model-registry.md`](docs/model-registry.md) as workloads that are written from Hub metadata, never
run, and gated by a `requires:` block that SKIPs them on a host without the GPUs. The table below is the state before that and is kept as it was:

| Model | Reason |
| --- | --- |
| DeepSeek-V3 / R1 (full) | 671B-parameter MoE: hundreds of GB even at 4-bit (not re-checked here) |
| Llama 4 (Scout, Maverick) | large MoE models, well beyond 24 GB even at 4-bit (not sized here); also gated (Scout: `gated: manual`, `license: other` / `llama4`) |
| Large Qwen, GLM, Kimi | tens to hundreds of billions of parameters |
| gpt-oss-120b | the MXFP4 GGUF (`ggml-org/gpt-oss-120b-GGUF`) is 63.4 GB (read from the Hub): needs an 80 GB card |

Notes on the results (details and bench numbers: `docs/benchmarks.md`):

- The three-token greedy references of the instruct models are the short continuations of "The capital of France is" (" Paris. The", etc.);
  Gemma 4 produces " Paris." and then its end-of-text token, which `llama-completion` prints as `[end of text]`, so that string is part of
  its reference. Longer decodes were not used (documented floating-point divergence between backends).
- SDXL: the functional output is image statistics (per-channel mean/std, an 8x8 grid of block means, mean absolute gradients) with an abs tolerance
  of 0.02, never pixels. Measured on the A10G: pixels were bit-identical in every run with default kernels (the recording run, 3 further
  processes, and 2 generations in an analysis process; uint8 sha256 `e137244f2217...`), a forced math SDPA kernel moved the statistics by at most
  0.0043, a different seed by up to 0.52. Other cards were not measured, so the tolerance is a reasoned bound, not a cross-card measurement.
- BLIP-2: the caption of astronaut.png is "a woman in an orange space suit with a space helmet" (greedy, fp16); the smallest top-1/top-2 logit gap
  over its 12 tokens is 0.141, above the 0.1 floor the workload enforces.
- Vision-language models (added 2026-10-09, a fifth g5.2xlarge, one A10G; torch 2.14.1+cu130, transformers 5.19.0, so the shared pins did not move; the extra pip
  requirements are in `workloads/_pytorch/requirements-vlm.txt`): chosen in place of SD3 and Llama 3.2 Vision (rows above, still blocked). Each workload feeds the
  same public-domain image (scikit-image `astronaut.png`, NASA, sha256 pinned, fetched at run time) and one fixed prompt through the chat template, decodes greedily in fp16,
  and compares the text and token ids exactly. A decoding step whose top-1 and top-2 logits are within 0.1 fails the run, so a reference is never a near tie. Measured:
  Qwen2.5-VL-7B "The image shows a person in an orange space suit with a NASA patch, standing in front of an American flag and a model of a space shuttle." (31 tokens,
  smallest gap 0.188); Qwen3-VL-4B "A smiling female astronaut in an orange flight suit with mission patches stands beside an American flag and a model of a space shuttle,
  holding a helmet." (29 tokens, 0.156); SmolVLM2-2.2B "A space shuttle model." for "What is the person holding or standing next to? Answer in one sentence." (5 tokens, 0.359).
  Each reference was recorded on the A10G and reproduced in 3 further runs. SmolVLM2 needed a different prompt: free-form captions of this image hit gaps of 0.016 to 0.07
  (one fp16 step), which the floor rightly rejects, so its output is short and tests the image path and a few decoding steps, not long-text stability. The `{wer: X}`
  tolerance of `bin/pw` was not needed. Memory (torch peak allocated, batch 1, this image): 16.7 GB for the 7B, 9.0 GB for the 4B, 5.0 GB for SmolVLM2; the 7B in fp16 fits one
  24 GB A10G with room for the activations of one image, a larger image or batch would not be checked. Qwen3-VL-8B (apache-2.0 on its card, 4 shards, about 17 GB) was not added:
  it would be a second Qwen3-VL point beside the 4B. The CPU target is a documented SKIP (`targets: [gpu]`); the simulator targets are not listed either, nothing was run on pantheonsim.

## More llama.cpp architectures: one 24 GB card (added 2026-10-09, g5.2xlarge, NVIDIA A10G)

Scope agreed with the user: open LLMs that fit one A10G, as llama.cpp functional and `llama-bench` workloads, to widen the architecture and quantisation
coverage for the simulator and GPU comparisons (the llama.cpp pin did not move: b11447 supports all of them). Everything was read on 2026-10-09 with no Hugging Face
credentials: API metadata (`gated` is false for every repo below), card front matter, and LICENSE files where a repo has one. Weights are never committed; every file's
sha256 is the Hub's LFS oid at the pinned commit and is verified on every run. References were recorded on the A10G (llama.cpp b11447, CUDA sm_86, `-ngl 99`, 3 greedy
tokens of "The capital of France is", no chat template) and reproduced by 3 further consecutive runs, all PASS (exact compare); bench medians of 5 repeats are in
`docs/benchmarks.md`. None is `restricted`: all seven are Apache-2.0 or MIT.

| Model, quantisation (file size) | Workloads | Licence as read | Pin | Architecture / what it adds | Peak GPU memory of a decode |
| --- | --- | --- | --- | --- | ---: |
| Qwen3-30B-A3B, Q4_K_M (18.6 GB) | `llamacpp-qwen3-30b-a3b`, `llamacpp-bench-qwen3-30b-a3b` | Apache-2.0: LICENSE file (Apache License 2.0 text) and front matter of the official GGUF repo `Qwen/Qwen3-30B-A3B-GGUF` and of the base `Qwen/Qwen3-30B-A3B` | GGUF repo `e4d4bafd...`, file sha256 `0d003f66...`; base `ad44e777...`; Qwen's own GGUF | mixture of experts (Qwen3-MoE), K-quant | 17.4 GiB |
| Qwen3-8B, Q6_K (6.7 GB) | `llamacpp-qwen3-8b`, `llamacpp-bench-qwen3-8b` | Apache-2.0: LICENSE file and front matter of `Qwen/Qwen3-8B-GGUF` and `Qwen/Qwen3-8B` | GGUF repo `7c41481f...`, file sha256 `cb042ccd...`; base `b968826d...`; Qwen's own GGUF | dense Qwen3 (QK-norm), Q6_K | 6.0 GiB |
| Phi-4 (14B), Q4_K (9.1 GB) | `llamacpp-phi4-14b`, `llamacpp-bench-phi4-14b` | MIT: LICENSE file (MIT License, Copyright (c) Microsoft Corporation) and front matter of `microsoft/phi-4-gguf` and of `microsoft/phi-4` | GGUF repo `6edc2ef6...`, file sha256 `5652b9be...`; base `2db69c1c...`; Microsoft's own GGUF | Phi-3-family dense model, K-quant | 8.6 GiB |
| Granite 4.0 H-Small, Q4_K_M (19.5 GB) | `llamacpp-granite40-h-small`, `llamacpp-bench-granite40-h-small` | Apache-2.0: front matter of `ibm-granite/granite-4.0-h-small-GGUF` and of the base `ibm-granite/granite-4.0-h-small` (card line "License: Apache 2.0"); neither repo has a LICENSE file | GGUF repo `65220950...`, file sha256 `23c6019f...`; base `b8c0982b...`; IBM's own GGUF | hybrid: Mamba-2 state-space layers + attention + mixture of experts (32B total) | 18.7 GiB |
| Mistral-Small-3.2-24B-Instruct-2506 (text part), Q4_K_M (14.3 GB) | `llamacpp-mistral-small-32-24b`, `llamacpp-bench-mistral-small-32-24b` | Apache-2.0: front matter of `bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF` (`license: apache-2.0`, `base_model_relation: quantized`) and of the official `mistralai/Mistral-Small-3.2-24B-Instruct-2506`; no LICENSE file in either; both have an `extra_gated_description` but anonymous access works | GGUF repo `b3592d09...`, file sha256 `80f5bda6...`; base `95a6d26c...`; **third-party conversion** (bartowski, imatrix): the official `mistralai/...-GGUF` repo answers 401 without credentials | dense 24B (Mistral-Small architecture), imatrix K-quant | 13.5 GiB |
| OLMo-2-1124-7B-Instruct, Q8_0 (7.8 GB) | `llamacpp-olmo2-7b-instruct`, `llamacpp-bench-olmo2-7b-instruct` | Apache-2.0: front matter of Ai2's own `allenai/OLMo-2-1124-7B-Instruct-GGUF` and of `allenai/OLMo-2-1124-7B-Instruct` (card: "OLMo 2 is licensed under the Apache 2.0 license"); no LICENSE file | GGUF repo `410e0069...`, file sha256 `fc410f17...`; base `470b1fba...`; Ai2's own GGUF | OLMo-2 (post-norm blocks, QK-norm) | 7.3 GiB |
| Qwen3-Embedding-4B, Q8_0 (4.3 GB) | `llamacpp-qwen3-embedding-4b`, `llamacpp-bench-qwen3-embedding-4b` | Apache-2.0: front matter of `Qwen/Qwen3-Embedding-4B-GGUF` (no LICENSE file) and of `Qwen/Qwen3-Embedding-4B` | GGUF repo `f4602530...`, file sha256 `b60ae5ce...`; base `5cf2132a...`; Qwen's own GGUF | embedding model: last-token pooling, no sampling | not measured |

Notes:

- Memory is the peak of `nvidia-smi` sampled every 0.5 s during one `-c 256` decode with all layers offloaded (a lower bound of the true peak). The two biggest files (Granite
  19.5 GB, Qwen3-30B-A3B 18.6 GB) fit with 3 to 5 GiB to spare.
- **Near ties and short outputs.** All references are the 3-token greedy decode used by the other llama.cpp workloads: Qwen3-30B-A3B and Qwen3-8B " Paris. The", Phi-4,
  OLMo-2 and Mistral-Small " Paris. It", and Granite 4.0 H-Small " Paris. [end of text]" (it stops after two tokens, like Gemma 4 in the table above, so the reference is shorter
  than 3 tokens of text). Longer decodes were not used: the existing rule is that long greedy decodes diverge on floating-point differences between backends.
- **The embedding workload** (`llamacpp-qwen3-embedding-4b`) runs `llama-embedding` (an example program in llama.cpp, so `tools/llamacpp/build.sh` needs
  `LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS=llama-embedding`; the workload sets both for a first build, and SKIPs on a prefix built without them) on four fixed texts
  (an instruction-format query, "What is the capital of France?", and three passages) with last-token pooling and L2 normalisation. Reference (2560 dimensions): cosine of the query
  with the Paris passage 0.6436, with the Berlin passage 0.4316, with the mitochondria passage 0.1699; passage ranking "1 3 2" (Paris, Berlin, mitochondria), compared exactly.
  The 2560-value vectors were **bit-identical in all 4 runs** on the A10G (maximum element difference 0.0); the compare tolerance (abs 0.005 on cosines rounded to 4 decimals) is
  therefore a reasoned allowance for other cards or simulators, not a measured spread. `llama-bench` also runs on this model (pp512/tg128, benchmarks.md).
- `bin/pw` reports `repo_commit` in the bench records from the checkout it runs in; these records were produced on a rig without a `.git`, so `repo_commit` was filled in afterwards
  with the commit that added the workloads (the tree was identical).
- Not added, with the reason:

| Model | Why not |
| --- | --- |
| Falcon-H1 (7B, 34B; hybrid attention + Mamba-2) | `tiiuae/Falcon-H1-7B-Instruct-GGUF` is not gated but its licence is `license: other` (TII Falcon-LLM licence, not OSI): would be `restricted: true`. Not added because Granite 4.0 H-Small already covers the hybrid state-space design; available if wanted |
| Mistral-Small-3.2-24B official GGUF | `mistralai/Mistral-Small-3.2-24B-Instruct-2506-GGUF` answers 401 without Hugging Face credentials (gated or private); the bartowski conversion above is used instead and labelled third-party |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct-gguf` answers "Invalid username or password" on the API without credentials (gated or absent); the 14B Phi-4 has an ungated official GGUF and was used instead |
| OLMo-2 32B | does not fit one A10G at a useful quantisation (not sized); 7B added |
| Qwen3-Embedding-8B / BGE-M3 | a second embedding point beside the 4B; BGE-M3 has no GGUF in its official repo |

## Other model groups, documented in their own files

- `docs/arch-coverage.md`: random-weight open architectures (no model licence applies; `transformers` is Apache-2.0).
- `docs/lib-coverage.md`: model-free GPU-library kernel checks.
- `docs/llamacpp-synth.md`: synthetic GGUFs (`gguf` package MIT, llama.cpp MIT).
- `docs/pretrained-reachable.md`: pretrained models with licences read from source (the only models in this repo
  whose licences were verified, besides code-only dependencies): Silero VAD, PP-OCRv3, ONNX zoo MNIST / MobileNetV2 /
  BiDAF / BERT-Squad int8, spaCy `en_core_web_sm` / `en_core_web_md` / ru / uk / nb / xx pipelines, all-MiniLM-L6-v2 (via
  a third-party npm package, licence second-hand), GloVe 50d vectors, py3langid, a SentencePiece test model. Also what
  was found about language-model (LLM) weights: none with a readable permissive licence.
- `docs/mlperf.md`: MLPerf Inference feasibility and terms.
