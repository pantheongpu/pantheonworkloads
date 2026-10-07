# Models

Weights are never committed; manifests point at them. A licence is recorded only after reading the model's
own card or repository. When that was not possible the manifest says `UNVERIFIED` in `model.licence` and
`bin/pw validate` warns.

**Status of this list: huggingface.co was not reachable from the machine that wrote it (the proxy answered
403 to CONNECT), so no model card could be read, no file downloaded or hashed, and no revision pinned.**
GitHub was reachable, which is how llama.cpp's MIT licence was read.

Each section below is self-contained and covers one workload family.

## llama.cpp workloads

| Model | Workload | Licence | State |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct (GGUF) | `llamacpp-smollm2-135m`, `llamacpp-bench-smollm2-135m` | UNVERIFIED (expected Apache-2.0) | file name and URL are from memory; revision and sha256 not pinned |
| Mistral-7B-v0.3 | `llamacpp-bench-mistral-7b-v03` (benchmark only, real GPU) | UNVERIFIED (believed Apache-2.0; the original repo may also require accepting terms on Hugging Face) | no default download; you supply the GGUF |
| Qwen2.5-0.5B-Instruct, TinyLlama-1.1B, SmolLM2-360M | not added | not read | candidates (expected Apache-2.0); add after the cards are read |
| Llama family (Meta) | not included: needs the user to accept the licence | Llama community licence (gated) | not fetched; bring your own GGUF with `PW_MODEL_FILE` and add a manifest that cites the licence you accepted |

To pin a model: read its card, put the licence and the commit hash in the manifest, download the file,
`sha256sum` it into `workloads/<name>/model.sha256`, and use the commit-pinned `resolve/<hash>/...` URL.

## PyTorch workloads (`pytorch-microsuite`, `gpt2-small-pytorch`, `bert-base-uncased-pytorch`, `resnet18-randinit-pytorch`)

Status: **written, not run.** The machine they were written on could reach PyPI but not
`download.pytorch.org` or `huggingface.co` (the egress proxy answered 403), and PyPI's Linux torch is the
~870 MB CUDA build with several GB of NVIDIA dependencies, so PyTorch was not installed, nothing was
executed, no `reference.json` exists, and every run reports SKIP until one is recorded on a trusted target
(`bin/pw record <workload> --target cpu`). The scripts are syntax-checked only. The tolerances are reasoned from
floating-point behaviour, not measured: check them against the first cpu-versus-gpu recording.

| Workload | What it runs | Model, licence, revision |
| --- | --- | --- |
| `pytorch-microsuite` | 13 ops (fp32/fp16/bf16 matmul, bmm, linear+GELU, conv2d x2, SDPA, hand-written attention, softmax, layernorm, reductions, cumsum) plus topk/argsort/argmax ids, on seeded CPU-generated inputs | none |
| `gpt2-small-pytorch` | GPT-2 124M, 5 greedy tokens, fp32, eager attention, KV cache | `openai-community/gpt2`; licence **not verified**; revision **unpinned** |
| `bert-base-uncased-pytorch` | BERT-base forward, top-5 masked-word prediction, SDPA attention | `google-bert/bert-base-uncased`; licence **not verified**; revision **unpinned** |
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
  `VGPU_TORCH_PYTHON`) when they import what the workload needs. The pins (torch 2.10.0, torchvision 0.25.0,
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

- **Ran:** `gpt-train-fp32` and `gpt-train-bf16` on the `cpu` target (CPU-only PyTorch 2.13.0 from
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
| Whisper tiny.en weights, ggml format (`ggml-tiny.en.bin`, 75 MiB) | `ggerganov/whisper.cpp` on Hugging Face (HF commit not pinned) | The model is OpenAI's: openai/whisper `README.md` at `86098128c0b4f24f0e2aa2994de830614b474227` says "Whisper's code and model weights are released under the MIT License". The ggml conversion's own model card was **not read** (Hugging Face was unreachable where this was written) | sha1 `c78c86eb1a8faa21b369bcd33207cc90d64ae9df` from whisper.cpp `models/README.md` at v1.9.5. **No sha256 recorded yet**: the file could not be downloaded to compute it |
| Test clip `jfk.wav` (11 s, 16 kHz mono, 352078 bytes) | `samples/jfk.wav` at the same commit, fetched from raw.githubusercontent.com | **Not verified.** The whisper.cpp repo states no provenance or licence for the recording (`samples/README.md` only says the folder holds "audio files used for testing"). The words are from a 1961 US presidential address, but the recording's copyright status is unknown to this repo. Therefore it is fetched, not committed | sha256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| `mlcommons-loadgen` (pip) | `6.0.17` | Apache-2.0: `LICENSE.md` of github.com/mlcommons/inference at `3fbc329939999c13d0a7b5e67fb2092287e06047` (the wheel's own metadata licence field is empty) | cp313 manylinux x86_64 wheel sha256 `79090cf79054bad8142d00b59aa81f40dc6e19cc5ef83ccafca9fce6d8ddaabb` (informational; installs are not hash-pinned because wheels differ per platform) |

MLPerf benchmark models and datasets are **not** used by any workload; see `docs/mlperf.md` for what
was found about their terms (several are member-only or gated, several unverified).

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
