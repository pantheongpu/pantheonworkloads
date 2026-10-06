# Models

Weights are never committed; manifests point at them. A licence is recorded only after reading the model's
own card or repository. When that was not possible the manifest says `UNVERIFIED` in `model.licence` and
`bin/pw validate` warns.

**Status of this list: huggingface.co was not reachable from the machine that wrote it (the proxy answered
403 to CONNECT), so no model card could be read, no file downloaded or hashed, and no revision pinned.**
GitHub was reachable, which is how llama.cpp's MIT licence was read.

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
