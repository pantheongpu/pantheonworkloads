# Models and workload sources

Each section below is self-contained and owned by whoever added that workload family.

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
