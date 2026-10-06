# Running the workloads on pantheonsim's simulated GPUs

Date of these runs: 2026-10-06. Host: 4 cores, 15 GB RAM, no GPU and no NVIDIA toolkit. Simulator:
pantheonsim `origin/main` at 3ce6885 plus the three fix branches below (not yet merged). Every run
is `bin/pw run <workload> --target sim:nvidia/<gpu>` with the environment from `tools/sim-env.sh`.
The references are the CPU recordings (`reference.json`); the runs are checked by `bin/pw`'s own
comparison, not by eye.

## Setting up

```bash
tools/sim-env.sh            # builds into $SIM_BUILD (~/sim-main-build), installs torch, prints exports
eval "$(tools/sim-env.sh)"  # VGPU_BUILD_DIR, PANTHEONSIM_DIR, VGPU_TORCH_CUDA_PYTHON, CUDA_HOME
PW_PYTHON=<python with a CPU torch> bin/pw run gpt-train-fp32 --target sim:nvidia/rtx5090
tools/sim-env.sh --check    # only report what is missing
```

No toolkit is needed: the simulator's CUDA shims are compiled against the CUDA *headers*, which come
from the pip wheels PyTorch itself depends on (plus `nvidia-nvml-dev`, `-npp`, `-nvjpeg`, `-cuda-crt`,
`-cuda-cccl`); see the header of `tools/sim-env.sh`. The simulator needs nothing else: `vgpu_cli` and
the shims for the CUDA driver and runtime, cuBLAS/cuBLASLt, cuDNN, NVML, NVRTC, cuFFT, cuRAND,
cuSOLVER, cuSPARSE, NCCL, CUPTI and HIP/HSA/ROCm SMI/rocprofiler build in roughly ten minutes at `-j3`.

**PyTorch for CUDA 13 comes from PyPI.** On Linux x86-64, `pip install torch` gives 2.14.1+cu130 (the
`cuda-toolkit==13.0.3` pin). Download: torch 0.53 GB, with cuDNN 0.53, cuBLAS 0.40, NCCL 0.21,
Triton 0.24, cuFFT 0.20, cuSOLVER 0.19, cuSPARSE 0.14, cuSPARSELt 0.16 and the rest, **2.8 GB**;
5.3 GB installed. It went into `~/.local/share/torch-cu13` (the venv `torch_env.sh` looks in).
torchvision 0.29.1 is added for the ResNet workload.

**PyTorch for ROCm is not obtainable here**, so no workload that needs it ran on `sim:amd/*`:
`download.pytorch.org` and `repo.radeon.com` answer 403 / no route, PyPI carries no ROCm torch
(`torch-rocm` is 404; `pytorch-triton-rocm` is only Triton), and conda-forge builds torch for CPU and CUDA
only. Nothing was substituted: those cells say SKIP.

**Weights are not obtainable**: `huggingface.co` answers 403, so `gpt2-small-pytorch` and
`bert-base-uncased-pytorch` cannot run on any target here, and the llama.cpp, whisper.cpp, Ollama and
vLLM workloads (which need a model, and for CUDA an `nvcc` build or Ollama, for HIP a vLLM for ROCm) were
not run.

## Results

PASS = `bin/pw` compared the output with the CPU reference (or the workload's own checks) and it passed.
"Dev" is the largest difference from the reference. Times are wall-clock for the whole workload,
including starting the simulator and importing torch.

| Workload | Target | Result | Detail | Date |
| --- | --- | --- | --- | --- |
| gpt-train-fp32 | sim:nvidia/rtx5090 | PASS | 22 s; loss 3.583 -> 2.976; dev 1.0e-6 (tolerance 1e-3) | 2026-10-06 |
| gpt-train-fp32 | sim:nvidia/h100 | PASS | 26 s; dev 1.0e-6 | 2026-10-06 |
| gpt-train-fp32 | sim:nvidia/a100 | PASS | 26 s; dev 1.0e-6 | 2026-10-06 |
| gpt-train-fp32 | sim:nvidia/b200 | PASS | 25 s; dev 1.0e-6 | 2026-10-06 |
| gpt-train-fp32 | sim:nvidia/t4 | PASS | 26 s; dev 1.0e-6 | 2026-10-06 |
| gpt-train-fp32 | sim:nvidia/rtx3060 | PASS | 21 s; dev 1.0e-6 | 2026-10-06 |
| gpt-train-bf16 | sim:nvidia/rtx5090 | PASS | 28 s; loss 3.583 -> 2.973; dev 2.7e-3 (tolerance 2e-2) | 2026-10-06 |
| gpt-train-bf16 | sim:nvidia/h100 | PASS | 22 s; dev 1.5e-3 | 2026-10-06 |
| gpt-train-bf16 | sim:nvidia/b200 | PASS | 24 s; dev 1.5e-3 | 2026-10-06 |
| gpt-train-bf16 | sim:nvidia/a100 | PASS | 22 s; dev 2.7e-3 | 2026-10-06 |
| gpt-train-bf16 | sim:nvidia/l4 | PASS | 21 s; dev 2.7e-3 | 2026-10-06 |
| gpt-train-bf16 | sim:nvidia/t4 | SKIP | the workload's own check: bf16 is not supported on Tesla T4 (sm_75) | 2026-10-06 |
| pytorch-microsuite | sim:nvidia/{rtx5090, h100, t4, a100, b200, rtx3060, l4, rtx3080ti} | PASS | 6.7 to 10 s each; 13 ops and 3 index checks; every output within 3e-7 relative of the CPU reference, index lists identical; worst error vs float64 1.39e-2 of rms (bf16; limit 1e-1) | 2026-10-06 |
| resnet18-randinit-pytorch | sim:nvidia/{rtx5090, h100, t4, a100, b200, rtx3060, l4, rtx3080ti} | PASS | 8.5 to 9.6 s each; same class ids and logits within tolerance | 2026-10-06 |
| selftest | sim:nvidia/h100, sim:amd/mi300x | PASS | runner self-check | 2026-10-06 |
| loadgen-plumbing-check | cpu | PASS | 3.6 s | 2026-10-06 |
| loadgen-plumbing-check | sim:* | SKIP | its manifest lists only `cpu`: it drives a CPU system under test, no GPU is involved | 2026-10-06 |
| gpt-train-fp32, gpt-train-bf16, pytorch-microsuite, resnet18-randinit-pytorch | sim:amd/* | SKIP | no PyTorch for ROCm obtainable (see above) | 2026-10-06 |
| gpt2-small-pytorch, bert-base-uncased-pytorch | sim:nvidia/* | SKIP | no weights: huggingface.co is blocked | 2026-10-06 |
| llamacpp-smollm2-135m | sim:nvidia/h100, sim:amd/mi300x | SKIP | cannot build llama.cpp for cuda / hip here (no nvcc; the model is on Hugging Face) | 2026-10-06 |
| smollm2-135m-ollama | sim:nvidia/h100 | SKIP | ollama is not installed (ollama.com blocked) | 2026-10-06 |
| vllm-greedy-smollm2-135m | sim:amd/mi300x | SKIP | no Python with vLLM for ROCm | 2026-10-06 |
| llamacpp-bench-*, whisper-cpp-*, vllm-bench-throughput, gpt-train-bench | sim:* | SKIP | not simulator targets (benchmarks, or whisper/llama.cpp on `gpu` only) | 2026-10-06 |

The gpt-train fp32 deviation is 1.0e-6 on every simulated card: the simulator runs the cuBLAS and
cuDNN math on the host, so it lands where the CPU backend does, 1000x inside the tolerance. The
bf16 deviation (1.5e-3 to 2.7e-3) is rounding of bf16 activations; it differs by card because
PyTorch's kernels differ per architecture (SASS per SM). Neither tolerance was changed.

## Issues found and fixed

PyPI's torch 2.14.1 is newer than what pantheonsim's own PyTorch tests used, and `import torch` failed
three times in a row on symbols that `libtorch_cuda.so` binds at load time (it is linked with
`-z now`, so a missing name fails the import, not the call). Each fix is on its own pantheonsim branch
off `origin/main`, with a unit test, and none of them are in the workloads repository:

| # | Failure | Cause | pantheonsim branch (commit) |
| --- | --- | --- | --- |
| 1 | `undefined symbol: ncclCommResume` | libtorch binds 13 NCCL 2.29+ names (`ncclCommSuspend/Resume/Revoke/Grow/GetUniqueId/MemStats`, `ncclPutSignal/Signal/WaitSignal`, `ncclDevCommCreate/Destroy`, `ncclGetLsaMultimemDevicePointer`, `ncclGetPeerDevicePointer`); libvgpunccl did not export them | `sim-fix-nccl-2-30-symbols` (1f213c7): exported, each answers `ncclInvalidUsage` with a message; `test_nccl_symbols` |
| 2 | `undefined symbol: cudaKernelSetAttributeForDevice` | the runtime shim had no user objects (`cudaUserObjectCreate/Release`, `cudaGraphRetainUserObject`), no library API (`cudaLibraryLoadData/GetKernel/Unload`), and no `cudaKernelSetAttributeForDevice` | `sim-fix-runtime-libraries-userobjects` (22bd7f2): user objects are real (refcounts; graphs and the execs instantiated from them hold references; destructor runs once), the library calls forward to the driver's `cuLibrary*`; `test_runtime_user_objects` |
| 3 | `cublasLtMatmulAlgoInit` and 3 more unresolved | cuBLASLt's algorithm configuration (`Init`, `Check`, `ConfigGet/SetAttribute`) was missing | same branch (7098b94): records the configuration, `Check` answers what the matmul would; `test_cublaslt_algo` |

Known gap left as it was: `cuKernelSetAttribute` accepts and keeps nothing, so a kernel obtained from
a library keeps no shared-memory ceiling (a `CUfunction` does).

Workload-side fixes (this branch): `workloads/_pytorch/env.sh` printed `tmp: unbound variable` on every
run (its `trap` expanded a local after the function returned), ignored `VGPU_BUILD_DIR`, and with
`download.pytorch.org` blocked spent 8 minutes in pip retries before skipping; `_shared/torch_env.sh`
now honours `VGPU_BUILD_DIR` too. References were recorded on the CPU for `pytorch-microsuite` and
`resnet18-randinit-pytorch` (which had none); the manifests' notes say so.

## Remaining

- sim:amd for PyTorch needs a ROCm torch build, from an index this host cannot reach.
- Hugging Face weights (gpt2, bert, SmolLM2, whisper) cannot be fetched here.
- Nothing was run on a physical GPU: the tolerances are validated against the simulator and the CPU only.
- The fix branches must be merged into pantheonsim before `tools/sim-env.sh` on a clean `origin/main`
  gives a torch that imports; until then `SIM_SRC` must carry them (here: `~/wt-sim-main`, with the
  three commits cherry-picked).
