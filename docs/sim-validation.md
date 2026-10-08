# Running the workloads on pantheonsim's simulated GPUs

Date of these runs: 2026-10-06. Host: 4 cores, 15 GB RAM, no GPU and no NVIDIA toolkit. Simulator:
pantheonsim `origin/main` at 3ce6885 plus the three fix branches below (not yet merged); the arch-* and lib-* runs below
also carry the three fixes of the second round (`sim-fix-nvrtc-builtins`, `sim-fix-cublaslt-int8`,
`sim-fix-sass-lea-neg-carry`; without them lib-fft-linalg, lib-attention-precision and, on a T4, lib-composite-blocks fail). Every run
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
| arch-llama-family | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | 9 random-weight architectures (Llama GQA, Mistral, Qwen2, Gemma, Phi-3, NeoX, Falcon...): 31 to 131 s; token ids and routing strings identical, statistics within 1.0e-6 of the CPU reference (tolerance 1e-3 + 1e-3 rel); each run also passes its own float64-CPU and cache-vs-recompute checks | 2026-10-06 |
| arch-moe | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | 28 to 78 s; Mixtral top-2-of-8 and top-3-of-5 through transformers' `grouped_mm` expert kernel (no `PW_ARCH_EXPERTS=eager` fallback needed): router choices identical, stats within 6e-7 | 2026-10-06 |
| arch-ssm | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | Mamba-1 and Mamba-2 (sequential scan in PyTorch ops): 24 s (t4) to 406 s (rtx5090, under load from other jobs); ids identical, stats within 3e-7 | 2026-10-06 |
| arch-vision | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | ViT and CLIP pair: 13 to 17 s; ids identical, stats within 1.1e-6 | 2026-10-06 |
| arch-encdec | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | Whisper-style, T5 gated-GELU and one more: 20 to 34 s; ids identical, stats within 7e-7 | 2026-10-06 |
| lib-fft-linalg | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | 10 to 14 s; 38 ops checked, worst error vs float64 8.9e-4 of rms (bound per op); 1 op (half-precision FFT) unchecked: unsupported in the CPU reference | 2026-10-06 |
| lib-sparse-embedding | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | 11 to 15 s; 54 ops checked, worst error 1.9e-6 of rms | 2026-10-06 |
| lib-rnn-conv | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | 12 to 17 s; 78 ops checked, worst error 2.6e-2 of rms (fp16 / bf16 RNN ops; every op inside its own bound) | 2026-10-06 |
| lib-attention-precision | sim:nvidia/{rtx5090, h100} | PASS | 14 to 19 s; 68 checked (worst error 7.5e-2 of rms: bf16 / fp8 ops, inside their bounds), 2 unsupported on any GPU (`unsupported_ok`), 14 unchecked (mem-efficient / cuDNN SDPA: unsupported in the CPU reference) | 2026-10-06 |
| lib-attention-precision | sim:nvidia/a100 | PASS | 14 s; 67 checked, 3 unsupported (also fp8 `_scaled_mm`: needs sm_89), 14 unchecked | 2026-10-06 |
| lib-attention-precision | sim:nvidia/t4 | PASS | 11 s; 53 checked, 17 unsupported (also every FlashAttention op: needs sm_80, and fp8), 9 unchecked | 2026-10-06 |
| lib-composite-blocks | sim:nvidia/{rtx5090, h100, a100, t4} | PASS | ViT block, diffusion U-Net step, DLRM forward / backward; 14 to 25 s; 11 ops, worst error 4.1e-2 of rms (bf16 U-Net; bound 2e-1) | 2026-10-06 |
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

## Nightly

`.github/workflows/sim-nightly.yml` repeats the arch-*, lib-* and gpt-train-fp32 runs every night (05:41 UTC) and on
`workflow_dispatch` (inputs: `sim-ref`, `workloads`), on hosted `ubuntu-24.04` runners, for `sim:nvidia/{h100, a100, rtx3060, t4, rtx5090}`
(one job each, `fail-fast: false`). A change to the workflow or to `tools/sim-env.sh` runs it on the pull request too.
Each run is checked by `bin/pw` against the CPU reference. The `report` job publishes `bin/pw matrix` as the job summary
and uploads `results.tsv` (columns: target, workload, result, seconds, detail) and `matrix.md` as the `results` artifact;
every GPU's own files are the `results-<gpu>` artifacts.

pantheonsim has no prebuilt simulator image (its GHCR image, `pantheonsim-ci`, is a CUDA toolchain that `tools/sim-env.sh`
does not need), so the first job builds the simulator with `tools/sim-env.sh` at a pinned pantheonsim commit and caches
the build by that commit; the GPU jobs unpack it and install torch (`torch==2.14.1`, `torchvision==0.29.1`, and the
`transformers` pin from `sim-env.sh`). To move the pin, change `SIM_REF_PIN` in the workflow and look at a run first.
`tests/test_sim_nightly.py` checks the file parses, the workloads exist and are functional, and the pin is a full commit.

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

### Second round: the arch-* and lib-* workloads (PyTorch on sim:nvidia)

The 10 PyTorch workloads of `workloads-coverage` (references recorded on the CPU only) were run on rtx5090, h100,
a100 and t4 (`tools/sim-env.sh` now installs `transformers==5.18.0`, the version the references were recorded with,
into the torch environment: 13 MB wheel plus pure-Python dependencies). First runs on rtx5090: the 5 arch-* and
lib-sparse-embedding, lib-rnn-conv, lib-composite-blocks passed; lib-fft-linalg and lib-attention-precision failed.
Every failure, with its cause:

| # | Failure | Cause | Fix |
| --- | --- | --- | --- |
| 4 | lib-fft-linalg: `eigvals_general...` and `det_slogdet` "unsupported" | not the ops: PyTorch compiles complex `abs` and `det` with its jiterator (NVRTC) and the simulator's NVRTC shim needs `nvcc` (`vgpu nvrtc: nvcc was not found`), which this host does not have. The shim also loads the real `libnvrtc` from the pip wheel when told (`VGPU_NVRTC_LIB`), but that library then dlopens `libnvrtc-builtins.so.13.0` by name, which sits beside it on no search path, so every compile failed with "failed to open libnvrtc-builtins" | pantheonsim `sim-fix-nvrtc-builtins` (2cc17e8): the shim opens the sibling builtins by full path first; test `nvidia/tests/e2e/run_nvrtc_builtins.sh` (needs no nvcc). Workloads: `_shared/torch_env.sh` sets `VGPU_NVRTC_LIB` from the torch environment's `nvidia/cu13/lib` |
| 5 | lib-attention-precision: `int8_matmul_int32_exact`, `int8_linear_with_dequantise` fail with `CUBLAS_STATUS_NOT_SUPPORTED` from `cublasLtMatmul` (`torch._int_mm`) | the cuBLASLt shim knew no integer types | pantheonsim `sim-fix-cublaslt-int8` (fee8ac4): int8 x int8 into int32 with `CUBLAS_COMPUTE_32I`, exact on the host; other integer combinations stay refused; `lt_paths.cu` `int8_into_int32` |
| 6 | lib-composite-blocks on a T4: `vit_block_forward_bf16` ends in `VirtualGPU error [invalid-pointer]` in `im2col_kernel<BFloat16>` (patch embedding, kernel = stride = 4, no padding), and the sticky error fails the 8 ops after it | simulator bug in the SASS executor: `LEA Rd, P0, Ra, -c[pad], 0x1` (low word of the 64-bit `index - pad`) negated its addend in 32 bits, so `pad = 0` produced no carry out (hardware adds `~x + 1`, which carries); the next `IADD3.X` made the high word 0xffffffff and the kernel read 2^32 elements before the image. IADD3 and LEA's negated shifted operand were already right. Only sm_75 reaches this kernel as SASS here | pantheonsim `sim-fix-sass-lea-neg-carry` (21fccc6): both LEA paths; `nvidia/tests/pytorch/sass_sm75.py` (`e2e_pytorch_sass_sm75`) |
| 7 | the 15 lib-* ops the CPU reference records as "unsupported" (mem-efficient / cuDNN SDPA, half FFT) made a GPU run FAIL, though returning numbers there is correct | reference semantics: a CPU lacks kernels a GPU has | `bin/pw`: a key whose reference is `unsupported` but whose target returns numbers is unchecked and counted in the detail (`unchecked: N ops (supported on this target, unsupported in the reference)`); the reverse stays a FAIL unless the manifest lists the key in `unsupported_ok` (list, or target pattern -> list). Unit tests in `tests/test_lib_coverage.py` (`UncheckedOps`) |
| 8 | lib-attention-precision: `sdpa_flash_fp32_causal`, `dynamic_quantized_linear` (all GPUs), `fp8_scaled_mm_e4m3` (a100, t4) and the 9 FlashAttention ops (t4) are "unsupported" against numbers in the CPU reference | not simulator bugs: FlashAttention exists for fp16 / bf16 and sm_80+, `torch._scaled_mm` needs sm_89, the dynamic-quantised Linear is a CPU path (the op raises that itself); the simulated GPUs behave as the hardware does | listed in the manifest's `unsupported_ok` per target; they show in the counts above |

`arch-moe` ran through transformers' `grouped_mm` on the simulated GPUs (the expected risk did not materialise);
`PW_ARCH_EXPERTS=eager` was not needed. No tolerance was changed and no reference was replaced: all
references remain the CPU recordings (CPU PyTorch 2.13.0, transformers 5.18.0). The statistics of the arch-*
workloads differ from them by at most 1.1e-6 on every simulated card, about 1000x inside the tolerance.

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
- Run times above are wall-clock on a host shared with other jobs (load average 6 to 8): arch-ssm took 406 s on the first card and 24 s on the last.
- Only 4 of 8 simulated NVIDIA cards were run for the second-round workloads (rtx5090, h100, a100, t4); b200, l4 and the rtx30xx ones are not run.
- The fix branches must be merged into pantheonsim before `tools/sim-env.sh` on a clean `origin/main`
  gives a torch that imports; until then `SIM_SRC` must carry them (here: `~/wt-sim-main`, with the
  three commits cherry-picked).
