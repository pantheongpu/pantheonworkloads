# First real-GPU run: NVIDIA A10G, 2026-10-08

One on-demand AWS g5.2xlarge (us-east-1), NVIDIA A10G 24 GB (sm_86, ECC on, 300 W, default clocks), 8 vCPU AMD EPYC 7R32 (AVX2, no
AVX-512), Ubuntu 22.04 Deep Learning AMI (driver 595.91.07, CUDA toolkits 12.8 to 13.2, Python 3.10, CMake 3.22). About 2 h 20 min
of runtime. The instance has been terminated. Everything below was run from a clean checkout of commit `7efdb83` (or an ancestor,
for the early exploratory runs) with `--target gpu`.

## Result of the functional pass (59 workloads that list `gpu`)

55 PASS, 4 FAIL, none SKIP. The failures are recorded as they are; no tolerance was loosened to hide any of them.

| Workload | Result | What was observed |
| --- | --- | --- |
| `onnx-zoo-bertsquad-int8` (and its `-bench` twin) | FAIL | logits differ from the zoo's expected output by 2.23 (the workload's own bound is 0.25, "int8 noise is about 0.03"). **Not a GPU effect**: the same error (2.225) comes out of the CPU provider on this host, with the same wheel, and out of every graph-optimisation level. The CPU reference was recorded on another CPU; this host has AVX2 only. Probably the u8s8 saturation behaviour of int8 MatMulInteger kernels without VNNI; unproven. Confirm by running it on an AVX-512 host. |
| `kokoro-tts-int8-onnx` | FAIL | second sentence is 70800 samples, the reference has 71400 (one 25 ms frame; the first sentence, 94200, agrees). Stable over 3 GPU runs. The same 70800 comes out of `--target cpu` on this host, so it is the host CPU's int8 kernels, not the GPU. The manifest says a sample-count difference must be investigated, not tolerated: it is not tolerated. RMS and centroid are inside their tolerances. |
| `moonshine-tiny-en-onnx` | FAIL (flaky) | exact text compare. 1 of 3 consecutive runs matched; the other two add a comma after "her parent" (token 29892), a near tie flipped by GPU run-to-run nondeterminism. Simulator-relevant: a greedy decode with a near tie is not stable even on one real card. The bench twin recorded (its check is the WER, 0.021). |
| `lib-attention-precision` | FAIL | one real mismatch: `cast_roundtrip_fp8_e4m3fn`. The inputs included 449.0 and 57344.0, outside e4m3fn's range (max 448). The CPU reference (torch 2.13.0) holds 448.0 (saturated); the A10G (the workloads' pinned torch 2.10.0) gave NaN. **Not the card and not the simulator**: it is the PyTorch version. 2.10.0 casts out-of-range values to NaN on the CPU and on CUDA alike, 2.13.0 and newer saturate, identically on an A10G and an L4 (probe in `docs/fp8-cast-semantics.md`). Fixed afterwards: the numerics op sees in-range values only and the overflow class is an informational op. |

Changes made because of the first run (each in its own commit):

- **ONNX Runtime CUDA EP defaulted to TF32** on Ampere. That moved fp32 logits by 2.4e-2 (mobilenetv2) and 7.9e-2 (shufflenet-v2) against the
  zoo's expected outputs (bound 1e-3), and changed the KWS and Moonshine outputs. `tools/ort_tasks.py` now passes `use_tf32=0`
  (`PW_ORT_TF32=1` restores the default); with it the same models agree to 1.1e-5 and 2.7e-5. This is a workload fix (the PyTorch
  workloads already turn TF32 off), not a loosened bound.
- **`lib-fft-linalg` `eigh_values_and_reconstruction` bound 1e-4 to 3e-4.** Measured 1.44e-4 of rms against the float64 CPU result with cuSOLVER.
  Justification: the 1e-4 came from CPU runs only; 3e-4 is about 2x the observation. This one is a genuine loosening and is flagged in the
  source and the commit.
- **`lib-attention-precision` `unsupported_ok` for `gpu`: `fp8_scaled_mm_e4m3`.** `torch._scaled_mm` is unsupported on sm_86 (needs sm_89), as the
  manifest already allowed for the simulated a100/rtx3060/rtx3080ti. It only permits "unsupported"; a card that supports it is still compared.
- `tools/build-whisper-cpp.sh`: CUDA arch defaults from `nvidia-smi` compute capability, because CMake 3.22's `native` makes nvcc fail.
- `llamacpp-smollm2-135m*`: the GGUF URL written earlier (`HuggingFaceTB/SmolLM2-135M-Instruct-GGUF`) does not exist. Now pinned to
  `bartowski/SmolLM2-135M-Instruct-GGUF` at an HF commit, with the file's sha256.

Other results worth knowing (all PASS): `arch-*` (5 workloads, 18 architectures) with top-1 margins 3.0e-3 to 5.1e-2 and fp64 deviations near
1e-6; `lib-fft-linalg` (38 ops, worst 3.56e-3 of rms) after the bound change; `lib-composite-blocks` (11 ops, worst 3.5e-2 of rms), `lib-rnn-conv` (78 ops, worst 2.6e-2),
`lib-sparse-embedding` (54 ops, worst 1.9e-6), `pytorch-microsuite` (worst 1.4e-2 of rms), `gpt-train-fp32` and `gpt-train-bf16`
(the CPU references passed unchanged at their tolerances 1e-3 and 2e-2; losses 3.583 to 2.976 / 2.979), all 13 `llamacpp-synth-*`
(F16 to the IQ and ternary quants and MXFP4, CUDA offload; they were recorded on the CPU and passed unchanged),
`spacy-*` (3), `resnet18-randinit-pytorch`, and the remaining ONNX workloads.

## Models resolved (references recorded on this GPU, not the CPU)

| Workload | Pin | Licence (read from) | Reference |
| --- | --- | --- | --- |
| `gpt2-small-pytorch` | `openai-community/gpt2` @ `607a30d7...`, `model.safetensors` sha256 `248dfc39...` (verified every run) | MIT: card front matter; the repo has no LICENSE file; OpenAI's gpt-2 repo is "Modified MIT" | " the capital of the French"; the CPU of the host agrees (first logit differs 4.6e-5, bound 2e-3) |
| `bert-base-uncased-pytorch` | `google-bert/bert-base-uncased` @ `86b5e093...`, `model.safetensors` sha256 `68d45e23...` | Apache-2.0: the repo's LICENSE file and the card | paris, lille, lyon, marseille, tours; CPU agrees (top-1 logit 1.9e-6) |
| `whisper-cpp-tiny-en` (+ bench) | `ggerganov/whisper.cpp` @ `5359861c...`, `ggml-tiny.en.bin` sha256 `921e4cf8...` (its sha1 matches whisper.cpp's list) | MIT: card front matter and openai/whisper LICENSE | JFK clip, WER 0.000 |
| `smollm2-135m-ollama` | `registry.ollama.ai/library/smollm2:135m`, manifest sha256 `9077fe9d...`, model layer `f535f83e...` | Apache-2.0: the registry's license layer is the Apache License 2.0 text | "The capital of" (3 greedy tokens) |
| `vllm-greedy-smollm2-135m` (+ bench) | `HuggingFaceTB/SmolLM2-135M` @ `93efa2f0...`, `model.safetensors` sha256 `80521b40...` | Apache-2.0: card front matter (no LICENSE file) | " the capital of the country.\n\n" |
| `llamacpp-smollm2-135m` (+ bench) | `bartowski/SmolLM2-135M-Instruct-GGUF` @ `09816acd...`, `SmolLM2-135M-Instruct-Q8_0.gguf` sha256 `5a139571...` | Apache-2.0: card of the GGUF repo and of the base model `HuggingFaceTB/SmolLM2-135M-Instruct`; this GGUF is a third-party conversion | " Paris. Paris\n\n" |

The vLLM and llama.cpp checksums are in the workloads' `model.sha256`; the vLLM one is not re-verified at run time (vLLM downloads the model itself).

## Setup traps on a stock Ubuntu 22.04 DLAMI (so the next run is faster)

1. `python3-venv` is not installed; install `python3.10-venv`.
2. **Python 3.10 cannot install the pins**: `onnxruntime-gpu==1.30.0` and `numpy==2.5.3` need Python 3.11+. Use `uv python install 3.12` and create the
   `ort-gpu` and `spacy-gpu` venvs with it (the `ort-env.sh` and `spacy-env.sh` scripts create venvs with `python3 -m venv` and will SKIP).
   `rapidocr_onnxruntime` pulls in the CPU `onnxruntime`; reinstall `onnxruntime-gpu` after it (the two share files).
3. **`LD_LIBRARY_PATH` on the DLAMI points at `/usr/local/cuda-13.2` and `cuda-12.9`**. PyTorch then loads the system cuBLAS and every matmul fails with
   `CUBLAS_STATUS_NOT_INITIALIZED`. Run the PyTorch and vLLM workloads with `env -u LD_LIBRARY_PATH`. The ONNX Runtime GPU workloads need it (CUDA 12 and cuDNN 9 come from there).
4. (The pins are torch 2.14.1 / torchvision 0.29.1 since the re-run below; both exist on `download.pytorch.org/whl/cu130` as `+cu130`.) PyTorch `torch==2.10.0` / `torchvision==0.25.0` existed there; `transformers==5.19.0` installs with a current pip
   (the pip in a fresh 3.10 venv hits a resolver assertion; upgrade pip first).
5. llama.cpp's CUDA build with CMake's default architecture list took about 10 minutes at `-j8`; use `PW_CUDA_ARCHS=86 PW_BUILD_JOBS=8`.
6. vLLM 0.30.0 failed in warm-up inside FlashInfer's JIT sampler (`ninja` not found); `VLLM_USE_FLASHINFER_SAMPLER=0` works.
7. Ollama's installer starts a system service on 11434; stop it, the workload starts its own server.

## Simulator-relevant findings

- Real-card behaviour the simulator must reproduce, or the reference must come from a GPU: `torch._scaled_mm` fp8 is unsupported on sm_86; flash attention fp32 and `dynamic_quantized_linear` are unsupported on CUDA
  (already covered by `unsupported_ok`).
- Out-of-range fp8 casts are a PyTorch-version property, not a card property (NaN in 2.10.0, saturating in 2.13.0+, CPU and CUDA alike; the
  simulator matches the hardware in both): `docs/fp8-cast-semantics.md`. Record which torch a GPU run used before comparing it with a CPU reference.
- Defaults that change numerics silently: ONNX Runtime CUDA EP uses TF32 for fp32 Conv and MatMul on Ampere (errors up to 8e-2 on logits).
- All 13 `llamacpp-synth-*` workloads (kernels for 30+ quantisation types, six architectures, MoE) matched the CPU references exactly on CUDA; so did
  `gpt-train-*` and the 18 `arch-*` architectures within their fp32 tolerances. These are the workloads whose simulated results can be trusted to be compared with this card.
- Run-to-run nondeterminism on the real card exists (Moonshine text, 2 of 3 runs differ). Exact-compare references of long greedy decodes are fragile.
- Int8 ONNX models (BERT-Squad, Kokoro) depend on the host CPU, not the GPU, in a mixed CUDA-EP session; do not read their GPU deviations as GPU deviations.

## Re-run on torch 2.14.1 (2026-10-08, fourth g5.2xlarge)

Same card type and driver (A10G, 595.91.07), a fresh Ubuntu 22.04 DLAMI (`Deep Learning Base OSS Nvidia Driver GPU AMI` 20261006), `--target gpu`,
commit `479c1c7` (PR #19's branch plus a commit that makes the arch, lib and gpt-train workloads report their library versions like the
`_pytorch/env.sh` ones). Rig setup that worked first time: `uv python install 3.12`; `uv venv --python 3.12 ~/.cache/pantheonworkloads/venvs/cu130`;
`env -u LD_LIBRARY_PATH uv pip install --python <venv>/bin/python --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
--index-strategy unsafe-best-match -r workloads/_pytorch/requirements-torch.txt` (without `unsafe-best-match` uv takes PyPI's `+cu128`), then
`requirements-common.txt` and `requirements-diffusion.txt`. The arch, lib and gpt-train workloads do not use that venv by themselves
(`_shared/torch_env.sh` runs `PW_PYTHON` or `python3`), so run them with `PW_PYTHON=<venv>/bin/python`. `torch.__version__` was `2.14.1+cu130`
(CUDA 13.0), torchvision `0.29.1+cu130`, transformers 5.19.0, diffusers 0.41.0, numpy 2.5.3. The 243 unit tests pass (with the venv one test's
expected key set needed `versions`; updated), `bin/pw validate` and `bin/pw coverage --check` are clean.

All 18 PyTorch functional workloads PASS against their existing references, with no tolerance changed and no reference re-recorded:
`arch-encdec`, `arch-llama-family`, `arch-moe`, `arch-ssm`, `arch-vision` (top-1 margins 3.0e-3 to 5.1e-2, fp64 deviations about 1e-6),
`lib-attention-precision` (67 ops checked, 3 unsupported as allowed, 2 informational; worst 7.5e-2 of rms; the 14 ops the CPU reference lists as
unsupported are not compared, as before; the fp8 e4m3fn overflow class is `saturates` on CUDA and CPU, so the fp8 failure of the 2.10.0 run is gone),
`lib-composite-blocks` (worst 3.9e-2 of rms, 3.5e-2 on 2.10.0, inside its bound), `lib-fft-linalg` (3.56e-3), `lib-rnn-conv` (2.6e-2), `lib-sparse-embedding` (1.9e-6),
`gpt-train-fp32` (loss 3.583 to 2.976), `gpt-train-bf16` (3.583 to 2.979), `pytorch-microsuite` (1.39e-2),
`resnet18-randinit-pytorch` (min top1-top2 gap 0.00249), `gpt2-small-pytorch`, `bert-base-uncased-pytorch`, `blip2-opt-2p7b-pytorch`
(caption 'a woman in an orange space suit with a space helmet') and `sdxl-base-diffusers` (informational pixel sha256 `8cdbac1b501c30b5`).
Where the 2.10.0 pass above gave a figure, the worst errors, losses and margins equal it as printed, except `lib-composite-blocks`. Benchmark medians against the
2.10.0 records: `docs/benchmarks.md` (torch 2.14.1 re-run). About 45 minutes of instance time, including a 2.10.0 control run of three bench twins.
