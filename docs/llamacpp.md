# llama.cpp workloads

[llama.cpp](https://github.com/ggml-org/llama.cpp) is MIT-licensed (its `LICENSE` file, read at the
pinned commit). The pin is in `tools/llamacpp/pin.env`: tag `b11447`, commit
`da263e7275dfbaeefcd61504eaa4fd5247540e11`. Results from other revisions are different measurements.

| Workload | Kind | Targets | What it does |
| --- | --- | --- | --- |
| `llamacpp-smollm2-135m` | functional | `cpu`, `gpu`, `sim:nvidia/*`, `sim:amd/*` | `llama-completion`, 3 greedy tokens (temp 0, seed 1, no chat template) after a fixed prompt, compared with `reference.json` |
| `llamacpp-bench-smollm2-135m` | benchmark | `cpu`, `gpu` | `llama-bench -o json`: prompt processing (`pp512_tokens_per_s`) and token generation (`tg128_tokens_per_s`) |
| `llamacpp-bench-mistral-7b-v03` | benchmark | `gpu` | the same, for a Mistral-7B-v0.3 GGUF you supply (no default download) |

## Backends and building

`tools/llamacpp/build.sh <cpu|cuda|hip> <prefix>` fetches the pinned commit (shallow, by hash), builds
`llama-completion` and `llama-bench` with cmake and installs them under `<prefix>`. A workload calls it when
`$PW_CACHE/llama.cpp-<tag>-<backend>/bin` has no binaries (`PW_CACHE` defaults to
`~/.cache/pantheonworkloads`) and reports SKIP (exit 77) when the machine cannot build: no git/cmake/compiler,
no network, no `nvcc` for `cuda`, no `hipcc` for `hip`.

- `cpu`: portable build (`GGML_NATIVE=OFF`), so the same binary gives the same text on any x86-64 host.
- `cuda`: `GGML_CUDA=ON`; `PW_CUDA_ARCHS` sets `CMAKE_CUDA_ARCHITECTURES`.
- `hip`: `GGML_HIP=ON`; `PW_HIP_ARCHS` sets `AMDGPU_TARGETS` (default `gfx942`: set it for your card).

Target to backend: `cpu` -> cpu; `gpu` -> cuda if `nvidia-smi` sees a card, else hip if `/dev/kfd` or `rocminfo`
exists (`PW_LLAMACPP_BACKEND` overrides); `sim:nvidia/*` -> cuda; `sim:amd/*` -> hip.
`PW_LLAMACPP_BIN_DIR` uses binaries you built yourself; `PW_LLAMACPP_NO_BUILD=1` never builds.

## Simulated GPUs

The wiring copies pantheonsim's: `tests/e2e/run_ollama.sh` (preload `libcuda`, `libcudart`, `libcublas`,
`libcublasLt` from `$PANTHEONSIM_DIR/build/shim`, set `VGPU_GPU`) and `amd/tests/e2e/run_ollama_amd.sh`
(preload `libamdhip64.so.7` and `libhsa-runtime64.so.1`, `HIP_VISIBLE_DEVICES=0`, `ROCBLAS_USE_HIPBLASLT=0`).
**It has not been run**: the machine that wrote it had neither a CUDA/ROCm toolchain nor pantheonsim built.
Expect to adjust it on the first real attempt. The reference for the functional workload must come from
the CPU build or a real GPU, never from a simulated target.

## Models and overrides

| Variable | Meaning |
| --- | --- |
| `PW_MODEL_URL` | direct `.gguf` link (the SmolLM2 workloads have a default; Mistral has none) |
| `PW_MODEL_SHA256` | expected sha256; default is `workloads/<name>/model.sha256` when that file exists. A mismatch fails the run. With no hash a warning prints the file's actual hash so you can pin it |
| `PW_MODEL_FILE` | a local GGUF, used instead of downloading |
| `PW_NGL` | layers offloaded (default 99 on gpu targets, 0 on cpu) |
| `PW_THREADS` | CPU threads (functional 2, bench 4) |
| `PW_BENCH_PP`, `PW_BENCH_TG`, `PW_BENCH_REPS`, `PW_BENCH_EXTRA` | llama-bench prompt tokens (512), generated tokens (128), repetitions (3), extra arguments such as `-fa 1` |

A recorded benchmark carries the manifest's model pointer, not your override: if you benchmark another
model, add a manifest for it (copy `llamacpp-bench-mistral-7b-v03`) rather than recording under the wrong name.

## Recording

```bash
bin/pw record llamacpp-smollm2-135m --target cpu          # functional reference (CPU build)
bin/pw run llamacpp-bench-smollm2-135m --target gpu --bench --repeat 3 --device "NVIDIA ..."
```
