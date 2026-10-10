# llama.cpp workloads

[llama.cpp](https://github.com/ggml-org/llama.cpp) is MIT-licensed (its `LICENSE` file, read at the
pinned commit). The pin is in `tools/llamacpp/pin.env`: tag `b11447`, commit
`da263e7275dfbaeefcd61504eaa4fd5247540e11`. Results from other revisions are different measurements.

| Workload | Kind | Targets | What it does |
| --- | --- | --- | --- |
| `llamacpp-smollm2-135m` | functional | `cpu`, `gpu`, `sim:nvidia/*`, `sim:amd/*` | `llama-completion`, 3 greedy tokens (temp 0, seed 1, no chat template) after a fixed prompt, compared with `reference.json` |
| `llamacpp-bench-smollm2-135m` | benchmark | `cpu`, `gpu` | `llama-bench -o json`: prompt processing (`pp512_tokens_per_s`) and token generation (`tg128_tokens_per_s`) |
| `llamacpp-qwen25-0p5b` | functional | `cpu`, `gpu`, `sim:nvidia/*`, `sim:amd/*` | the same for Qwen2.5-0.5B-Instruct Q8_0 (official GGUF) |
| `llamacpp-bench-qwen25-0p5b` | benchmark | `cpu`, `gpu` | the same bench for Qwen2.5-0.5B-Instruct |
| `llamacpp-mistral-7b-v03` | functional | `gpu` | the same 3-token greedy decode for Mistral-7B-v0.3 Q8_0 (7.7 GB, `-ngl 99`); reference recorded on an A10G |
| `llamacpp-bench-mistral-7b-v03` | benchmark | `gpu` | the same bench for Mistral-7B-v0.3 Q8_0 (default download, pinned; override with `PW_MODEL_URL`/`PW_MODEL_FILE` for other quantisations) |
| `llamacpp-llama32-1b-instruct`, `llamacpp-llama32-3b-instruct` (+ `llamacpp-bench-*`) | functional, benchmark | `gpu` | Llama-3.2-1B/3B-Instruct Q8_0 (bartowski conversion). **restricted** (Llama 3.2 Community Licence): skipped by default selections |
| `llamacpp-gemma4-e4b-it` (+ bench) | functional, benchmark | `gpu` | Gemma 4 E4B-it, Google's own QAT Q4_0 GGUF (5.2 GB). Apache-2.0, not restricted |
| `llamacpp-gpt-oss-20b` (+ bench) | functional, benchmark | `gpu` | gpt-oss-20b MXFP4 GGUF from ggml-org (12.1 GB, fits one 24 GB card with `-ngl 99`). Apache-2.0 |
| `llamacpp-deepseek-r1-distill-qwen-7b`, `llamacpp-deepseek-r1-distill-qwen-1p5b` (+ bench) | functional, benchmark | `gpu` | DeepSeek-R1-Distill-Qwen Q8_0 (bartowski conversions). MIT |
| `llamacpp-qwen3-30b-a3b` (+ bench) | functional, benchmark | `gpu` | Qwen3-30B-A3B (mixture of experts, 128 experts) Q4_K_M, Qwen's own GGUF (18.6 GB, fits one 24 GB card). Apache-2.0 |
| `llamacpp-qwen3-8b` (+ bench) | functional, benchmark | `gpu` | Qwen3-8B (dense) Q6_K, Qwen's own GGUF (6.7 GB). Apache-2.0 |
| `llamacpp-phi4-14b` (+ bench) | functional, benchmark | `gpu` | Phi-4 14B Q4_K, Microsoft's own GGUF (9.1 GB). MIT |
| `llamacpp-granite40-h-small` (+ bench) | functional, benchmark | `gpu` | Granite 4.0 H-Small, IBM's own Q4_K_M GGUF (19.5 GB): a hybrid of Mamba-2 state-space layers, attention and a mixture of experts. Apache-2.0 |
| `llamacpp-mistral-small-32-24b` (+ bench) | functional, benchmark | `gpu` | Mistral-Small-3.2-24B-Instruct-2506 (text part) Q4_K_M, bartowski's conversion (14.3 GB). Apache-2.0 |
| `llamacpp-olmo2-7b-instruct` (+ bench) | functional, benchmark | `gpu` | OLMo-2-1124-7B-Instruct Q8_0, Ai2's own GGUF (7.8 GB). Apache-2.0 |
| `llamacpp-qwen3-embedding-4b` (+ bench) | functional, benchmark | `gpu` | Qwen3-Embedding-4B Q8_0, Qwen's own GGUF (4.3 GB). The functional workload runs `llama-embedding` (last-token pooling) on a query and three passages and compares cosine similarities and the passage ranking; the bench twin is the same `llama-bench` pp512/tg128. Apache-2.0 |

The model catalog ([`docs/model-registry.md`](model-registry.md)) adds eighty-two `llamacpp-<model>` / `llamacpp-bench-<model>` pairs for models from
3 B to 2.4 T parameters. They are written, never run. They differ from the workloads above in three ways: many GGUFs are split into shards
(`tools/llamacpp/common.sh` `lc_fetch_shards` downloads and verifies each file listed in `model.sha256` and hands llama.cpp the first shard); they
call `lc_require_gpus <n> <GiB>` first (exit 77 with the needs when the host lacks them); and the layers are split over the n GPUs llama.cpp sees
(its default `--split-mode layer`, `-ngl 99`).

## Backends and building

`tools/llamacpp/build.sh <cpu|cuda|hip> <prefix>` fetches the pinned commit (shallow, by hash), builds
`llama-completion` and `llama-bench` with cmake and installs them under `<prefix>`. `LLAMACPP_EXTRA_TARGETS` adds more programs; `llama-embedding` is an example, not a tool, so it also needs `LLAMACPP_BUILD_EXAMPLES=1` (the embedding workload sets both for a first build; a prefix built earlier without it makes that workload SKIP, so delete the prefix or build into a new one). A workload calls it when
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
| `PW_MODEL_URL` | direct `.gguf` link (every workload has a pinned default) |
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
