# Model registry

Model size does not limit what this repository supports. Every open-weight model worth tracking has an entry here, however
large, and a workload that runs only on GPUs that can hold it: the workload's `requires:` block (docs/manifest.md) names the
GPU count and the memory each GPU needs, `bin/pw` reports SKIP on a host that cannot meet it, and the workload's `run.sh`
repeats the check (exit 77 with the numbers) when it is started by hand.

**Everything in this file was written from public Hugging Face and GitHub metadata, on 2026-10-09, without credentials. When it
was written no weights had been downloaded, no GPU had been used, and no workload had been run.** None had a `reference.json` or
a `bench/` record; the coverage table in the README marks the ones still in that state as never run (family "Model catalog",
reference "no", GPU-validated "no"). A manifest's `notes` starts with `WRITTEN, NEVER RUN` and ends with what was not verified,
until its workload has had its first real run. After it, the notes start with `VERIFIED ON <card>` (a reference and a bench
record from a real GPU exist) or `BLOCKED(run)` (the first run showed it cannot work as pinned; the notes carry the evidence),
and the generator stops writing the workload's files, which are maintained by hand from then on. The first real runs so far
are in "First real runs" below; for every other row a SKIP from the runner is the expected result.

The tables at the end are generated (`tools/catalog/generate.py`) from three files, so a row can always be traced to evidence:

| File | What it holds |
| --- | --- |
| `tools/catalog/catalog.yaml` | the choices: which models, which repository to download, which runtime, which quantisation |
| `tools/catalog/hub.json` | the facts, read from `https://huggingface.co/api/models/<id>?blobs=true` and the raw card / LICENSE / config files at that commit: commit sha, gated flag, licence tags, parameter counts, GGUF architecture, every file with its size and LFS sha256 |
| `tools/catalog/support.json` | which architectures the pinned runtimes list, read from their source at the pinned tags: llama.cpp `src/llama-arch.cpp` at b11447, vLLM `registry.py` at v0.30.0, transformers `auto_mappings.py` at v5.19.0, diffusers `__init__.py` at v0.41.0 |

To refresh after a change: `tools/catalog/fetch_hub.py` (only what is missing; `--refresh` for all), `tools/catalog/fetch_support.py`,
`tools/catalog/generate.py`, then `bin/pw coverage --write`. `tests/test_model_catalog.py` fails when anything on disk differs from
what the generator would write, and checks the invariants listed under "Tests" below.

## How a row is decided

**Runtime plan.** In order: llama.cpp with a GGUF when a GGUF repository that is not gated exists and the GGUF header's architecture
is in the pinned llama.cpp's list (providers tried in order: the model's own `-GGUF` repository, `ggml-org`, `unsloth`, `bartowski`,
`lmstudio-community`; third-party conversions are labelled as such); otherwise vLLM (official weights, tensor parallel) when the pinned
vLLM lists the architecture; transformers for vision-language, OCR, retrieval, vision, speech and time-series models; diffusers for
image and video generation; whisper.cpp for Whisper. An entry whose runtime does not list the architecture is blocked, not guessed
(for example DeepSeek-V4.1-Flash has community GGUFs that declare `deepseek41`, which llama.cpp b11447 does not know, so it runs on
vLLM, which lists `DeepseekV41ForCausalLM`).

**Quantisation / dtype.** GGUF: Q4_K_M, else the nearest 4-bit the provider offers (UD-Q4_K_XL, MXFP4, Q4_0, IQ4_XS); models up to
14.5 B parameters use Q8_0. A model's own QAT file is used when it publishes one (Gemma 4: Q4_0). vLLM and transformers: the
checkpoint as stored (FP8 / FP4 / MXFP4 repositories stay quantised), loaded bf16 when it is stored in 16 or 32 bits. Diffusers: the bf16
variant when the largest component has one, else the files as stored, loaded bf16.

**Weights size.** Summed from the Hub file sizes of exactly the files pinned in `model.sha256` (the shards an index file names, or the
single file; never duplicate formats). Stored-fp32 diffusers and transformers checkpoints that are loaded in bf16 count half.

**Hardware need.** The fewest GPUs (1, 2, 4, 8) that hold the weights plus margin in a widely rented memory class (up to 80 GiB), each
of the smallest class that fits; if even 8 x 80 GiB is not enough, the cheapest n x class among 140 to 288 GiB cards; above that, 16 or 32
GPUs, flagged as more than one node (none needed so far). Classes are the GiB a driver reports, and `bin/pw` accepts a card that reports
98% of the class, so `22` means "any 24 GB-marketing card" (A10G and L4 report 22.5 GiB), `24` a card with a full 24 GiB, `44` an L40S
(45 GiB), `48` an A6000 / RTX 6000 Ada, `80` an A100 / H100 (79.6 GiB), `140` an H200, `180` a B200, `192` an MI300X, `256` an MI325X, `288`
an MI355X / B300. The margins are assumptions, not measurements (nothing was run), and each is stated in the manifest's
`requires.notes`:

| Plan | GiB needed on each of n GPUs | What the margin stands for |
| --- | --- | --- |
| llama.cpp (layer split) | `weights/n * 1.05 + 3.0` | 5% for uneven layer placement; 3 GiB for the CUDA context, compute buffers and the KV cache of a 4096-token context |
| whisper.cpp | `weights/n * 1.05 + 2.0` | as above, a smaller context |
| vLLM (tensor parallel) | `(weights/n + 6.0) / 0.90` | `gpu_memory_utilization` 0.90; 6 GiB for activations, CUDA context, NCCL buffers and the KV cache of 4096 tokens |
| transformers (`device_map=auto`) | `weights/n * 1.10 + 4.0` | 10% for uneven placement; 4 GiB for activations of one image, clip or short sequence |
| diffusers | `max(largest component, weights/n) + 6.0` (image) or `+ 16.0` (video) | latents, attention activations and the VAE decode at the workload's resolution; diffusers' `device_map=balanced` places whole components, so the largest one must fit one GPU |

Alternatives that also fit are not expressible in `requires` (it is one count and one memory): the listed one is the floor, and a host with
more memory per GPU or more GPUs passes. A host with the right total memory in a different shape (for example 4 x 256 GiB where the row says 8 x 140)
is skipped; set `PW_IGNORE_REQUIRES=1` to override at your own risk.

**Licence.** Read from the model's own card front matter (`license`, `license_name`, `license_link`) and the first bytes of its LICENSE
files at the pinned commit, with the downloaded repository's card read as well; the manifest's `notes` quote what was seen. A GGUF
conversion that disagrees with the official card is recorded at the more restrictive of the two, and a card that says `license: other`
without a name does not override an explicit licence on the other card. A model derived from a Llama base (`base_model: meta-llama/...`) is
treated as restricted even if its card says Apache-2.0 (Orpheus). `restricted: true` (excluded from default selections, docs/manifest.md) for
every licence that is not Apache-2.0, MIT, BSD, ISC, CC0 or CC-BY-4.0: Llama, Gemma, Qwen, Mistral research and non-production, Falcon, Cohere
(CC-BY-NC), FLUX, Stability, Hunyuan, CogVideoX, LTX, NVIDIA, OpenMDW, Modified-MIT variants, Kimi, GLM-5.3 and so on. A repository with no
licence statement anywhere is blocked as `licence unreadable`.

**Gated repositories.** No Hugging Face credentials are available and none are requested. A gated repository lists its files and sizes
but not their contents, so it is recorded with the access route (accept the terms on its page, then set `HF_TOKEN`) and gets no workload
of its own. Where a community conversion that is not gated exists (the Llama GGUFs of bartowski / unsloth / ggml-org / MaziyarPanahi), the
workload downloads that copy, is marked `restricted`, says in its notes that it is a third-party conversion of a gated model, and cites
Meta's public licence text (`meta-llama/llama-models` at a pinned commit; the Hub cards of the conversions often do not carry the text, which
the manifest says). The official repository stays "blocked-for-download" in the table. This follows the precedent of
`llamacpp-llama32-1b-instruct`; it is a judgement call, and deleting those entries from `catalog.yaml` removes the workloads.

## What is not verified

Everything that needs a run: that the pinned runtime loads the file, the output, speed, real memory use, the tolerances of the
statistics-based workloads, the prompts and chat templates of the OCR and vision-language models, and the calls into packages
(`chatterbox-tts`, `f5-tts`, NeMo, `chronos-forecasting`) whose pins in `workloads/_pytorch/requirements-speech.txt` were read from PyPI
and never installed together. The vLLM and llama.cpp quantisation kernels for a given GPU generation were not checked either. The
architecture checks are against the runtimes' source lists, which say a name is registered, not that the checkpoint's layout matches the
implementation (the OCR models, whose repositories were written for custom code, are the likeliest to need work).

## Tests

`tests/test_model_catalog.py` checks, offline: the files on disk equal what the catalog generates; every written manifest validates, is
pinned to a 40-hex commit, names a licence that is not `UNVERIFIED` and is identical to the card-derived one, carries a `requires` block, has
`model.sha256` lines equal to the Hub listing (64-hex oids, GGUF shard sets complete), has no `reference.json` and no bench record;
`restricted` follows the licence; blocked rows say why and gated ones carry the access route; `requires.gpus x gpu_memory_gb` is recomputed
from the formulas above (it must hold weights plus margin and be the smallest class), and big workloads fit their total memory; the runtime
architecture checks; each `run.sh` is valid bash, repeats the manifest's numbers, and, run against a fake `nvidia-smi`, exits 77 with the
needs on a host without GPUs (one per plan); and `tools/hwcheck.py` counts cards the way `bin/pw` does.

## Running one

Name the workload; it is left out of every default selection when restricted. The disk needs the weights size under `PW_CACHE`
(`run.sh` exits 77 with the figure when it does not fit). Multi-hundred-GiB downloads are resumable and verified file by file against
`model.sha256`, so a re-run continues. Big timeouts follow the weights size (12 s per GiB plus an hour).

```bash
bin/pw list --runnable                       # what this host can run, from the requires blocks
bin/pw run llamacpp-qwen3-coder-next --target gpu       # 1 x 80 GiB
bin/pw run vllm-gpt-oss-120b-vllm --target gpu          # 1 x 80 GiB, vLLM v0.30.0
PW_CACHE=/big/disk bin/pw run llamacpp-kimi-k3 --target gpu   # 8 x 192 GiB, 1405 GiB of weights
```

## First real runs

### Stage 2a, Tier A (2026-10-10): the 49 workloads that fit one 24 GB card

Every catalog workload with `requires.gpus == 1` and `gpu_memory_gb <= 22` was run, with `--target gpu`, on one on-demand g5.2xlarge in
us-east-1: NVIDIA A10G (22 GiB, sm_86, ECC on), driver 595.91.07, Ubuntu 22.04 (Deep Learning Base OSS Nvidia Driver AMI 20261009), 8 vCPU,
600 GB gp3. About 6 h 25 min of instance time. Software: torch 2.14.1+cu130, torchvision 0.29.1, transformers 5.19.0, diffusers 0.41.0,
numpy 2.5.3 (Python 3.12 venvs from `uv`, as in `docs/first-gpu-run-a10g.md`); llama.cpp b11447 (`da263e72`, CUDA, `PW_CUDA_ARCHS=86`, built
with `llama-embedding`); whisper.cpp v1.9.5 (CUDA). The exceptions are packages that pin their own stack, each in its own venv
(`PW_VENV_NAME`, `workloads/_pytorch/env.sh`): chatterbox (torch 2.6.0+cu124, numpy 1.26.4, transformers 5.2.0), NeMo (`nemo` venv,
3.0.0, torch 2.14.1) and f5-tts (`f5` venv, torchaudio 2.11.0 beside torch 2.14.1).

**Result: 49 of 49 PASS.** For each one the reference was recorded on the A10G (`bin/pw record --target gpu`) and was identical in 3 further
consecutive runs, with no tolerance loosened and no near-tie flipped; then the bench twin was recorded at 5 repeats
(`bin/pw run <twin> --target gpu --bench --repeat 5`; llama.cpp: `llama-bench` pp512 and tg128, 3 repetitions inside each of the 5 runs). The
bench records name `repo_commit` `ba94bc74` (the commit that carries the references and run-script fixes, on branch `stage2a-tier-a`) with
`repo_dirty: false`: the rig's checkout was that commit, clean, and the records were written outside the tree (`PW_BENCH_DIR`). The weights
are pointers: nothing was committed except references and records. The `llamacpp-*` pp512 and tg128 spreads over the 5 runs are 0.0 to 0.1 %.

The notes of all 98 manifests (functional and bench twin) now start with `VERIFIED ON NVIDIA A10G (model catalog, stage 2a, 2026-10-10)`. The
generator no longer writes or checks them (`tools/catalog/generate.py`, "graduated"); a fix goes into the workload directory.

| Workload | Result | Runtime | Bench median (bench twin, A10G) | Repeats |
| --- | --- | --- | --- | --- |
| `bge-m3-pytorch` | PASS | torch 2.14.1+cu130 | fwd/s 67.86, peak GB 1.15 | 5 |
| `bge-reranker-large-pytorch` | PASS | torch 2.14.1+cu130 | fwd/s 69.31, peak GB 1.13 | 5 |
| `bge-reranker-v2-m3-pytorch` | PASS | torch 2.14.1+cu130 | fwd/s 68.74, peak GB 1.15 | 5 |
| `canary-1b-v2-pytorch` | PASS | torch 2.14.1+cu130 | audio s/s 29.79 | 5 |
| `chatterbox-pytorch` | PASS | torch 2.6.0 | audio s/s 1.22 | 5 |
| `chronos-2-pytorch` | PASS | torch 2.14.1+cu130 | forecasts/s 36.08 | 5 |
| `chronos-bolt-base-pytorch` | PASS | torch 2.14.1+cu130 | forecasts/s 31.09 | 5 |
| `chronos-t5-large-pytorch` | PASS | torch 2.14.1+cu130 | forecasts/s 1.17 | 5 |
| `depth-anything-v2-large-pytorch` | PASS | torch 2.14.1+cu130 | img/s 8.93, peak GB 1.68 | 5 |
| `depth-anything-v2-small-pytorch` | PASS | torch 2.14.1+cu130 | img/s 52.41, peak GB 0.22 | 5 |
| `dinov2-giant-pytorch` | PASS | torch 2.14.1+cu130 | img/s 10.81, peak GB 4.59 | 5 |
| `f5-tts-pytorch` | PASS | torch 2.14.1+cu130 | audio s/s 2.21 | 5 |
| `flux2-klein-4b-diffusers` | PASS | torch 2.14.1+cu130 | peak GB 18.6, s/image 3.224, steps/s 1.241 | 5 |
| `glm-ocr-pytorch` | PASS | torch 2.14.1+cu130 | decode tok/s 51.9, peak GB 2.29, prefill img/s 14.851 | 5 |
| `olmocr-2-7b-pytorch` | PASS | torch 2.14.1+cu130 | decode tok/s 28.8, peak GB 16.68, prefill img/s 4.971 | 5 |
| `paddleocr-vl-pytorch` | PASS | torch 2.14.1+cu130 | decode tok/s 19.15, peak GB 1.92, prefill img/s 16.909 | 5 |
| `parakeet-tdt-0p6b-v3-pytorch` | PASS | torch 2.14.1+cu130 | audio s/s 111.3 | 5 |
| `qwen3-asr-1p7b-pytorch` | PASS | torch 2.14.1+cu130 | audio s/s 10.88 | 5 |
| `qwen3-reranker-0p6b-pytorch` | PASS | torch 2.14.1+cu130 | fwd/s 25.44, peak GB 1.32 | 5 |
| `qwen3-reranker-4b-pytorch` | PASS | torch 2.14.1+cu130 | fwd/s 12.59, peak GB 8.18 | 5 |
| `qwen3-reranker-8b-pytorch` | PASS | torch 2.14.1+cu130 | fwd/s 7.3, peak GB 16.52 | 5 |
| `sam2p1-hiera-large-pytorch` | PASS | torch 2.14.1+cu130 | img/s 4.84, peak GB 1.42 | 5 |
| `siglip2-giant-opt-patch16-384-pytorch` | PASS | torch 2.14.1+cu130 | img/s 8.64, peak GB 7.54 | 5 |
| `siglip2-so400m-patch16-512-pytorch` | PASS | torch 2.14.1+cu130 | img/s 10.14, peak GB 4.61 | 5 |
| `llamacpp-codestral-22b-v01` | PASS | llama.cpp b11447 | pp512 1388.32, tg128 34.16 | 5; spread 0.0% |
| `llamacpp-deepseek-r1-distill-llama-8b` | PASS | llama.cpp b11447 | pp512 3987.44, tg128 58.13 | 5; spread 0.0% |
| `llamacpp-deepseek-r1-distill-qwen-14b` | PASS | llama.cpp b11447 | pp512 2184.34, tg128 50.32 | 5; spread 0.0% |
| `llamacpp-devstral-small-2-24b-2512` | PASS | llama.cpp b11447 | pp512 1503.45, tg128 31.88 | 5; spread 0.1% |
| `llamacpp-falcon-h1r-7b` | PASS | llama.cpp b11447 | pp512 3042.53, tg128 50.78 | 5; spread 0.1% |
| `llamacpp-falcon3-10b-instruct` | PASS | llama.cpp b11447 | pp512 3063.96, tg128 43.77 | 5; spread 0.0% |
| `llamacpp-gemma4-12b-it` | PASS | llama.cpp b11447 | pp512 2630.86, tg128 59.44 | 5; spread 0.1% |
| `llamacpp-gemma4-26b-a4b-it` | PASS | llama.cpp b11447 | pp512 3711.63, tg128 122.95 | 5; spread 0.1% |
| `llamacpp-gemma4-31b-it` | PASS | llama.cpp b11447 | pp512 1082.91, tg128 25.31 | 5; spread 0.0% |
| `llamacpp-gemma4-e2b-it` | PASS | llama.cpp b11447 | pp512 8559.53, tg128 193.33 | 5; spread 0.1% |
| `llamacpp-glm47-flash` | PASS | llama.cpp b11447 | pp512 2729.01, tg128 111.43 | 5; spread 0.1% |
| `llamacpp-llama31-8b-instruct` | PASS | llama.cpp b11447 | pp512 3984.23, tg128 58.13 | 5; spread 0.0% |
| `llamacpp-magistral-small-2509` | PASS | llama.cpp b11447 | pp512 1514.43, tg128 32.83 | 5; spread 0.0% |
| `llamacpp-ministral-3-14b-instruct-2512` | PASS | llama.cpp b11447 | pp512 2483.52, tg128 33.63 | 5; spread 0.1% |
| `llamacpp-ministral-3-14b-reasoning-2512` | PASS | llama.cpp b11447 | pp512 2482.93, tg128 33.64 | 5; spread 0.1% |
| `llamacpp-nemotron-nano-9b-v2` | PASS | llama.cpp b11447 | pp512 2878.47, tg128 49.15 | 5; spread 0.0% |
| `llamacpp-north-mini-code-1p0` | PASS | llama.cpp b11447 | pp512 3201.2, tg128 143.52 | 5; spread 0.1% |
| `llamacpp-orpheus-3b-0p1-ft` | PASS | llama.cpp b11447 | pp512 7617.06, tg128 120.53 | 5; spread 0.0% |
| `llamacpp-phi4-mini-instruct` | PASS | llama.cpp b11447 | pp512 7419.82, tg128 105.81 | 5; spread 0.0% |
| `llamacpp-phi4-reasoning-plus` | PASS | llama.cpp b11447 | pp512 2409.54, tg128 51.22 | 5; spread 0.0% |
| `llamacpp-qwen3-coder-30b-a3b-instruct` | PASS | llama.cpp b11447 | pp512 3378.92, tg128 155.73 | 5; spread 0.1% |
| `llamacpp-qwen3-embedding-8b` | PASS | llama.cpp b11447 | pp512 3876.91, tg128 56.86 | 5; spread 0.0% |
| `llamacpp-qwen35-9b` | PASS | llama.cpp b11447 | pp512 3267.3, tg128 53.08 | 5; spread 0.0% |
| `llamacpp-qwen38-27b` | PASS | llama.cpp b11447 | pp512 1097.26, tg128 24.2 | 5; spread 0.0% |
| `whisper-cpp-large-v3-turbo` | PASS | whisper.cpp v1.9.5 | decode_ms_per_run 1.31, encode_ms_per_run 85.45 | 5 |

Bench numbers are medians of 5 runs on one A10G at its default clocks; the first column of a `llamacpp-*` row is pp512 (prompt processing) and the
second tg128 (generation), tokens per second. Peak GB is the PyTorch allocator's peak. A ranking of cards by these numbers is a fact about this
A10G, this pool and these run parameters, not about the card.

What the first run changed (all in run scripts, pins and shared Python bodies, each with the evidence in the manifest notes):

| Workload | First-run problem | Fix |
| --- | --- | --- |
| `qwen3-reranker-0p6b`, `-4b`, `-8b` | the margin check failed: P(yes) of the two irrelevant passages differs by 7e-5 to 7e-4, so noise orders them | the result is the scores and the top passage only (`ranking: "1"`); the checked margin is top vs runner-up (0.83 to 0.995) |
| `qwen3-asr-1p7b` | the transformers pipeline cannot drive it (beam search by default, features passed as `input_ids`); Qwen's checkpoint is in the layout of its own `qwen-asr` package (`thinker_config`, `thinker.*` weights, encoder `model_type` `qwen3_asr_audio_encoder`), which transformers 5.19.0's native class loads as every weight UNEXPECTED, i.e. a random model | `asr.py` engine `qwen3-asr`: config from `thinker_config`, a key mapping (708 of 708 tensors match), greedy decode through the processor's chat template; the transcript is exact (WER 0.000) |
| `parakeet-tdt-0p6b-v3` | the feature extractor needs `librosa` | `requirements-asr.txt` (`librosa==1.0.0`) |
| `chatterbox` | `chatterbox-tts` 0.1.7 pins torch 2.6.0, numpy 1.26.4, transformers 5.2.0, safetensors 0.5.3; `resemble-perth` needs `pkg_resources` (setuptools 81 removed it), else `PerthImplicitWatermarker` is `None` | own venv (`PW_VENV_NAME=chatterbox`, `PW_TORCH_ANY_CUDA=1`), `setuptools==80.10.2` |
| `f5-tts` | torchcodec needs the system FFmpeg libraries (`libavutil.so.56`); the vocabulary was taken from the wrong directory of two identical ones | own venv, `apt install ffmpeg` (a system prerequisite), vocabulary from the checkpoint's directory |
| `canary-1b-v2`, `chronos-*` | the shared requirements file mixed packages that cannot be installed together | `requirements-nemo.txt` (own venv), `requirements-chronos.txt` (five packages on the shared venv); `requirements-speech.txt` stays as the union list for reference only |

Findings worth knowing:

- `llamacpp-gemma4-12b-it` passes, but its reference is degenerate: with the raw prompt this checkpoint answers `011` (a longer decode is
  `0111111...`), on the GPU and on the CPU alike, for four different prompts, while the other Gemma 4 GGUFs (E2B, 26B-A4B, 31B) answer ` Paris`. In chat
  mode it starts sensibly (`<|channel>thought ...`), and the loader warns that token 212 `</s>` is not control-type. The reference documents
  determinism, not quality; a chat-template task would test this model better.
- `bin/pw` must run each of these with the venv that matches the workload; the DLAMI's `LD_LIBRARY_PATH` (CUDA 13.2 and 12.9) breaks every
  PyTorch workload (`CUBLAS_STATUS_NOT_INITIALIZED`) and has to be unset for them (`docs/first-gpu-run-a10g.md`); the llama.cpp ones need it.
- One run of `llamacpp-devstral-small-2-24b-2512` failed ("unable to create context") and one NeMo attempt ran out of memory because other jobs
  were sharing the card while these were being collected; both were rerun alone (3 more passes each) and the shared-card runs are not counted.
- Chatterbox's `duration_s` (2.24 s of audio, reported as 2.0) and F5-TTS's (4.41 s, reported as 4.5) are stable on this stack but sit close
  to the half-second rounding; another library stack may flip them.
- `torchaudio` stops at 2.11.0, so F5-TTS runs with torchaudio 2.11.0 beside torch 2.14.1; it imported and ran without error.

To repeat the run: `uv python install 3.12`; `uv venv --python 3.12 ~/.cache/pantheonworkloads/venvs/cu130` and the three requirement files as in
`docs/first-gpu-run-a10g.md` (add `requirements-vlm.txt`, `requirements-asr.txt` and `requirements-chronos.txt`); the `nemo`, `f5` and `chatterbox`
venvs from `requirements-nemo.txt`, `requirements-f5.txt` and `requirements-chatterbox.txt` (torch pins and `requirements-common.txt` first, except for
chatterbox); `tools/llamacpp/build.sh cuda <prefix>` with `PW_CUDA_ARCHS=86 PW_BUILD_JOBS=8 LLAMACPP_EXTRA_TARGETS=llama-embedding
LLAMACPP_BUILD_EXAMPLES=1`; `apt install ffmpeg`; run the PyTorch workloads with `env -u LD_LIBRARY_PATH`.

### Stage 2a, Tier B (2026-10-10): the 20 workloads of 24 to 44 GiB on one NVIDIA L40S

Every catalog workload with `requires.gpus == 1` and `24 <= gpu_memory_gb <= 44` (20 functional workloads and their 20 bench twins) was run with
`--target gpu` on one on-demand AWS g6e.xlarge in us-east-1b: NVIDIA L40S (46068 MiB, 44.4 GiB as torch sees it, sm_89), driver 595.91.07, Ubuntu 22.04
(Deep Learning Base OSS Nvidia Driver AMI 20261006), 4 vCPU, 32 GB RAM, 900 GB gp3 (raised to 1000 MB/s and 12000 IOPS). About 11.5 h of instance time
(model downloads, the llama.cpp build for `PW_CUDA_ARCHS=89` (an sm_86 A10G build is not reusable), first-run fixes, 4 or more passes per functional
workload and 5-repeat benches). Software: llama.cpp b11447 (CUDA 13.2, `-ngl 99`), torch 2.14.1+cu130, transformers 5.19.0, diffusers 0.41.0 (Python 3.12 `uv` venv as in
`docs/first-gpu-run-a10g.md`). All 20 pass: each reference was recorded on the card with `bin/pw record --target gpu` and reproduced exactly (the
diffusion workloads even bit-for-bit: the informational pixel hash was identical in every run) in at least 3 further runs; the bench twins were run
with `--bench --repeat 5` from a clean checkout of the pushed commit (`repo_dirty: false`, `repo_commit` is that commit; the diffusion twins use
`PW_DIFFUSION_RUNS=1`, i.e. one timed generation per repeat after one warm-up generation, so the 5 repeats are 5 timed generations).

| Workload | Result | Reference | Further passing runs | Bench (medians of 5) |
| --- | --- | --- | ---: | --- |
| `llamacpp-qwen3-32b` | PASS | ' Paris. This\n\n' | 4 | pp512 2452.62, tg128 33.73 tokens/s |
| `llamacpp-qwen25-coder-32b-instruct` | PASS | ' Paris. The\n\n' | 3 | pp512 2465.47, tg128 33.63 tokens/s |
| `llamacpp-deepseek-r1-distill-qwen-32b` | PASS | ' Paris. Paris\n\n' | 3 | pp512 2463.6, tg128 33.63 tokens/s |
| `llamacpp-qwen36-27b` | PASS | ' Paris.\n\n\n\n' | 3 | pp512 2589.59, tg128 33.57 tokens/s |
| `llamacpp-qwen36-35b-a3b` | PASS | ' Paris, a\n\n' | 3 | pp512 6234.63, tg128 146.61 tokens/s |
| `llamacpp-command-r-08-2024` | PASS | ' Paris.\n\n\n' | 3 | pp512 2581.92, tg128 34.59 tokens/s |
| `llamacpp-exaone-4p5-33b` | PASS | ' Paris.  \n\n' | 3 | pp512 2483.91, tg128 33.88 tokens/s |
| `llamacpp-falcon-h1-34b-instruct` | PASS | ' Paris.\n\n\n\n' | 3 | pp512 2183.57, tg128 29.32 tokens/s |
| `llamacpp-nemotron-3-nano-30b-a3b` | PASS | ' Paris." No\n\n' | 3 | pp512 7255.75, tg128 168.0 tokens/s |
| `llamacpp-nemotron-3p5-lightning-30b-a3b` | PASS | ' Paris.  \n\n' | 3 | pp512 7057.12, tg128 177.69 tokens/s |
| `llamacpp-kimi-linear-48b-a3b-instruct` | PASS | ' Paris. The\n\n' | 3 | pp512 5155.12, tg128 162.55 tokens/s |
| `wan21-t2v-1p3b-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 37.377 s per generation, 0.803 steps/s, peak 19.09 GB |
| `ltx-video-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 4.684 s per generation, 6.404 steps/s, peak 15.96 GB |
| `cogvideox-2b-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 49.349 s per generation, 0.608 steps/s, peak 29.54 GB |
| `wan22-ti2v-5b-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 21.914 s per generation, 1.369 steps/s, peak 27.65 GB |
| `flux1-schnell-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 2.252 s per generation, 1.776 steps/s, peak 36.3 GB |
| `qwen-image-2p1-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 12.068 s per generation, 2.486 steps/s, peak 39.53 GB |
| `cogvideox-5b-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 142.247 s per generation, 0.211 steps/s, peak 36.98 GB |
| `hunyuanvideo-15-720p-t2v-diffusers` | PASS | statistics (mean, std, block means, gradients) | 3 | 175.541 s per generation, 0.171 steps/s, peak 37.93 GB |
| `pixtral-12b-2409-pytorch` | PASS | Red, white and blue. | 3 | decode 27.0 tokens/s, prefill 5.932 images/s, peak 25.95 GB |

No tolerance was changed. Peak memory is `torch.cuda.max_memory_allocated`; every workload fit its declared `requires` (Qwen-Image-2.1 is the tightest: 39.5 GB
against the 40 GiB class). The diffusion tolerances remain guesses for cross-card comparison: on this card two runs never differed at all.

**First-run problems, all fixed in the scripts** (not in the manifests' numbers):

- `QwenImage21Pipeline` and `HunyuanVideo15Pipeline` have no `guidance_scale` argument (`TypeError`). `diffusion.run(guidance=None)` now omits it: Qwen-Image-2.1 then
  runs with the pipeline default (no classifier-free guidance; it takes `true_cfg_scale`), HunyuanVideo-1.5 keeps the 6.0 stored in its `guider` config.
- Qwen-Image-2.1's VAE is 4-channel: the output is RGBA and the statistics reshape failed; the statistics now composite over white (the pipeline does the same for
  its own vision encoder).
- HunyuanVideo-1.5 at 848x480 x 33 frames ran out of memory in the untiled VAE decode (after a complete 30-step denoise, on the 44.4 GiB card);
  `vae_tiling=True` (the diffusers example enables it) fixes it, peak 37.9 GB.
- Pixtral-12B: the generic image+text prompt and processor worked, but "Describe this image in one sentence." has an exact bf16 tie (top-1 = top-2) at the 4th token
  and a 0.0625 gap in fp16. Five prompts were compared on the card (bf16 / fp16 minimum top1-top2 gap over the generated tokens: description 0 / 0; "What is the woman ... wearing?" 0.125 / 0.047;
  "main subject" 0 / 0.14; "List three objects" 0.125 / 0.42; "What color is the flag on the left? Answer with one word." 1.0 / 0.92). The last one is used (answer "Red, white and blue.", floor 0.5 kept).
- Not a workload problem but worth knowing: two functional reruns (`cogvideox-2b`, `wan22-ti2v-5b`) and one llama.cpp rerun (`nemotron-3-nano`) failed with CUDA out of
  memory / "unable to create context" because another workload of the same session was using the card at the same time; they passed when repeated alone. Do not run
  two of these at once on one 48 GB card.

**Rig traps** (so the next run is faster): gp3 defaults (125 MB/s, 3000 IOPS) made the first model loads and sha256 checks take 10+ minutes while downloads ran; raise
throughput/IOPS with `aws ec2 modify-volume` and `echo 16384 > /sys/block/nvme0n1/queue/read_ahead_kb`. Every run re-verifies the sha256 of the whole model (18 to 31 GiB),
which is most of a llama.cpp functional run's 1.5 to 3 minutes. Loading the fp32-stored diffusion checkpoints (HunyuanVideo-1.5, Wan2.2) takes 5 to 8 minutes on 4 vCPU / 32 GB RAM; a
bench twin therefore takes 15 to 50 minutes. g6e.2xlarge had no capacity in any us-east-1 AZ at launch time; g6e.xlarge (32 GB RAM) was enough for every model here.

**Still not verified**: other cards (llama.cpp 3-token greedy outputs and Pixtral's bf16 text may differ where two tokens nearly tie; the diffusion tolerances were never
exercised), other runtime versions, and the quality of the generated pictures and clips (the checks are statistics, not a judgement of the content).

## The registry

<!-- registry:start -->

**147 entries: 129 with workloads written, 18 blocked.** Of the 129 written: 69 verified on a real GPU (reference and bench recorded), 0 blocked at their first run, 60 never run. Hub data read 2026-10-09; runtimes: llama.cpp b11447, vLLM v0.30.0, transformers 5.19.0, diffusers 0.41.0.

| Family | Entries | Workloads written | Blocked |
| --- | ---: | ---: | ---: |
| DeepSeek | 11 | 10 | 1 |
| Meta Llama | 8 | 6 | 2 |
| Qwen | 26 | 26 | 0 |
| Google Gemma | 5 | 4 | 1 |
| Mistral | 13 | 13 | 0 |
| OpenAI gpt-oss | 2 | 2 | 0 |
| GLM | 7 | 7 | 0 |
| Kimi | 5 | 5 | 0 |
| Cohere | 5 | 5 | 0 |
| Falcon | 4 | 3 | 1 |
| Microsoft Phi | 2 | 2 | 0 |
| NVIDIA Nemotron | 5 | 5 | 0 |
| MiniMax | 3 | 3 | 0 |
| Other LLMs | 5 | 5 | 0 |
| Image generation | 9 | 2 | 7 |
| Video generation | 14 | 11 | 3 |
| Speech | 6 | 6 | 0 |
| Retrieval | 3 | 3 | 0 |
| Vision | 8 | 6 | 2 |
| OCR and documents | 3 | 2 | 1 |
| Time series | 3 | 3 | 0 |

Blocked, by reason:

- 10 x gated
- 3 x custom model code
- 2 x gated, and llama.cpp b11447 has no mllama architecture
- 1 x the pipeline loads a Llama-3.1-8B-Instruct text encoder and tokenizer 
- 1 x own inference code
- 1 x the repository has no diffusers layout

Largest hardware needs among the written workloads (weights plus margin, per the formulas above):

| Workload | Weights GiB | Needs | Total GPU memory |
| --- | ---: | --- | ---: |
| `vllm-hy4-preview` | 1452.8 | 8 x 256 GiB (single node) | 2048 GiB |
| `llamacpp-kimi-k3` | 1405.1 | 8 x 192 GiB (single node) | 1536 GiB |
| `llamacpp-qwen38-2p4t-a95b` | 1220.8 | 8 x 180 GiB (single node) | 1440 GiB |
| `llamacpp-deepseek-v4-pro-0813` | 791.3 | 4 x 256 GiB (single node) | 1024 GiB |
| `llamacpp-kimi-k2-thinking` | 578.6 | 4 x 180 GiB (single node) | 720 GiB |
| `llamacpp-kimi-k26` | 543.6 | 8 x 80 GiB (single node) | 640 GiB |
| `llamacpp-kimi-k27-code` | 543.6 | 8 x 80 GiB (single node) | 640 GiB |
| `vllm-deepseek-v4p1-flash` | 475.3 | 8 x 80 GiB (single node) | 640 GiB |
| `qwen3-vl-235b-a22b-pytorch` | 439.0 | 8 x 80 GiB (single node) | 640 GiB |
| `llamacpp-glm53` | 435.2 | 8 x 80 GiB (single node) | 640 GiB |
| `llamacpp-glm52` | 433.8 | 8 x 80 GiB (single node) | 640 GiB |
| `llamacpp-mistral-large-3-675b-instruct-2512` | 379.0 | 8 x 80 GiB (single node) | 640 GiB |
| `llamacpp-deepseek-v3p2` | 377.6 | 8 x 80 GiB (single node) | 640 GiB |
| `llamacpp-deepseek-v3p1-terminus` | 377.6 | 8 x 80 GiB (single node) | 640 GiB |
| `llamacpp-deepseek-r1-0528` | 377.1 | 8 x 80 GiB (single node) | 640 GiB |

### DeepSeek

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| DeepSeek-V4-Pro-0813 | `unsloth/DeepSeek-V4-Pro-0813-GGUF` `6d053616d152293da72569f56b861466f10ace7d` (base `deepseek-ai/DeepSeek-V4-Pro-0813` `72e1d3230f6c080a530b0a1d46f8eb4602340597`) | 1.65T | MIT | no | llama.cpp (GGUF) | UD-Q4_K_XL | 791.3 | 4 x 256 GiB (791.3/4*1.05 + 3 = 210.7) | manifest written | `llamacpp-deepseek-v4-pro-0813`, `llamacpp-bench-deepseek-v4-pro-0813` |
| DeepSeek-V4-Flash-0731 | `ggml-org/DeepSeek-V4-Flash-0731-GGUF` `f559fd6005309e5f6bd650342ee8711ff189b3b8` (base `deepseek-ai/DeepSeek-V4-Flash-0731` `7872f01b1d1fe23eabc4c98b48bffcef5a386062`) | 304B | MIT | no | llama.cpp (GGUF) | MXFP4 | 144.3 | 4 x 44 GiB (144.3/4*1.05 + 3 = 40.9) | manifest written | `llamacpp-deepseek-v4-flash-0731`, `llamacpp-bench-deepseek-v4-flash-0731` |
| DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash` `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | 763B | MIT | no | vLLM | as stored | 475.3 | 8 x 80 GiB ((475.3/8 + 6)/0.9 = 72.7) | manifest written | `vllm-deepseek-v4p1-flash`, `vllm-bench-deepseek-v4p1-flash` |
| DeepSeek-V3.2 | `unsloth/DeepSeek-V3.2-GGUF` `a787696863bafd5c736955ef81cc869a0bf6178a` (base `deepseek-ai/DeepSeek-V3.2` `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`) | 685B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 377.6 | 8 x 80 GiB (377.6/8*1.05 + 3 = 52.6) | manifest written | `llamacpp-deepseek-v3p2`, `llamacpp-bench-deepseek-v3p2` |
| DeepSeek-V3.1-Terminus | `unsloth/DeepSeek-V3.1-Terminus-GGUF` `fe48342e95b8b3ca863189919605651fc2e88f4e` (base `deepseek-ai/DeepSeek-V3.1-Terminus` `19510d6dc61f79dbd925bd51ee8a9081c509a4b6`) | 685B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 377.6 | 8 x 80 GiB (377.6/8*1.05 + 3 = 52.6) | manifest written | `llamacpp-deepseek-v3p1-terminus`, `llamacpp-bench-deepseek-v3p1-terminus` |
| DeepSeek-R1-0528 | `unsloth/DeepSeek-R1-0528-GGUF` `72f5d5bd7f0821bc0be039b094cc1501bdbf232a` (base `deepseek-ai/DeepSeek-R1-0528` `4236a6af538feda4548eca9ab308586007567f52`) | 685B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 377.1 | 8 x 80 GiB (377.1/8*1.05 + 3 = 52.5) | manifest written | `llamacpp-deepseek-r1-0528`, `llamacpp-bench-deepseek-r1-0528` |
| DeepSeek-R1-Distill-Llama-70B | `unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF` `732dd974083ea5877d7b6d788b36fe7c2e5eab36` (base `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` `b1c0b44b4369b597ad119a196caf79a9c40e141e`) | 70.6B | Llama 3.3 Community Licence (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 39.6 | 1 x 48 GiB (39.6/1*1.05 + 3 = 44.6) | manifest written | `llamacpp-deepseek-r1-distill-llama-70b`, `llamacpp-bench-deepseek-r1-distill-llama-70b` |
| DeepSeek-R1-Distill-Llama-8B | `unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF` `615f8936e16dfde29dcc00be71145d4d5ce8ed53` (base `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` `6a6f4aa4197940add57724a7707d069478df56b1`) | 8.03B | Llama 3.1 Community Licence (restricted) | no | llama.cpp (GGUF) | Q8_0 | 8.0 | 1 x 15 GiB (8.0/1*1.05 + 3 = 11.4) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-deepseek-r1-distill-llama-8b`, `llamacpp-bench-deepseek-r1-distill-llama-8b` |
| DeepSeek-R1-Distill-Qwen-32B | `unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF` `1938d05cc893a60f37be1dc16e7465038f4fca63` (base `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` `711ad2ea6aa40cfca18895e8aca02ab92df1a746`) | 32.8B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 18.5 | 1 x 24 GiB (18.5/1*1.05 + 3 = 22.4) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-deepseek-r1-distill-qwen-32b`, `llamacpp-bench-deepseek-r1-distill-qwen-32b` |
| DeepSeek-R1-Distill-Qwen-14B | `unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF` `7b05b58b41f623e66fc74cd27b35475267b2f3e3` (base `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` `1df8507178afcc1bef68cd8c393f61a886323761`) | 14.8B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 8.4 | 1 x 15 GiB (8.4/1*1.05 + 3 = 11.8) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-deepseek-r1-distill-qwen-14b`, `llamacpp-bench-deepseek-r1-distill-qwen-14b` |
| DeepSeek-OCR-2 | `deepseek-ai/DeepSeek-OCR-2` `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | 3.39B | Apache-2.0 | no | transformers | as stored | 6.3 | 1 x 15 GiB (6.3/1*1.1 + 4 = 10.9) | blocked: custom model code (auto_map ['AutoConfig', 'AutoModel']); model_type 'deepseek_vl_v2' is not in transformers 5.19.0 and trust_remote_code is not used | - |

### Meta Llama

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Llama-3.1-8B-Instruct | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` `bf5b95e96dac0462e2a09145ec66cae9a3f12067` (base `meta-llama/Llama-3.1-8B-Instruct` `0e9e39f249a16976918f6564b8830bc894c89659`) | 8.03B | Llama 3.1 Community Licence (restricted) | official: manual; downloaded copy: no | llama.cpp (GGUF) | Q8_0 | 8.0 | 1 x 15 GiB (8.0/1*1.05 + 3 = 11.4) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, then set HF_TOKEN) | `llamacpp-llama31-8b-instruct`, `llamacpp-bench-llama31-8b-instruct` |
| Llama-3.1-70B-Instruct | `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF` `83fb6e83d0a8aada42d499259bc929d922e9a558` (base `meta-llama/Llama-3.1-70B-Instruct` `1605565b47bb9346c5515c34102e054115b4f98b`) | 70.6B | Llama 3.1 Community Licence (restricted) | official: manual; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 39.6 | 1 x 48 GiB (39.6/1*1.05 + 3 = 44.6) | manifest written (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct, then set HF_TOKEN) | `llamacpp-llama31-70b-instruct`, `llamacpp-bench-llama31-70b-instruct` |
| Llama-3.1-405B-Instruct | `bullerwins/Meta-Llama-3.1-405B-Instruct-GGUF` `ad731614180e6fd90426ce87b5ef44f64c897aad` (base `meta-llama/Llama-3.1-405B-Instruct` `be673f326cab4cd22ccfef76109faf68e41aa5f1`) | 406B | Llama 3.1 Community Licence (restricted) | official: manual; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 228.8 | 4 x 80 GiB (228.8/4*1.05 + 3 = 63.1) | manifest written (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct, then set HF_TOKEN) | `llamacpp-llama31-405b-instruct`, `llamacpp-bench-llama31-405b-instruct` |
| Llama-3.3-70B-Instruct | `bartowski/Llama-3.3-70B-Instruct-GGUF` `b6c5c9f176f3279204034e1d16d393105e95cb88` (base `meta-llama/Llama-3.3-70B-Instruct` `6f6073b423013f6a7d4d9f39144961bfbfbc386b`) | 70.6B | Llama 3.3 Community Licence (restricted) | official: manual; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 39.6 | 1 x 48 GiB (39.6/1*1.05 + 3 = 44.6) | manifest written (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct, then set HF_TOKEN) | `llamacpp-llama33-70b-instruct`, `llamacpp-bench-llama33-70b-instruct` |
| Llama-4-Scout-17B-16E-Instruct | `ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF` `42675345da11ade9203a5187595da7b74d4ff2ac` (base `meta-llama/Llama-4-Scout-17B-16E-Instruct` `92f3b1597a195b523d8d9e5700e57e4fbb8f20d3`) | 109B | Llama 4 Community License (restricted) | official: manual; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 60.9 | 1 x 80 GiB (60.9/1*1.05 + 3 = 66.9) | manifest written (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct, then set HF_TOKEN) | `llamacpp-llama4-scout-17b-16e-instruct`, `llamacpp-bench-llama4-scout-17b-16e-instruct` |
| Llama-4-Maverick-17B-128E-Instruct | `unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF` `41032e5471dd6ea5b349062d978626215d6c5bba` (base `meta-llama/Llama-4-Maverick-17B-128E-Instruct` `73d14711bcc77c16df3470856949c3764056b617`) | 402B | Llama 4 Community License (restricted) | official: manual; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 226.1 | 4 x 80 GiB (226.1/4*1.05 + 3 = 62.3) | manifest written (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct, then set HF_TOKEN) | `llamacpp-llama4-maverick-17b-128e-instruct`, `llamacpp-bench-llama4-maverick-17b-128e-instruct` |
| Llama-3.2-11B-Vision-Instruct | `meta-llama/Llama-3.2-11B-Vision-Instruct` `9eb2daaa8597bf192a8b0e73f848f3a102794df5` | 10.7B | Llama 3.2 Community Licence (restricted) | manual | transformers | as stored | 19.9 | 1 x 40 GiB (19.9/1*1.1 + 4 = 25.9) | blocked: gated, and llama.cpp b11447 has no mllama architecture. Access: accept the terms at https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct (gated: manual), then set HF_TOKEN | - |
| Llama-3.2-90B-Vision-Instruct | `meta-llama/Llama-3.2-90B-Vision-Instruct` `e305d2a43a4adc6987308fe7d896fb8ec5f1a5d8` | 88.6B | Llama 3.2 Community Licence (restricted) | manual | transformers | as stored | 165.0 | 4 x 80 GiB (165.0/4*1.1 + 4 = 49.4) | blocked: gated, and llama.cpp b11447 has no mllama architecture. Access: accept the terms at https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct (gated: manual), then set HF_TOKEN | - |

### Qwen

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct-GGUF` `7ca3bc388f97b264c4283bc9bf1055e2abc38441` (base `Qwen/Qwen2.5-72B-Instruct` `495f39366efef23836d0cfae4fbe635880d2be31`) | 72.7B | Qwen License (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 41.0 | 1 x 48 GiB (41.0/1*1.05 + 3 = 46.0) | manifest written | `llamacpp-qwen25-72b-instruct`, `llamacpp-bench-qwen25-72b-instruct` |
| Qwen2.5-Coder-32B-Instruct | `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF` `9d3053fce650fe1cdbdb75998c2a87add9d178ef` (base `Qwen/Qwen2.5-Coder-32B-Instruct` `381fc969f78efac66bc87ff7ddeadb7e73c218a7`) | 32.8B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 18.5 | 1 x 24 GiB (18.5/1*1.05 + 3 = 22.4) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-qwen25-coder-32b-instruct`, `llamacpp-bench-qwen25-coder-32b-instruct` |
| Qwen3-32B | `Qwen/Qwen3-32B-GGUF` `938a7432affaec9157f883a87164e2646ae17555` (base `Qwen/Qwen3-32B` `9216db5781bf21249d130ec9da846c4624c16137`) | 32.8B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 18.4 | 1 x 24 GiB (18.4/1*1.05 + 3 = 22.3) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 4 further runs | `llamacpp-qwen3-32b`, `llamacpp-bench-qwen3-32b` |
| Qwen3-235B-A22B-Instruct-2507 | `unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF` `437d6915c5c512869e4cf8b16b840bbe3fb172bc` (base `Qwen/Qwen3-235B-A22B-Instruct-2507` `ac9c66cc9b46af7306746a9250f23d47083d689e`) | 235B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 132.4 | 2 x 80 GiB (132.4/2*1.05 + 3 = 72.5) | manifest written | `llamacpp-qwen3-235b-a22b-instruct-2507`, `llamacpp-bench-qwen3-235b-a22b-instruct-2507` |
| Qwen3-235B-A22B-Thinking-2507 | `unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF` `3eba50b4e3ea2c0df1c43e8c8f5ce2db22645e81` (base `Qwen/Qwen3-235B-A22B-Thinking-2507` `6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5`) | 235B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 132.4 | 2 x 80 GiB (132.4/2*1.05 + 3 = 72.5) | manifest written | `llamacpp-qwen3-235b-a22b-thinking-2507`, `llamacpp-bench-qwen3-235b-a22b-thinking-2507` |
| Qwen3-Coder-480B-A35B-Instruct | `unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF` `b86deeefd82f1a3374c5536dfc1dd0ce27ac092d` (base `Qwen/Qwen3-Coder-480B-A35B-Instruct` `9d90cf8fca1bf7b7acca42d3fc9ae694a2194069`) | 480B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 270.1 | 4 x 80 GiB (270.1/4*1.05 + 3 = 73.9) | manifest written | `llamacpp-qwen3-coder-480b-a35b-instruct`, `llamacpp-bench-qwen3-coder-480b-a35b-instruct` |
| Qwen3-Coder-30B-A3B-Instruct | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` `b17cb02dd882d5b6ab62fc777ad2995f19668350` (base `Qwen/Qwen3-Coder-30B-A3B-Instruct` `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`) | 30.5B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 17.3 | 1 x 22 GiB (17.3/1*1.05 + 3 = 21.1) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-qwen3-coder-30b-a3b-instruct`, `llamacpp-bench-qwen3-coder-30b-a3b-instruct` |
| Qwen3-Coder-Next | `Qwen/Qwen3-Coder-Next-GGUF` `b82fb7382639d97b38fa7672e526c760c2fb358e` (base `Qwen/Qwen3-Coder-Next` `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`) | 79.7B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 45.1 | 1 x 80 GiB (45.1/1*1.05 + 3 = 50.3) | manifest written | `llamacpp-qwen3-coder-next`, `llamacpp-bench-qwen3-coder-next` |
| Qwen3-Next-80B-A3B-Instruct | `Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF` `4c8630cf7af926a9c5095cb4bbbbc65d36e20f77` (base `Qwen/Qwen3-Next-80B-A3B-Instruct` `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`) | 81.3B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 45.1 | 1 x 80 GiB (45.1/1*1.05 + 3 = 50.3) | manifest written | `llamacpp-qwen3-next-80b-a3b-instruct`, `llamacpp-bench-qwen3-next-80b-a3b-instruct` |
| Qwen3.5-397B-A17B | `unsloth/Qwen3.5-397B-A17B-GGUF` `da33c16fa4440f831149fcf53b98a22bc07785e5` (base `Qwen/Qwen3.5-397B-A17B` `8472618112abcbd45acbcdc58436aff4233c23f7`) | 403B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 227.3 | 4 x 80 GiB (227.3/4*1.05 + 3 = 62.7) | manifest written | `llamacpp-qwen35-397b-a17b`, `llamacpp-bench-qwen35-397b-a17b` |
| Qwen3.5-122B-A10B | `unsloth/Qwen3.5-122B-A10B-GGUF` `51eab4d59d53f573fb9206cb3ce613f1d0aa392b` (base `Qwen/Qwen3.5-122B-A10B` `dc4d348443bc740c68e2d77492492c11606384d5`) | 125B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 71.3 | 1 x 80 GiB (71.3/1*1.05 + 3 = 77.8) | manifest written | `llamacpp-qwen35-122b-a10b`, `llamacpp-bench-qwen35-122b-a10b` |
| Qwen3.5-9B | `unsloth/Qwen3.5-9B-GGUF` `3885219b6810b007914f3a7950a8d1b469d598a5` (base `Qwen/Qwen3.5-9B` `c202236235762e1c871ad0ccb60c8ee5ba337b9a`) | 9.65B | Apache-2.0 | no | llama.cpp (GGUF) | Q8_0 | 8.9 | 1 x 15 GiB (8.9/1*1.05 + 3 = 12.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-qwen35-9b`, `llamacpp-bench-qwen35-9b` |
| Qwen3.6-35B-A3B | `ggml-org/Qwen3.6-35B-A3B-GGUF` `baec3ebee244827cda0f4557eafa8b28f7545fa6` (base `Qwen/Qwen3.6-35B-A3B` `995ad96eacd98c81ed38be0c5b274b04031597b0`) | 36B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 19.0 | 1 x 24 GiB (19.0/1*1.05 + 3 = 23.0) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-qwen36-35b-a3b`, `llamacpp-bench-qwen36-35b-a3b` |
| Qwen3.6-27B | `ggml-org/Qwen3.6-27B-GGUF` `8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8` (base `Qwen/Qwen3.6-27B` `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`) | 27.8B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 17.8 | 1 x 24 GiB (17.8/1*1.05 + 3 = 21.7) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-qwen36-27b`, `llamacpp-bench-qwen36-27b` |
| Qwen3.8-27B | `ggml-org/Qwen3.8-27B-GGUF` `71bc7b627595dc8a91039addd9c791ae548d6747` (base `Qwen/Qwen3.8-27B` `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) | 27.8B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 17.7 | 1 x 22 GiB (17.7/1*1.05 + 3 = 21.6) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-qwen38-27b`, `llamacpp-bench-qwen38-27b` |
| Qwen3.8-Flash-Next | `bartowski/Qwen3.8-Flash-Next-GGUF` `928589fdb66c6ff07f22ac561e3fbce76553548f` (base `Qwen/Qwen3.8-Flash-Next` `de4b8e4d43b917e7706784d8bb445c9af86a3540`) | 180B | Qwen Community License 1.0 (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 111.4 | 2 x 80 GiB (111.4/2*1.05 + 3 = 61.5) | manifest written | `llamacpp-qwen38-flash-next`, `llamacpp-bench-qwen38-flash-next` |
| Qwen3.8-2.4T-A95B | `unsloth/Qwen3.8-2.4T-A95B-GGUF` `567d3e6ac26c5474b18311e619c04350fb9a5556` (base `Qwen/Qwen3.8-2.4T-A95B` `207bd685a7e3696cfaff12ded7c6a7ea0f88c996`) | 2.45T | qwen3.8-max (restricted) | no | llama.cpp (GGUF) | UD-IQ4_XS | 1220.8 | 8 x 180 GiB (1220.8/8*1.05 + 3 = 163.2) | manifest written | `llamacpp-qwen38-2p4t-a95b`, `llamacpp-bench-qwen38-2p4t-a95b` |
| Qwen2.5-VL-72B-Instruct | `Qwen/Qwen2.5-VL-72B-Instruct` `89c86200743eec961a297729e7990e8f2ddbc4c5` | 73.4B | Qwen License (restricted) | no | transformers | as stored | 136.7 | 4 x 44 GiB (136.7/4*1.1 + 4 = 41.6) | manifest written | `qwen25-vl-72b-pytorch`, `qwen25-vl-72b-bench` |
| Qwen3-VL-32B-Instruct | `Qwen/Qwen3-VL-32B-Instruct` `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | 33.4B | Apache-2.0 | no | transformers | as stored | 62.1 | 1 x 80 GiB (62.1/1*1.1 + 4 = 72.3) | manifest written | `qwen3-vl-32b-pytorch`, `qwen3-vl-32b-bench` |
| Qwen3-VL-235B-A22B-Instruct | `Qwen/Qwen3-VL-235B-A22B-Instruct` `710c13861be6c466e66de3f484069440b8f31389` | 236B | Apache-2.0 | no | transformers | as stored | 439.0 | 8 x 80 GiB (439.0/8*1.1 + 4 = 64.4) | manifest written | `qwen3-vl-235b-a22b-pytorch`, `qwen3-vl-235b-a22b-bench` |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B-GGUF` `69d0e58a13e463cd99a9b83e3f5fee7c10265fab` (base `Qwen/Qwen3-Embedding-8B` `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`) | 7.57B | Apache-2.0 | no | llama.cpp embedding (GGUF) | Q8_0 | 7.5 | 1 x 15 GiB (7.5/1*1.05 + 3 = 10.9) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-qwen3-embedding-8b`, `llamacpp-bench-qwen3-embedding-8b` |
| Qwen3-Reranker-8B | `Qwen/Qwen3-Reranker-8B` `77d193c791ed757ca307ee72715aa132723da912` | 8.19B | Apache-2.0 | no | transformers | as stored | 15.3 | 1 x 22 GiB (15.3/1*1.1 + 4 = 20.8) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `qwen3-reranker-8b-pytorch`, `qwen3-reranker-8b-bench` |
| Qwen3-Reranker-4B | `Qwen/Qwen3-Reranker-4B` `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | 4.02B | Apache-2.0 | no | transformers | as stored | 7.5 | 1 x 15 GiB (7.5/1*1.1 + 4 = 12.2) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `qwen3-reranker-4b-pytorch`, `qwen3-reranker-4b-bench` |
| Qwen3-Reranker-0.6B | `Qwen/Qwen3-Reranker-0.6B` `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 596M | Apache-2.0 | no | transformers | as stored | 1.1 | 1 x 10 GiB (1.1/1*1.1 + 4 = 5.2) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `qwen3-reranker-0p6b-pytorch`, `qwen3-reranker-0p6b-bench` |
| Qwen3-ASR-1.7B | `Qwen/Qwen3-ASR-1.7B` `7278e1e70fe206f11671096ffdd38061171dd6e5` | 2.35B | Apache-2.0 | no | transformers / NeMo | as stored | 4.4 | 1 x 10 GiB (4.4/1*1.1 + 4 = 8.8) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `qwen3-asr-1p7b-pytorch`, `qwen3-asr-1p7b-bench` |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` `d26bb61231c349cf6b7896fa83353113880e1ba3` | 7.12B | Qwen Research License (restricted) | no | diffusers | as stored | 30.8 | 1 x 40 GiB (max(16.3 largest component, 30.8/1) + 6 = 36.8) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `qwen-image-2p1-diffusers`, `qwen-image-2p1-bench` |

### Google Gemma

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| gemma-4-31B-it | `google/gemma-4-31B-it-qat-q4_0-gguf` `59dde24573e7e61570dba08b18a2e1fe246955ed` (base `google/gemma-4-31B-it` `842da3794eaa0b77d5f08bae87a17459d91ff475`) | 31.3B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_0 | 16.4 | 1 x 22 GiB (16.4/1*1.05 + 3 = 20.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-gemma4-31b-it`, `llamacpp-bench-gemma4-31b-it` |
| gemma-4-26B-A4B-it | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` `d1c082be9cf3c8a514acf63b8761f4b41935842e` (base `google/gemma-4-26B-A4B-it` `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`) | 25.8B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_0 | 13.4 | 1 x 22 GiB (13.4/1*1.05 + 3 = 17.1) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-gemma4-26b-a4b-it`, `llamacpp-bench-gemma4-26b-a4b-it` |
| gemma-4-12B-it | `google/gemma-4-12B-it-qat-q4_0-gguf` `29d097773436b69ff9feafd636ab4cf873786537` (base `google/gemma-4-12B-it` `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`) | 12B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_0 | 6.5 | 1 x 15 GiB (6.5/1*1.05 + 3 = 9.8) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-gemma4-12b-it`, `llamacpp-bench-gemma4-12b-it` |
| gemma-4-E2B-it | `google/gemma-4-E2B-it-qat-q4_0-gguf` `675cff42a74c774d6cb76f76d8eacb49b48c9b93` (base `google/gemma-4-E2B-it` `3e22461f65e89153144f8adb70e3b8c2cc9845a7`) | 5.12B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_0 | 3.1 | 1 x 10 GiB (3.1/1*1.05 + 3 = 6.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-gemma4-e2b-it`, `llamacpp-bench-gemma4-e2b-it` |
| gemma-3-27b-it | `google/gemma-3-27b-it` `005ad3404e59d6023443cb575daa05336842228a` | 27.4B | Gemma Terms of Use (restricted) | manual | llama.cpp (GGUF) | - | - | - | blocked: gated (manual); Gemma Terms of Use. Access: accept the terms at https://huggingface.co/google/gemma-3-27b-it (gated: manual), then set HF_TOKEN | - |

### Mistral

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Mistral-Large-3-675B-Instruct-2512 | `unsloth/Mistral-Large-3-675B-Instruct-2512-GGUF` `922217b859d90a8d94088a1c900fdf03f8fd078e` (base `mistralai/Mistral-Large-3-675B-Instruct-2512` `383ffea2c7d60dfd44ca960e8e691709d4fdb9cd`) | 673B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 379.0 | 8 x 80 GiB (379.0/8*1.05 + 3 = 52.7) | manifest written | `llamacpp-mistral-large-3-675b-instruct-2512`, `llamacpp-bench-mistral-large-3-675b-instruct-2512` |
| Mistral-Medium-3.5-128B | `unsloth/Mistral-Medium-3.5-128B-GGUF` `c8f5b1477e1b22cd2d819157d450f001f7047298` (base `mistralai/Mistral-Medium-3.5-128B` `22b2b868a15677cfa6061277ed2f653d1349a9ab`) | 128B | custom licence (LICENSE file begins 'Modified MIT License Attribution notice: 2026 - Mistral AI P') (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 69.8 | 1 x 80 GiB (69.8/1*1.05 + 3 = 76.2) | manifest written | `llamacpp-mistral-medium-3p5-128b`, `llamacpp-bench-mistral-medium-3p5-128b` |
| Mistral-Small-4-119B-2603 | `bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF` `f569d04efea9546de10ec040819908fa4e8f1c92` (base `mistralai/Mistral-Small-4-119B-2603` `a11f36bebf709121056b1dbcc943d1c6afbe494d`) | 119B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 67.6 | 1 x 80 GiB (67.6/1*1.05 + 3 = 74.0) | manifest written | `llamacpp-mistral-small-4-119b-2603`, `llamacpp-bench-mistral-small-4-119b-2603` |
| Devstral-2-123B-Instruct-2512 | `unsloth/Devstral-2-123B-Instruct-2512-GGUF` `1f2bfbe35f7f9071d9b318374bf5eeffefab4459` (base `mistralai/Devstral-2-123B-Instruct-2512` `1613bf01adb5e1c6fdc196b46e6b173eae75eb4a`) | 125B | custom licence (LICENSE file begins 'Modified MIT License Attribution notice: 2025 - Mistral AI P') (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 69.8 | 1 x 80 GiB (69.8/1*1.05 + 3 = 76.2) | manifest written | `llamacpp-devstral-2-123b-2512`, `llamacpp-bench-devstral-2-123b-2512` |
| Devstral-Small-2-24B-Instruct-2512 | `unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF` `6e458b8add42681bfd023de5eab93637694aaf82` (base `mistralai/Devstral-Small-2-24B-Instruct-2512` `55c5b41e98c2dbd21b0c8afffc540dcfc9eb5128`) | 24B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 13.3 | 1 x 22 GiB (13.3/1*1.05 + 3 = 17.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-devstral-small-2-24b-2512`, `llamacpp-bench-devstral-small-2-24b-2512` |
| Ministral-3-14B-Instruct-2512 | `mistralai/Ministral-3-14B-Instruct-2512-GGUF` `74fac473c43357d7fb2671713608183cc72496d0` (base `mistralai/Ministral-3-14B-Instruct-2512` `29439f81c2be264d8d393273f99e7db9c0961120`) | 13.9B | Apache-2.0 | no | llama.cpp (GGUF) | Q8_0 | 13.4 | 1 x 22 GiB (13.4/1*1.05 + 3 = 17.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-ministral-3-14b-instruct-2512`, `llamacpp-bench-ministral-3-14b-instruct-2512` |
| Ministral-3-14B-Reasoning-2512 | `mistralai/Ministral-3-14B-Reasoning-2512-GGUF` `fe3b038f30334729263d860d5dadbaa34e0f2a18` (base `mistralai/Ministral-3-14B-Reasoning-2512` `51f9210f3cd20f3452a80d5819d15dc61cc50630`) | 13.9B | Apache-2.0 | no | llama.cpp (GGUF) | Q8_0 | 13.4 | 1 x 22 GiB (13.4/1*1.05 + 3 = 17.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-ministral-3-14b-reasoning-2512`, `llamacpp-bench-ministral-3-14b-reasoning-2512` |
| Magistral-Small-2509 | `mistralai/Magistral-Small-2509-GGUF` `429b90d8a8f0037241db6fab46a20b0f90859b03` (base `mistralai/Magistral-Small-2509` `a31cc96ab10cf19bc42c628fedf1e359e0853c49`) | 24B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 13.3 | 1 x 22 GiB (13.3/1*1.05 + 3 = 17.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-magistral-small-2509`, `llamacpp-bench-magistral-small-2509` |
| Mixtral-8x22B-Instruct-v0.1 | `MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF` `9f2f6c5ec37f9bce5f5f3a7ff07b11d573443e62` (base `mistralai/Mixtral-8x22B-Instruct-v0.1` `cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d`) | 141B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 79.7 | 2 x 48 GiB (79.7/2*1.05 + 3 = 44.9) | manifest written | `llamacpp-mixtral-8x22b-instruct-v01`, `llamacpp-bench-mixtral-8x22b-instruct-v01` |
| Codestral-22B-v0.1 | `bartowski/Codestral-22B-v0.1-GGUF` `0e6abe14d6aeaf2c99d5dc9973205e8e38692d90` (base `mistralai/Codestral-22B-v0.1` `28b1c1a51dabe9d86ca8c41420ada1984632498f`) | 22.2B | Mistral AI Non-Production License (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 12.4 | 1 x 22 GiB (12.4/1*1.05 + 3 = 16.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-codestral-22b-v01`, `llamacpp-bench-codestral-22b-v01` |
| Mistral-Large-Instruct-2411 | `bartowski/Mistral-Large-Instruct-2411-GGUF` `cc2729f0da879f162f8d2a7e74c7324bab6e51b7` (base `mistralai/Mistral-Large-Instruct-2411` `ba78820945ae22361b0274cf0ae6d696c967c1a4`) | 123B | Mistral Research License (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 68.2 | 1 x 80 GiB (68.2/1*1.05 + 3 = 74.6) | manifest written | `llamacpp-mistral-large-instruct-2411`, `llamacpp-bench-mistral-large-instruct-2411` |
| Pixtral-12B-2409 (HF-format conversion) | `mistral-community/pixtral-12b` `c2756cbbb9422eba9f6c5c439a214b0392dfc998` | 12.7B | Apache-2.0 | no | transformers | as stored | 23.6 | 1 x 40 GiB (23.6/1*1.1 + 4 = 30.0) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `pixtral-12b-2409-pytorch`, `pixtral-12b-2409-bench` |
| Pixtral-Large-Instruct-2411 | `mistralai/Pixtral-Large-Instruct-2411` `c1e51f6f11974a1199685d35c62f5a425c2d001e` | ? | Mistral Research License (restricted) | no | vLLM | as stored (Mistral format) | 231.2 | 4 x 80 GiB ((231.2/4 + 6)/0.9 = 70.9) | manifest written | `vllm-pixtral-large-instruct-2411`, `vllm-bench-pixtral-large-instruct-2411` |

### OpenAI gpt-oss

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| gpt-oss-120b | `ggml-org/gpt-oss-120b-GGUF` `238abdd290bb874b90a5da1b4549881b7d05c091` (base `openai/gpt-oss-120b` `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`) | 117B | Apache-2.0 | no | llama.cpp (GGUF) | MXFP4 | 59.0 | 1 x 80 GiB (59.0/1*1.05 + 3 = 65.0) | manifest written | `llamacpp-gpt-oss-120b`, `llamacpp-bench-gpt-oss-120b` |
| gpt-oss-120b (vLLM; official MXFP4 weights) | `openai/gpt-oss-120b` `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | 117B | Apache-2.0 | no | vLLM | as stored | 60.8 | 1 x 80 GiB ((60.8/1 + 6)/0.9 = 74.2) | manifest written | `vllm-gpt-oss-120b-vllm`, `vllm-bench-gpt-oss-120b-vllm` |

### GLM

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| GLM-5.3 | `unsloth/GLM-5.3-GGUF` `346b3591c7f28d1a23716f97a065ecf12ec14771` (base `zai-org/GLM-5.3` `aca966e4e02791568aa6a4ced368624b3d897f42`) | 753B | GLM-5.3 License (restricted) | no | llama.cpp (GGUF) | UD-Q4_K_XL | 435.2 | 8 x 80 GiB (435.2/8*1.05 + 3 = 60.1) | manifest written | `llamacpp-glm53`, `llamacpp-bench-glm53` |
| GLM-5.3-Flash | `unsloth/GLM-5.3-Flash-GGUF` `a38483c8cd5df544f53d70fb281afe97369d5ab6` (base `zai-org/GLM-5.3-Flash` `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`) | 321B | MIT | no | llama.cpp (GGUF) | UD-Q4_K_XL | 186.0 | 4 x 80 GiB (186.0/4*1.05 + 3 = 51.8) | manifest written | `llamacpp-glm53-flash`, `llamacpp-bench-glm53-flash` |
| GLM-5.2 | `unsloth/GLM-5.2-GGUF` `abc55e72527792c6e77069c99b4cb7de16fa9f23` (base `zai-org/GLM-5.2` `cf457fa734ab149ffef225f80893eb38c6ff5cdc`) | 753B | MIT | no | llama.cpp (GGUF) | UD-Q4_K_M | 433.8 | 8 x 80 GiB (433.8/8*1.05 + 3 = 59.9) | manifest written | `llamacpp-glm52`, `llamacpp-bench-glm52` |
| GLM-4.7 | `unsloth/GLM-4.7-GGUF` `70deda3fcbe622671ff2111d7c0c7392f618e3b0` (base `zai-org/GLM-4.7` `602d01efcdd332c5238ca4bcede555defbe83eb7`) | 358B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 201.6 | 4 x 80 GiB (201.6/4*1.05 + 3 = 55.9) | manifest written | `llamacpp-glm47`, `llamacpp-bench-glm47` |
| GLM-4.7-Flash | `unsloth/GLM-4.7-Flash-GGUF` `0d32489ecb9db6d2a4fc93bd27ef01519f95474d` (base `zai-org/GLM-4.7-Flash` `7dd20894a642a0aa287e9827cb1a1f7f91386b67`) | 31.2B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 17.1 | 1 x 22 GiB (17.1/1*1.05 + 3 = 20.9) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-glm47-flash`, `llamacpp-bench-glm47-flash` |
| GLM-4.6V | `zai-org/GLM-4.6V` `4e2d47eb0b41c5280d8294b17cef9e94fdcfff46` | 108B | MIT | no | transformers | as stored | 200.6 | 4 x 80 GiB (200.6/4*1.1 + 4 = 59.2) | manifest written | `glm46v-pytorch`, `glm46v-bench` |
| GLM-OCR | `zai-org/GLM-OCR` `2e85a62840ccac27daa451df36c736c4636b8628` | 1.33B | MIT | no | transformers | as stored | 2.5 | 1 x 10 GiB (2.5/1*1.1 + 4 = 6.7) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `glm-ocr-pytorch`, `glm-ocr-bench` |

### Kimi

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Kimi-K3 | `unsloth/Kimi-K3-GGUF` `a0836360ce58dfec088d966a97f2ddc8a606279b` (base `moonshotai/Kimi-K3` `f831ab66814297da540d832a5235f8e904f29d06`) | 2.78T | Kimi K3 License (restricted) | no | llama.cpp (GGUF) | UD-Q4_K_XL | 1405.1 | 8 x 192 GiB (1405.1/8*1.05 + 3 = 187.4) | manifest written | `llamacpp-kimi-k3`, `llamacpp-bench-kimi-k3` |
| Kimi-K2.6 | `unsloth/Kimi-K2.6-GGUF` `47c7cab1e440dd9fcfc57c469e4737983408a6f2` (base `moonshotai/Kimi-K2.6` `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`) | 1.03T | Modified MIT (restricted) | no | llama.cpp (GGUF) | UD-Q4_K_XL | 543.6 | 8 x 80 GiB (543.6/8*1.05 + 3 = 74.4) | manifest written | `llamacpp-kimi-k26`, `llamacpp-bench-kimi-k26` |
| Kimi-K2.7-Code | `unsloth/Kimi-K2.7-Code-GGUF` `46352ca8dc32aa60f9754f5bd3fb778deeb9e430` (base `moonshotai/Kimi-K2.7-Code` `74797c9c62378b951a1f6fcf5c4631024e9b8bef`) | 1.03T | Modified MIT (restricted) | no | llama.cpp (GGUF) | UD-Q4_K_XL | 543.6 | 8 x 80 GiB (543.6/8*1.05 + 3 = 74.4) | manifest written | `llamacpp-kimi-k27-code`, `llamacpp-bench-kimi-k27-code` |
| Kimi-K2-Thinking | `unsloth/Kimi-K2-Thinking-GGUF` `7e4f5d413a9bb469174b2cc9e47d457d16bc6702` (base `moonshotai/Kimi-K2-Thinking` `a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55`) | 1.03T | Modified MIT (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 578.6 | 4 x 180 GiB (578.6/4*1.05 + 3 = 154.9) | manifest written | `llamacpp-kimi-k2-thinking`, `llamacpp-bench-kimi-k2-thinking` |
| Kimi-Linear-48B-A3B-Instruct | `bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF` `228dbe476e5a02091624a19068f4c962caa8a1c5` (base `moonshotai/Kimi-Linear-48B-A3B-Instruct` `e1df551a447157d4658b573f9a695d57658590e9`) | 49.1B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 28.0 | 1 x 40 GiB (28.0/1*1.05 + 3 = 32.4) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-kimi-linear-48b-a3b-instruct`, `llamacpp-bench-kimi-linear-48b-a3b-instruct` |

### Cohere

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| command-a-plus-05-2026 | `bartowski/command-a-plus-05-2026-GGUF` `a26ba921242484b5e1a8ac8519169fe7421c3ab3` (base `CohereLabs/command-a-plus-05-2026-bf16` `5fb6fde5fd12ff89356aae552e11883bc49f069b`) | 219B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 125.8 | 2 x 80 GiB (125.8/2*1.05 + 3 = 69.1) | manifest written | `llamacpp-command-a-plus-05-2026`, `llamacpp-bench-command-a-plus-05-2026` |
| North-Mini-Code-1.0 | `bartowski/North-Mini-Code-1.0-GGUF` `6ff6563002170723a6f7a672bf4c99775be6c0dd` (base `CohereLabs/North-Mini-Code-1.0` `d11e61a842617a22dc328552fa5bb86231ee4f37`) | 30.5B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 17.5 | 1 x 22 GiB (17.5/1*1.05 + 3 = 21.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-north-mini-code-1p0`, `llamacpp-bench-north-mini-code-1p0` |
| c4ai-command-a-03-2025 | `unsloth/c4ai-command-a-03-2025-GGUF` `5ca94ecdf2f60896faaf110a93902bcb1eaefd22` (base `CohereLabs/c4ai-command-a-03-2025` `066daaa11d57404b07cc650d1068936d31431979`) | 111B | CC-BY-NC-4.0 (restricted) | official: auto; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 62.5 | 1 x 80 GiB (62.5/1*1.05 + 3 = 68.7) | manifest written (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/CohereLabs/c4ai-command-a-03-2025, then set HF_TOKEN) | `llamacpp-command-a-03-2025`, `llamacpp-bench-command-a-03-2025` |
| c4ai-command-r-plus-08-2024 | `bartowski/c4ai-command-r-plus-08-2024-GGUF` `4cb2af9f9be9b2753915e67aa8d658cd031b8a21` (base `CohereLabs/c4ai-command-r-plus-08-2024` `e808c1a2249354ca211c9f08d1338e5039f633f8`) | 104B | CC-BY-NC-4.0 (restricted) | official: auto; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 58.4 | 1 x 80 GiB (58.4/1*1.05 + 3 = 64.4) | manifest written (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/CohereLabs/c4ai-command-r-plus-08-2024, then set HF_TOKEN) | `llamacpp-command-r-plus-08-2024`, `llamacpp-bench-command-r-plus-08-2024` |
| c4ai-command-r-08-2024 | `bartowski/c4ai-command-r-08-2024-GGUF` `0390adf30f4e73160774db7d8fcd1f6172a16e65` (base `CohereLabs/c4ai-command-r-08-2024` `dc835b893cd3fb8f14b24970dbc2a0a6d3c22ee3`) | 32.3B | CC-BY-NC-4.0 (restricted) | official: auto; downloaded copy: no | llama.cpp (GGUF) | Q4_K_M | 18.4 | 1 x 24 GiB (18.4/1*1.05 + 3 = 22.4) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/CohereLabs/c4ai-command-r-08-2024, then set HF_TOKEN) | `llamacpp-command-r-08-2024`, `llamacpp-bench-command-r-08-2024` |

### Falcon

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Falcon-H1-34B-Instruct | `tiiuae/Falcon-H1-34B-Instruct-GGUF` `ceb29e9c241944eb64c44325972c4e9470271d92` (base `tiiuae/Falcon-H1-34B-Instruct` `6e6890e26cba1f58b1e6ee70d654ddf054f9fce8`) | 33.6B | Falcon LLM License (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 18.9 | 1 x 24 GiB (18.9/1*1.05 + 3 = 22.9) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-falcon-h1-34b-instruct`, `llamacpp-bench-falcon-h1-34b-instruct` |
| Falcon-H1R-7B | `tiiuae/Falcon-H1R-7B-GGUF` `2dc053e015a9e3c5b954aa81e00aaed24bef830f` (base `tiiuae/Falcon-H1R-7B` `a6f74bf181389908efd6970878d7ee2b42f5d417`) | 7.59B | Falcon LLM License (restricted) | no | llama.cpp (GGUF) | Q8_0 | 7.5 | 1 x 15 GiB (7.5/1*1.05 + 3 = 10.9) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-falcon-h1r-7b`, `llamacpp-bench-falcon-h1r-7b` |
| Falcon3-10B-Instruct | `tiiuae/Falcon3-10B-Instruct-GGUF` `0072d525c72df8cbb1af9561672da481bd9e5595` (base `tiiuae/Falcon3-10B-Instruct` `8799bc6aec0152757221dc6b272d824642db6202`) | 10.3B | Falcon LLM License (restricted) | no | llama.cpp (GGUF) | Q8_0 | 10.2 | 1 x 15 GiB (10.2/1*1.05 + 3 = 13.7) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-falcon3-10b-instruct`, `llamacpp-bench-falcon3-10b-instruct` |
| Falcon-OCR | `tiiuae/Falcon-OCR` `fe757d59ecd79d4d68760162306a70a015761ad9` | 270M | Apache-2.0 | no | transformers | as stored | 0.5 | 1 x 10 GiB (0.5/1*1.1 + 4 = 4.6) | blocked: custom model code (auto_map ['AutoConfig', 'AutoModelForCausalLM']); model_type 'falcon_ocr' is not in transformers 5.19.0 and trust_remote_code is not used | - |

### Microsoft Phi

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Phi-4-reasoning-plus | `unsloth/Phi-4-reasoning-plus-GGUF` `80fff8542dc7b88dba725b660beefd80e91e80c9` (base `microsoft/Phi-4-reasoning-plus` `69baf8528e1bcf05f475034d9e5dd32875ed125f`) | 14.7B | MIT | no | llama.cpp (GGUF) | Q4_K_M | 8.4 | 1 x 15 GiB (8.4/1*1.05 + 3 = 11.9) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-phi4-reasoning-plus`, `llamacpp-bench-phi4-reasoning-plus` |
| Phi-4-mini-instruct | `unsloth/Phi-4-mini-instruct-GGUF` `78eb92a46fc37e6b524df991ed9aca9bc6aa7b80` (base `microsoft/Phi-4-mini-instruct` `cfbefacb99257ffa30c83adab238a50856ac3083`) | 3.84B | MIT | no | llama.cpp (GGUF) | Q8_0 | 3.8 | 1 x 10 GiB (3.8/1*1.05 + 3 = 7.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-phi4-mini-instruct`, `llamacpp-bench-phi4-mini-instruct` |

### NVIDIA Nemotron

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| NVIDIA-Nemotron-3-Ultra-550B-A55B | `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF` `2fb7d5b3f4eae7aedb18b4839b6a6300111e46f6` (base `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` `77df655d5e9f8362164ed14dd8b48f8bce657498`) | 561B | OpenMDW-1.1 (restricted) | no | llama.cpp (GGUF) | UD-Q4_K_M | 334.6 | 8 x 48 GiB (334.6/8*1.05 + 3 = 46.9) | manifest written | `llamacpp-nemotron-3-ultra-550b-a55b`, `llamacpp-bench-nemotron-3-ultra-550b-a55b` |
| NVIDIA-Nemotron-3-Super-120B-A12B | `lmstudio-community/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF` `16410500b9bdc62081c34cc2dff52471e001f970` (base `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` `2dc98e2afe4face0e4ce40972a915c45368bd34a`) | 124B | NVIDIA Nemotron Open Model License (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 80.1 | 2 x 48 GiB (80.1/2*1.05 + 3 = 45.1) | manifest written | `llamacpp-nemotron-3-super-120b-a12b`, `llamacpp-bench-nemotron-3-super-120b-a12b` |
| NVIDIA-Nemotron-3-Nano-30B-A3B | `ggml-org/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF` `f9d9b441a049bf27473c3cdd9cec220f5657b862` (base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`) | 31.6B | NVIDIA Nemotron Open Model License (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 20.9 | 1 x 40 GiB (20.9/1*1.05 + 3 = 24.9) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-nemotron-3-nano-30b-a3b`, `llamacpp-bench-nemotron-3-nano-30b-a3b` |
| NVIDIA-Nemotron-3.5-Lightning-30B-A3B | `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` `f0eec2267ae843d9eb21ea3926ab0046da0a8628` (base `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` `a9904d24bcc1d289a1950fa9d2b978c47cf903b9`) | 31.6B | OpenMDW-1.1 (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 23.7 | 1 x 40 GiB (23.7/1*1.05 + 3 = 27.9) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-nemotron-3p5-lightning-30b-a3b`, `llamacpp-bench-nemotron-3p5-lightning-30b-a3b` |
| NVIDIA-Nemotron-Nano-9B-v2 | `bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF` `b2daccafb3d123ea94653720a0db1f5db9c698ff` (base `nvidia/NVIDIA-Nemotron-Nano-9B-v2` `6533e8de2c68e4536bf7c411d7a3ce5734111476`) | 8.89B | NVIDIA Open Model License (restricted) | no | llama.cpp (GGUF) | Q8_0 | 8.8 | 1 x 15 GiB (8.8/1*1.05 + 3 = 12.2) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `llamacpp-nemotron-nano-9b-v2`, `llamacpp-bench-nemotron-nano-9b-v2` |

### MiniMax

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| MiniMax-M3 | `bartowski/MiniMax-M3-GGUF` `2ee48e5fcf21a2522ee2fb9ffaa15592ec5498e6` (base `MiniMaxAI/MiniMax-M3` `f0e1c1e04d40177e4673a22097036854f536e9c0`) | 427B | minimax-community (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 243.3 | 4 x 80 GiB (243.3/4*1.05 + 3 = 66.9) | manifest written | `llamacpp-minimax-m3`, `llamacpp-bench-minimax-m3` |
| MiniMax-M2.7 | `bartowski/MiniMaxAI_MiniMax-M2.7-GGUF` `2b13ec99437c3ddb36f692a3a90be57ed3ba43af` (base `MiniMaxAI/MiniMax-M2.7` `d494266a4affc0d2995ba1fa35c8481cbd84294b`) | 229B | custom licence (LICENSE file begins 'NON-COMMERCIAL LICENSE Non-commercial use permitted based on') (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 129.3 | 2 x 80 GiB (129.3/2*1.05 + 3 = 70.9) | manifest written | `llamacpp-minimax-m2p7`, `llamacpp-bench-minimax-m2p7` |
| MiniMax-M2.5 | `unsloth/MiniMax-M2.5-GGUF` `7c50dca0e5592483ad308ecffc876aecac725660` (base `MiniMaxAI/MiniMax-M2.5` `f710177d938eff80b684d42c5aa84b382612f21f`) | 229B | Modified MIT (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 128.8 | 2 x 80 GiB (128.8/2*1.05 + 3 = 70.6) | manifest written | `llamacpp-minimax-m2p5`, `llamacpp-bench-minimax-m2p5` |

### Other LLMs

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Tencent Hy3 | `bartowski/Hy3-GGUF` `3ee5a3c4d76226edd9cdaec017969d24b589b64e` (base `tencent/Hy3` `a960ebc3da325ba167f069f76c41eb62c9280d22`) | 299B | Apache-2.0 | no | llama.cpp (GGUF) | Q4_K_M | 169.6 | 4 x 80 GiB (169.6/4*1.05 + 3 = 47.5) | manifest written | `llamacpp-hy3`, `llamacpp-bench-hy3` |
| Tencent Hy4-preview | `tencent/Hy4-preview` `705d81ee51566a186d645b74c974d642ef2828fe` | 780B | Apache-2.0 | no | vLLM | as stored | 1452.8 | 8 x 256 GiB ((1452.8/8 + 6)/0.9 = 208.5) | manifest written | `vllm-hy4-preview`, `vllm-bench-hy4-preview` |
| Step-3.7-Flash | `stepfun-ai/Step-3.7-Flash-GGUF` `0b69336d2fd2adfdef9c66e425f7778196c31482` (base `stepfun-ai/Step-3.7-Flash` `5f6244077ac62e04eec3f320501ff8c2b293373a`) | 201B | Apache-2.0 | no | llama.cpp (GGUF) | IQ4_XS | 97.8 | 2 x 80 GiB (97.8/2*1.05 + 3 = 54.3) | manifest written | `llamacpp-step-3p7-flash`, `llamacpp-bench-step-3p7-flash` |
| Trinity-Large-Thinking | `arcee-ai/Trinity-Large-Thinking-GGUF` `352ed5f50f0148466e596950c29aecb51bed4cb3` (base `arcee-ai/Trinity-Large-Thinking` `dc6a99b6b74202880e31491a445a94884c97890c`) | 399B | OpenMDW-1.1 (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 225.2 | 4 x 80 GiB (225.2/4*1.05 + 3 = 62.1) | manifest written | `llamacpp-trinity-large-thinking`, `llamacpp-bench-trinity-large-thinking` |
| EXAONE-4.5-33B | `LGAI-EXAONE/EXAONE-4.5-33B-GGUF` `0e969634ef24db05151b435970297a6dee634b7e` (base `LGAI-EXAONE/EXAONE-4.5-33B` `570aa4b15a4f45ba1133072b45f50198f6e3b4fd`) | 34.4B | EXAONE AI Model License Agreement (restricted) | no | llama.cpp (GGUF) | Q4_K_M | 18.7 | 1 x 24 GiB (18.7/1*1.05 + 3 = 22.6) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `llamacpp-exaone-4p5-33b`, `llamacpp-bench-exaone-4p5-33b` |

### Image generation

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| FLUX.1-schnell (unsloth mirror of the gated official repo) | `unsloth/FLUX.1-schnell` `9df3faa7ae3b6ddf0b2b69bb78616372897cc65c` (base `black-forest-labs/FLUX.1-schnell` `741f7c3ce8b383c54771c7003378a50191e9efe9`) | 11.9B | Apache-2.0 | auto | diffusers | as stored | 31.4 | 1 x 40 GiB (max(22.1 largest component, 31.4/1) + 6 = 37.4) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `flux1-schnell-diffusers`, `flux1-schnell-bench` |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` | 11.9B | FLUX.1 [dev] Non-Commercial License (restricted) | auto | diffusers | as stored | 31.4 | 1 x 40 GiB (max(22.2 largest component, 31.4/1) + 6 = 37.4) | blocked: gated. Access: accept the terms at https://huggingface.co/black-forest-labs/FLUX.1-dev (gated: auto), then set HF_TOKEN | - |
| FLUX.1-Krea-dev | `black-forest-labs/FLUX.1-Krea-dev` `8162a9c7b05a641be098422bf2fcf335615c2f28` | 11.9B | FLUX.1 [dev] Non-Commercial License (restricted) | auto | diffusers | as stored | 31.4 | 1 x 40 GiB (max(22.2 largest component, 31.4/1) + 6 = 37.4) | blocked: gated. Access: accept the terms at https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev (gated: auto), then set HF_TOKEN | - |
| FLUX.2-dev | `black-forest-labs/FLUX.2-dev` `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c` | 32.2B | FLUX Non-Commercial License (restricted) | auto | diffusers | as stored | 105.1 | 2 x 80 GiB (max(60.0 largest component, 105.1/2) + 6 = 66.0) | blocked: gated. Access: accept the terms at https://huggingface.co/black-forest-labs/FLUX.2-dev (gated: auto), then set HF_TOKEN | - |
| FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-4B` `e7b7dc27f91deacad38e78976d1f2b499d76a294` | 3.88B | Apache-2.0 | no | diffusers | as stored | 14.9 | 1 x 22 GiB (max(7.5 largest component, 14.9/1) + 6 = 20.9) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `flux2-klein-4b-diffusers`, `flux2-klein-4b-bench` |
| FLUX.2-klein-9B | `black-forest-labs/FLUX.2-klein-9B` `92196c8e11f7b6cf2b7493e037d8c5345c559216` | 9.08B | FLUX Non-Commercial License (restricted) | auto | diffusers | as stored | 32.3 | 1 x 40 GiB (max(16.9 largest component, 32.3/1) + 6 = 38.3) | blocked: gated. Access: accept the terms at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B (gated: auto), then set HF_TOKEN | - |
| Stable Diffusion 3.5 Large | `stabilityai/stable-diffusion-3.5-large` `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f` | 8.15B | Stability AI Community License (restricted) | auto | diffusers | as stored | 25.7 | 1 x 40 GiB (max(15.2 largest component, 25.7/1) + 6 = 31.7) | blocked: gated. Access: accept the terms at https://huggingface.co/stabilityai/stable-diffusion-3.5-large (gated: auto), then set HF_TOKEN | - |
| Stable Diffusion 3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` `b940f670f0eda2d07fbb75229e779da1ad11eb80` | 2.47B | Stability AI Community License (restricted) | auto | diffusers | as stored | 15.2 | 1 x 22 GiB (max(8.9 largest component, 15.2/1) + 6 = 21.2) | blocked: gated. Access: accept the terms at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium (gated: auto), then set HF_TOKEN | - |
| HiDream-I1-Full | `HiDream-ai/HiDream-I1-Full` `8ccbbfb270ccdae26d6bb0081df67dc81e4033bf` | 17.1B | MIT | no | diffusers | as stored | 43.9 | 1 x 80 GiB (max(31.9 largest component, 43.9/1) + 6 = 49.9) | blocked: the pipeline loads a Llama-3.1-8B-Instruct text encoder and tokenizer from the gated meta-llama repository (HiDream's model card), which this repo cannot fetch without credentials | - |

### Video generation

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Wan2.2-T2V-A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` `5be7df9619b54f4e2667b2755bc6a756675b5cd7` | 14.3B | Apache-2.0 | no | diffusers | stored fp32, loaded bf16 | 58.8 | 1 x 80 GiB (max(26.6 largest component, 58.8/1) + 16 = 74.8) | manifest written | `wan22-t2v-a14b-diffusers`, `wan22-t2v-a14b-bench` |
| Wan2.2-TI2V-5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` `b8fff7315c768468a5333511427288870b2e9635` | 5B | Apache-2.0 | no | diffusers | stored fp32, loaded bf16 | 15.9 | 1 x 40 GiB (max(9.3 largest component, 15.9/1) + 16 = 31.9) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `wan22-ti2v-5b-diffusers`, `wan22-ti2v-5b-bench` |
| Wan2.1-T2V-14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` | 14.3B | Apache-2.0 | no | diffusers | stored fp32, loaded bf16 | 37.4 | 1 x 80 GiB (max(26.6 largest component, 37.4/1) + 16 = 53.4) | manifest written | `wan21-t2v-14b-diffusers`, `wan21-t2v-14b-bench` |
| Wan2.1-T2V-1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` `0fad780a534b6463e45facd96134c9f345acfa5b` | 1.42B | Apache-2.0 | no | diffusers | stored fp32, loaded bf16 | 13.5 | 1 x 40 GiB (max(10.6 largest component, 13.5/1) + 16 = 29.5) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `wan21-t2v-1p3b-diffusers`, `wan21-t2v-1p3b-bench` |
| HunyuanVideo (diffusers conversion) | `hunyuanvideo-community/HunyuanVideo` `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` | 12.8B | Tencent Hunyuan Community License (restricted) | no | diffusers | as stored | 39.0 | 1 x 80 GiB (max(23.9 largest component, 39.0/1) + 16 = 55.0) | manifest written | `hunyuanvideo-diffusers`, `hunyuanvideo-bench` |
| HunyuanVideo-1.5 720p T2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` `f4dbc4a1efa4ac8ea56680cdf79d9f455105e814` | 8.33B | Tencent Hunyuan Community License (restricted) | no | diffusers | stored fp32, loaded bf16 | 24.9 | 1 x 44 GiB (max(15.5 largest component, 24.9/1) + 16 = 40.9) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `hunyuanvideo-15-720p-t2v-diffusers`, `hunyuanvideo-15-720p-t2v-bench` |
| CogVideoX-2b | `zai-org/CogVideoX-2b` `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01` | 1.69B | Apache-2.0 | no | diffusers | as stored | 12.8 | 1 x 40 GiB (max(8.9 largest component, 12.8/1) + 16 = 28.8) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `cogvideox-2b-diffusers`, `cogvideox-2b-bench` |
| CogVideoX-5b | `zai-org/CogVideoX-5b` `8fc5b281006c82b82d34fd2543d2f0ebb4e7e321` | 5.57B | custom licence (LICENSE file begins 'The CogVideoX License 1. Definitions “Licensor” means the Co') (restricted) | no | diffusers | as stored | 20.0 | 1 x 40 GiB (max(10.4 largest component, 20.0/1) + 16 = 36.0) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `cogvideox-5b-diffusers`, `cogvideox-5b-bench` |
| CogVideoX1.5-5B | `zai-org/CogVideoX1.5-5B` `fdc5267c90b5c06492985b966e43aae984e189e0` | 5.57B | custom licence (LICENSE file begins 'The CogVideoX License 1. Definitions “Licensor” means the Co') (restricted) | no | diffusers | as stored | 28.9 | 1 x 48 GiB (max(17.7 largest component, 28.9/1) + 16 = 44.9) | manifest written | `cogvideox15-5b-diffusers`, `cogvideox15-5b-bench` |
| Mochi 1 preview | `genmo/mochi-1-preview` `cd1d0fe89aa8382fcd9ae760bc3d7afe6183579f` | 10B | Apache-2.0 | no | diffusers | bf16 variant | 37.3 | 1 x 80 GiB (max(18.7 largest component, 37.3/1) + 16 = 53.3) | manifest written | `mochi-1-preview-diffusers`, `mochi-1-preview-bench` |
| LTX-Video 0.9.x | `Lightricks/LTX-Video` `8984fa25007f376c1a299016d0957a37a2f797bb` | 1.92B | custom licence (LICENSE file begins 'LTX Video ("LTXV") By Lightricks Ltd. ("Lightricks") LTXV Op') (restricted) | no | diffusers | stored fp32, loaded bf16 | 13.2 | 1 x 40 GiB (max(8.9 largest component, 13.2/1) + 16 = 29.2) | verified on a real GPU: NVIDIA L40S 48 GB (model catalog, stage 2a, Tier B; sm_89): reference recorded and identical in 3 further runs | `ltx-video-diffusers`, `ltx-video-bench` |
| MiniMax-H3 | `MiniMaxAI/MiniMax-H3` `42ed227ee7df40d41602854ae760620d6eb651fe` | 33.1B | minimax-h3-community-license-agreement (restricted) | no | diffusers | as stored | 464.1 | 2 x 256 GiB (max(134.1 largest component, 464.1/2) + 16 = 248.1) | blocked: own inference code (library minimax-h3 on its card; per-task FL2VA / Ref2VA directories, no model_index.json), not a diffusers pipeline | - |
| LTX-2.3 | `Lightricks/LTX-2.3` `3c6a4e66e5d0a684231950b9c74dd4ded7b6fadc` | ? | ltx-2-community-license-agreement (restricted) | no | diffusers | as stored | - | 1 x 10 GiB (max(0.0 largest component, 0.0/1) + 6 = 6.0) | blocked: the repository has no diffusers layout (no model_index.json; single-file checkpoints for ComfyUI), so diffusers' from_pretrained cannot load it | - |
| LTX-2.5 | `Lightricks/LTX-2.5` `2356ce76915d6c48d313d7e8b25900e1dd3abaa8` | ? | ltx-2.x-community-license-agreement (restricted) | auto | diffusers | as stored | - | 1 x 10 GiB (max(0.0 largest component, 0.0/1) + 6 = 6.0) | blocked: gated (auto) and single-file layout. Access: accept the terms at https://huggingface.co/Lightricks/LTX-2.5 (gated: auto), then set HF_TOKEN | - |

### Speech

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Orpheus 3B 0.1 ft (text to speech tokens) | `lex-au/Orpheus-3b-FT-Q8_0.gguf` `6d58c3d2601e03f2016380cc6069d643d1119501` (base `canopylabs/orpheus-3b-0.1-ft` `4206a56e5a68cf6cf96900a8a78acd3370c02eb6`) | 3.78B | Apache-2.0 (restricted) | official: auto; downloaded copy: no | llama.cpp (GGUF) | Q8_0 | 3.8 | 1 x 10 GiB (3.8/1*1.05 + 3 = 6.9) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/canopylabs/orpheus-3b-0.1-ft, then set HF_TOKEN) | `llamacpp-orpheus-3b-0p1-ft`, `llamacpp-bench-orpheus-3b-0p1-ft` |
| Whisper large-v3-turbo (whisper.cpp ggml) | `ggerganov/whisper.cpp` `5359861c739e955e79d9a303bcbc70fb988958b1` (base `openai/whisper-large-v3-turbo` `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`) | 809M | MIT | no | whisper.cpp | ggml (fp16) | 1.5 | 1 x 10 GiB (1.5/1*1.05 + 2 = 3.6) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `whisper-cpp-large-v3-turbo`, `whisper-cpp-bench-large-v3-turbo` |
| parakeet-tdt-0.6b-v3 | `nvidia/parakeet-tdt-0.6b-v3` `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | 627M | CC-BY-4.0 | no | transformers / NeMo | as stored | 1.2 | 1 x 10 GiB (1.2/1*1.1 + 4 = 5.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `parakeet-tdt-0p6b-v3-pytorch`, `parakeet-tdt-0p6b-v3-bench` |
| canary-1b-v2 | `nvidia/canary-1b-v2` `d455706339a6b32e1aa40f82c713a482a0c938e2` | 979M | CC-BY-4.0 | no | transformers / NeMo | as stored (.nemo) | 1.8 | 1 x 10 GiB (1.8/1*1.1 + 4 = 6.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `canary-1b-v2-pytorch`, `canary-1b-v2-bench` |
| Chatterbox (Resemble AI) | `ResembleAI/chatterbox` `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | ? | MIT | no | package | as stored | 3.0 | 1 x 10 GiB (3.0/1*1.1 + 4 = 7.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `chatterbox-pytorch`, `chatterbox-bench` |
| F5-TTS (v1 Base) | `SWivid/F5-TTS` `84e5a410d9cead4de2f847e7c9369a6440bdfaca` | ? | CC-BY-NC-4.0 (restricted) | no | package | as stored | 1.3 | 1 x 10 GiB (1.3/1*1.1 + 4 = 5.4) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `f5-tts-pytorch`, `f5-tts-bench` |

### Retrieval

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| BGE-M3 (dense head) | `BAAI/bge-m3` `5617a9f61b028005a4858fdac845db406aefb181` | ? | MIT | no | transformers | as stored (pytorch .bin) | 2.1 | 1 x 10 GiB (2.1/1*1.1 + 4 = 6.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `bge-m3-pytorch`, `bge-m3-bench` |
| bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 568M | Apache-2.0 | no | transformers | as stored | 1.1 | 1 x 10 GiB (1.1/1*1.1 + 4 = 5.2) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `bge-reranker-v2-m3-pytorch`, `bge-reranker-v2-m3-bench` |
| bge-reranker-large | `BAAI/bge-reranker-large` `55611d7bca2a7133960a6d3b71e083071bbfc312` | 560M | MIT | no | transformers | as stored | 2.1 | 1 x 10 GiB (2.1/1*1.1 + 4 = 6.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `bge-reranker-large-pytorch`, `bge-reranker-large-bench` |

### Vision

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| SigLIP 2 so400m patch16 512 | `google/siglip2-so400m-patch16-512` `ceea1cba8130d8271436da4828633198c176a775` | 1.14B | Apache-2.0 | no | transformers | as stored | 2.1 | 1 x 10 GiB (2.1/1*1.1 + 4 = 6.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `siglip2-so400m-patch16-512-pytorch`, `siglip2-so400m-patch16-512-bench` |
| SigLIP 2 giant-opt patch16 384 | `google/siglip2-giant-opt-patch16-384` `a713301b217d38485fb2204c808367d10bc3cc40` | 1.87B | Apache-2.0 | no | transformers | as stored | 3.5 | 1 x 10 GiB (3.5/1*1.1 + 4 = 7.8) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `siglip2-giant-opt-patch16-384-pytorch`, `siglip2-giant-opt-patch16-384-bench` |
| DINOv3 ViT-7B/16 | `facebook/dinov3-vit7b16-pretrain-lvd1689m` `b80367753773648a6793235ab9c65cdbb029506f` | 6.72B | DINOv3 License (restricted) | manual | transformers | as stored | 12.5 | 1 x 22 GiB (12.5/1*1.1 + 4 = 17.8) | blocked: gated. Access: accept the terms at https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m (gated: manual), then set HF_TOKEN | - |
| DINOv2 giant | `facebook/dinov2-giant` `611a9d42f2335e0f921f1e313ad3c1b7178d206d` | 1.14B | Apache-2.0 | no | transformers | as stored | 2.1 | 1 x 10 GiB (2.1/1*1.1 + 4 = 6.3) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `dinov2-giant-pytorch`, `dinov2-giant-bench` |
| SAM 2.1 Hiera Large | `facebook/sam2.1-hiera-large` `665f8e2ad61cf5f53d65644ff27c8ee525124610` | 224M | Apache-2.0 | no | transformers | as stored | 0.4 | 1 x 10 GiB (0.4/1*1.1 + 4 = 4.5) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `sam2p1-hiera-large-pytorch`, `sam2p1-hiera-large-bench` |
| SAM 3 | `facebook/sam3` `3c879f39826c281e95690f02c7821c4de09afae7` | 860M | custom licence (card: license other, no name given) (restricted) | manual | transformers | as stored | 1.6 | 1 x 10 GiB (1.6/1*1.1 + 4 = 5.8) | blocked: gated. Access: accept the terms at https://huggingface.co/facebook/sam3 (gated: manual), then set HF_TOKEN | - |
| Depth Anything V2 Large | `depth-anything/Depth-Anything-V2-Large-hf` `7581137eff8d4e94f6e796d3baea0e9fa79b22d2` | 335M | CC-BY-NC-4.0 (restricted) | no | transformers | as stored | 0.6 | 1 x 10 GiB (0.6/1*1.1 + 4 = 4.7) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `depth-anything-v2-large-pytorch`, `depth-anything-v2-large-bench` |
| Depth Anything V2 Small | `depth-anything/Depth-Anything-V2-Small-hf` `5426e4f0f36572d16453bbda7a8389317b1bef99` | 24.8M | Apache-2.0 | no | transformers | as stored | 0.0 | 1 x 10 GiB (0.0/1*1.1 + 4 = 4.1) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `depth-anything-v2-small-pytorch`, `depth-anything-v2-small-bench` |

### OCR and documents

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| olmOCR-2-7B-1025 | `allenai/olmOCR-2-7B-1025` `e52d6f090b7a9007afffbbd6ce510876222fea93` | 8.29B | Apache-2.0 | no | transformers | as stored | 15.4 | 1 x 22 GiB (15.4/1*1.1 + 4 = 21.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `olmocr-2-7b-pytorch`, `olmocr-2-7b-bench` |
| PaddleOCR-VL | `PaddlePaddle/PaddleOCR-VL` `7fa00a8c55b735ba51ba49a9058f3f9c57a99a11` | 959M | Apache-2.0 | no | transformers | as stored | 1.8 | 1 x 10 GiB (1.8/1*1.1 + 4 = 6.0) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `paddleocr-vl-pytorch`, `paddleocr-vl-bench` |
| dots.ocr | `rednote-hilab/dots.ocr` `c0111ce6bc07803dbc267932ffef0ae3a51dc951` | 3.04B | MIT | no | transformers | as stored | 5.7 | 1 x 15 GiB (5.7/1*1.1 + 4 = 10.2) | blocked: custom model code (auto_map, library dots_ocr): not in transformers 5.19.0 | - |

### Time series

| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |
| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |
| Chronos-Bolt Base | `amazon/chronos-bolt-base` `5d9f166d69f47aef3401367a7b842e78fe97b121` | 205M | Apache-2.0 | no | chronos | as stored | 0.4 | 1 x 10 GiB (0.4/1*1.1 + 4 = 4.4) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `chronos-bolt-base-pytorch`, `chronos-bolt-base-bench` |
| Chronos-2 | `amazon/chronos-2` `29ec3766d36d6f73f0696f85560a422f50e8498c` | 119M | Apache-2.0 | no | chronos | as stored | 0.2 | 1 x 10 GiB (0.2/1*1.1 + 4 = 4.2) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `chronos-2-pytorch`, `chronos-2-bench` |
| Chronos T5 Large | `amazon/chronos-t5-large` `0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a` | 709M | Apache-2.0 | no | chronos | as stored | 1.3 | 1 x 10 GiB (1.3/1*1.1 + 4 = 5.5) | verified on a real GPU: NVIDIA A10G (model catalog, stage 2a, 2026-10-10): first real run | `chronos-t5-large-pytorch`, `chronos-t5-large-bench` |

<!-- registry:end -->
