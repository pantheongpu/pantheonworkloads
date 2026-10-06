# Synthetic GGUF workloads for llama.cpp (`llamacpp-synth-*`)

**These are random-weight models for kernel and graph coverage. They are NOT the quality or behaviour of
any named pretrained model.** Every weight is a seeded pseudo-random number placed in the tensor names and
shapes of an architecture. Nothing here says how Llama, Mistral, Mixtral, Qwen2, Gemma or Phi-3 models
perform, and the "Llama" / "Mistral" / ... names only say which llama.cpp architecture graph (and which
tensor layout) a file selects. Every manifest says so (`notes`, `model.id`), as does the GGUF metadata of
every generated file (`general.description`, `pantheonworkloads.synth.disclaimer`).

Why: the workloads exercise llama.cpp's graph for each architecture and its quantised matmul kernels
(dequantisation, `mmvq`, `mmq`, `MUL_MAT_ID`, flash attention, RoPE, norms) on the CUDA and HIP backends with
**no model licence and no download** (Hugging Face is not needed). A CPU run is the reference that a simulated
GPU must reproduce: token ids exactly, logits and perplexity within a tolerance.

Built on `docs/llamacpp.md` (same pinned llama.cpp `b11447`, same backends, `sim:` wiring and variables).

## Parts

| File | What it is |
| --- | --- |
| `tools/llamacpp/synth_gguf.py` | the generator: `python3 synth_gguf.py <llama\|mistral\|mixtral\|qwen2\|gemma\|phi3> -o model.gguf [--seed N] [--size tiny\|s\|m\|l\|xl]`; also `--text-out`, `--imatrix-out`, `--list` (parameter counts) |
| `tools/llamacpp/probe/pw-probe.cpp` | 100-line C++ program on llama.cpp's public C API: greedy decode that prints the exact token ids and per-step logit statistics as JSON (`llama-completion` prints text, not ids). `build.sh` compiles it with plain `c++` beside the pinned build when `LLAMACPP_BUILD_PROBE=1` |
| `tools/llamacpp/synth_run.py` | what a functional workload runs: build and cache the models, quantise with `llama-quantize` (same pinned build), run `pw-probe` and `llama-perplexity` per quantisation type, print the contract's JSON |
| `tools/llamacpp/synth.sh` | shell helpers the `run.sh` files source (builds the extra tools into the usual cache prefix when missing, finds or makes a python with numpy and `gguf`) |
| `tools/llamacpp/synth_workloads.py` | the table of workloads; `generate` writes every manifest, `run.sh` and `models.sha256`; `check` and the tests fail when a file differs; `tune` measures prompts, spreads and hashes |
| `tools/llamacpp/synth_tune.py`, `synth_spreads.json` | the measurements behind prompts and tolerances (below) |

Needs `numpy` and the `gguf` package (`pip install gguf==0.19.0`; `synth.sh` makes a venv in
`~/.cache/pantheonworkloads/venvs/synth` when python lacks them) and a build with `llama-quantize`,
`llama-perplexity` and `pw-probe` (built automatically into the cache prefix on first use; with
`PW_LLAMACPP_BIN_DIR` you must provide them: `LLAMACPP_EXTRA_TARGETS="llama-quantize llama-perplexity"
LLAMACPP_BUILD_PROBE=1 tools/llamacpp/build.sh cpu <prefix>`).

## The models

Seeded, deterministic, byte-identical for the same `(architecture, seed, size, vocab, generator version)`:

- **Weights**: a counter-based generator (splitmix64 of seed, tensor name and position; each value is the sum
  of twelve 16-bit uniform integers, centred and divided by 65536, then scaled and cast to float16). Only
  integer arithmetic, one exact division and an IEEE cast are involved, so the bytes do not depend on numpy's
  or the platform's random or libm implementation. `tests/test_synth.py` pins the sha256 of every
  architecture's file; `models.sha256` next to each workload pins every quantised file, and a run fails when a
  generated file differs (`PW_SYNTH_SKIP_SHA=1` turns that off).
- **Quantisation**: `llama-quantize` from the same pinned build, from the F16 file; the output is identical for
  any thread count (checked). The IQ1/IQ2/IQ3_XXS/IQ3_XS/Q2_K_S types refuse to quantise without an importance
  matrix, so the generator also writes a **seeded synthetic importance matrix** (noise, not measured data; it is
  passed with a relative path because `llama-quantize` stores the path in the file).
- **Tokenizers**: generated, not copied. A 1024-token SentencePiece vocabulary (4 special tokens, 256 byte tokens,
  printable ASCII, word pieces with every prefix so merges work, filler tokens) for llama, mistral, mixtral, gemma
  and phi3; a byte-level BPE vocabulary (merges trained on the same word list, qwen2 pre-tokenizer) for qwen2. The
  vocabulary lists are original: llama.cpp's `models/ggml-vocab-*.gguf` were *not* used, because their per-file
  provenance and licence are not stated (only the repository's MIT licence covers them). Larger sizes pad the
  vocabulary to 32000 with unused tokens so the output layer has a realistic size.
- **Sizes** (`--size`): `tiny` (n_embd 256, 2 layers, head dim 32; 1.2 to 4 M parameters, 2 to 8 MB as F16;
  functional workloads) and, for benchmarks, `s` (34 to 93 M), `m` (160 to 510 M, default), `l` (0.8 to 2.5 B), `xl`
  (5.5 to 19 B). `synth_gguf.py --list` prints the exact counts. At `tiny` every weight row length is a
  multiple of 256, so k-quants really are k-quants (no fallback type), but these sizes do **not** test odd shapes.
- **Shaping**: output-layer rows have seeded per-row loudness from a fixed table (no `exp`/`log`), router weights
  are large for Mixtral, Gemma's attention/FFN outputs are scaled up; all only so greedy decoding has
  clear winners and does not hinge on near-ties. They are the only "tuning" done, and none uses real data.

### Architectures

| Workload family | `general.architecture` | Graph features exercised |
| --- | --- | --- |
| llama | `llama` | RMSNorm, MHA (8 heads), RoPE 1e4, SwiGLU, untied output |
| mistral | `llama` | same graph with grouped-query attention (8 heads / 2 KV), RoPE 1e6. llama.cpp has no separate Mistral architecture |
| mixtral | `llama` | 8 experts, 2 used (`ffn_gate_inp` routing, merged `ffn_*_exps` tensors, `MUL_MAT_ID`), GQA, RoPE 1e6 |
| qwen2 | `qwen2` | GQA, Q/K/V biases, RoPE 1e6, byte-level BPE |
| gemma | `gemma` | multi-query attention (1 KV head), embeddings scaled by sqrt(n_embd), GeGLU, output tied to the embeddings |
| phi3 | `phi3` | fused `attn_qkv`, fused gate+up in `ffn_up`, SwiGLU |

Not covered: Gemma 2/3 (soft-capping, sliding window), sliding-window attention, other MoE routers, RoPE scaling
variants (YaRN, LongRoPE), multi-GPU splits, embedding or vision models.

## Workloads

Functional (targets `cpu`, `gpu`, `sim:nvidia/*`, `sim:amd/*`, reference recorded on the **cpu** target):

| Workload | Architecture | Quantisation types | Kernel path it exercises (CUDA/HIP backends) | State |
| --- | --- | --- | --- | --- |
| `llamacpp-synth-llama` | llama | F16, Q8_0, Q4_0, Q4_K_M | dense GEMM/GEMV; Q8_0/Q4_0 `mmvq` (decode) and `mmq` (prompt); Q4_K_M mixes Q4_K and Q6_K | cpu PASS, gpu/sim not run |
| `llamacpp-synth-mistral` | mistral (GQA) | F16, Q8_0, Q4_0, Q4_K_M | as above with grouped-query attention | cpu PASS, gpu/sim not run |
| `llamacpp-synth-mixtral` | mixtral (MoE) | F16, Q8_0, Q4_0, Q4_K_M | as above plus expert routing and `MUL_MAT_ID` | cpu PASS, gpu/sim not run |
| `llamacpp-synth-qwen2` | qwen2 | F16, Q8_0, Q4_0, Q4_K_M | as above plus attention biases, BPE | cpu PASS, gpu/sim not run |
| `llamacpp-synth-gemma` | gemma | F16, Q8_0, Q4_0, Q4_K_M | as above, MQA, tied output matrix quantised too | cpu PASS, gpu/sim not run |
| `llamacpp-synth-phi3` | phi3 | F16, Q8_0, Q4_0, Q4_K_M | as above, fused QKV and gate/up | cpu PASS, gpu/sim not run |
| `llamacpp-synth-quant-legacy` | llama | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | 32-weight block types: `mmvq`, `mmq`, dequantise-then-GEMM | cpu PASS, gpu/sim not run |
| `llamacpp-synth-quant-kquant` | llama | Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K | 256-weight super-block types and llama-quantize's per-tensor mixes | cpu PASS, gpu/sim not run |
| `llamacpp-synth-quant-iquant` | llama | IQ3_S, IQ3_M, IQ4_NL, IQ4_XS | lookup-table dequantisation (IQ4 codebooks, IQ3 grids) | cpu PASS, gpu/sim not run |
| `llamacpp-synth-quant-iquant-imatrix` | llama | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, Q2_K_S | the grid-lookup types that need an importance matrix (synthetic one) | cpu PASS, gpu/sim not run |
| `llamacpp-synth-quant-float` | llama | F32, F16, BF16 | unquantised cuBLAS/rocBLAS GEMM, float mat-vec, BF16 paths | cpu PASS, gpu/sim not run |
| `llamacpp-synth-quant-lowbit` | llama | Q1_0, Q2_0, TQ1_0, TQ2_0 | 1/2-bit and ternary types (TQ1_0 and TQ2_0 give identical numbers: both are exact ternary encodings of the same quantisation) | cpu PASS, gpu/sim not run |
| `llamacpp-synth-quant-moe` | mixtral | Q4_0, Q8_0, Q4_K_M, Q6_K, IQ4_XS, MXFP4_MOE | quantised expert matmuls per block family; MXFP4 experts (6 expert tensors, the rest Q8_0) | cpu PASS, gpu/sim not run |

Coverage in one line: every architecture runs F16, Q8_0, Q4_0 and Q4_K_M; the llama architecture additionally runs every
other type the pinned `llama-quantize` writes (legacy, k-quants, IQ, F32/BF16, 1-2 bit, ternary), and mixtral additionally
runs Q6_K, IQ4_XS and MXFP4 experts.

The quantisation-group workloads use the llama architecture (the dense matmul kernels do not care about the
architecture); the architecture workloads fix four representative types. Combining every architecture with every
type would be 6 x 30 runs for little extra kernel coverage. The one type the pinned `llama-quantize` lists that is not
covered is `NVFP4` (it refused these models in a trial).

Benchmarks (`llama-bench`, real targets `cpu` and `gpu` only, **not run on any GPU**, nothing recorded under `bench/`):
`llamacpp-synth-bench-{llama,mistral,mixtral,qwen2,gemma,phi3}`. `PW_SYNTH_SIZE` (`tiny`, `s`, `m` default, `l`, `xl`) and
`PW_SYNTH_QUANT` (default `Q4_K_M`) choose the model; `PW_BENCH_PP`, `PW_BENCH_TG`, `PW_BENCH_REPS`, `PW_BENCH_EXTRA`, `PW_NGL`,
`PW_THREADS` as in `docs/llamacpp.md`. A record carries the manifest's model pointer (size `m`, `Q4_K_M`), so record only with the defaults. The
pipeline was exercised on the cpu target at size `s` (the numbers are not kept: they are CPU numbers).

## What a functional run outputs and how it is compared

For each quantisation type `Q` the run decodes 6 greedy tokens after a fixed three-word prompt and runs
`llama-perplexity` (8 chunks of 128 tokens) on a fixed generated text. The `output` object holds
`Q.tokens` (the 6 ids as a string: compared **exactly**), `Q.top1_logit` and `Q.logsumexp` (6 values each, absolute
tolerance) and `Q.ppl` (relative tolerance). `detail` names the build and the models; `metrics` is always empty.
The perplexities are large (1e3 to 1e5) because a random model is not a language model: they are a checksum of the
whole output distribution, not a quality number.

Prompts differ per workload: `synth_tune.py` tries 120 seeded three-word prompts and keeps the one whose smallest
top-1/top-2 logit margin, over all generated tokens and all quantisation types of that workload, is the largest
(`min_margin` in `synth_spreads.json`: 0.25 to 6 logit units on the tiny models; the lowest are the 1-2 bit types).

### Tolerances (how they were derived)

Per quantisation type: `max(class floor, 5 x the largest CPU spread, rounded up to 2 significant figures)`.

- **CPU spread** (measured by `synth_tune.py`, stored in `synth_spreads.json`): the same model and prompt are run with
  threads 1/2/4, micro-batch 2 vs full, flash attention on/off, f32 vs f16 KV cache and weight repacking on/off, and
  the largest absolute difference of the top-1 logits and log-sum-exps and the largest relative difference of the
  perplexity from the baseline run is kept. Observed: float types about 3e-3 to 3e-2 absolute (largest with flash attention off), quantised types 0.04 to 0.3 absolute (weight repacking changes which integer dot-product code runs; the tool keeps only the maximum over all settings), perplexity
  at most 1.2e-2 relative; **no setting changed a token id**.
- **Class floors** (judgement, not measurement): float 0.05 / 0.5 %; Q8_0, Q5, Q6_K 0.15 / 1 %; 4-bit types 0.3 / 2 %;
  2-3 bit and ternary types 0.5 / 5 %. A GPU backend quantises activations differently from the CPU one (q8_1 blocks) and
  accumulates in another order, so the CPU spreads are only a lower bound; the floors and the factor 5 are there for that.
  **The first real-GPU run will say whether they are right.** Loosen a tolerance only after looking at that run's
  numbers, and say so in the commit.
- Token ids are compared exactly everywhere, as requested. For the lowest-bit types the margin (as little as 0.25) is
  not many times the tolerance (up to 1.6 for Q3_K_M), so a GPU could in principle flip an id while its logits are still
  inside the tolerance. Treat such a failure as a near-tie to inspect, not automatically as a kernel bug.

## State, plainly

- Ran here: all 13 functional workloads on the **cpu** target with the CPU build of b11447 (portable `GGML_NATIVE=OFF`),
  references recorded with `bin/pw record <workload> --target cpu` (the recorder is the CPU backend), then re-run with
  `PW_THREADS` 1 and 4 and `PASS`ed. The benchmark pipeline (mixtral and gemma at size `s`, pp64/tg16) ran on cpu, and the `tiny` size under stub tools in the tests.
- **Not run**: any real GPU, and any `sim:nvidia/*` or `sim:amd/*` target (this machine has neither a CUDA/ROCm toolchain
  nor pantheonsim; the `sim:` wiring is the one from `docs/llamacpp.md`, itself never run). `pw-probe` also has to build
  against the CUDA/HIP builds; it uses only the public C API and `ggml_backend_load_all`, but that is untested.
- The MXFP4/ternary/1-bit types may have no dedicated CUDA/HIP kernel in b11447 (the backend then converts or falls back
  to the CPU): a pass there tells less than for the legacy and k-quants. Which kernels a particular GPU run really took is
  not recorded by these workloads.
- Odd shapes (row lengths not a multiple of 32 or 256, where llama-quantize falls back to another type), long contexts,
  batches of several sequences and multi-GPU are not covered.

## Running

```bash
bin/pw run llamacpp-synth-llama --target cpu                       # one family
bin/pw run $(ls workloads | grep '^llamacpp-synth-' | grep -v bench) --target cpu
bin/pw record llamacpp-synth-quant-kquant --target cpu             # only on a target you trust
bin/pw run llamacpp-synth-quant-moe --target sim:nvidia/h100       # needs PANTHEONSIM_DIR and a CUDA build
PW_SYNTH_SIZE=l PW_SYNTH_QUANT=Q8_0 bin/pw run llamacpp-synth-bench-mixtral --target gpu --bench --device "NVIDIA ..."
python3 tools/llamacpp/synth_workloads.py check                    # files match the table
python3 tools/llamacpp/synth_workloads.py tune --bin <prefix>/bin --cache <dir>   # re-measure after changing anything
```

After changing the generator, the vocabulary, the quantisation list or the llama.cpp pin: bump
`GENERATOR_VERSION` in `synth_gguf.py` if the generated bytes change, run `tune`, then `generate`, update the
sha256s in `tests/test_synth.py`, and re-record every reference on the cpu target.

## Licences (all read, not assumed)

- `gguf` Python package 0.19.0 (the writer): **MIT**, "Copyright (c) 2023 Georgi Gerganov" (the `LICENSE` file inside the
  PyPI wheel `gguf-0.19.0-py3-none-any.whl`, whose `METADATA` also carries the classifier `License :: OSI Approved :: MIT License`;
  the same text is `gguf-py/LICENSE` in llama.cpp at the pinned commit). It depends on numpy (BSD-3-Clause), PyYAML, requests, tqdm,
  none of which is redistributed here.
- llama.cpp: **MIT** (`LICENSE` at the pinned commit; `pw-probe` is compiled against its headers, our own source is Apache-2.0
  like the rest of this repository).
- Vocabularies and text: generated by `synth_gguf.py` from a word list written for it; nothing is copied from a tokenizer or a
  corpus. llama.cpp's `models/ggml-vocab-*.gguf` were deliberately not used (no per-file licence stated).
- The generated models: no third-party weights; `general.license` is `apache-2.0` (this repository's licence).
