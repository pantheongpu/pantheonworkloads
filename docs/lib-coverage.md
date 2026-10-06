# Library-kernel coverage (`lib-*` workloads)

Model-free workloads for the GPU library paths AI frameworks sit on: FFT, dense solvers, sparse, embeddings and
sort/scan, RNN, convolution variants, normalisation and activations, attention backends, int8/fp8/TF32/autocast, and three
small composite blocks. Seeded inputs, no downloads, no licences to read. They are written to run unchanged on `cpu`,
`gpu`, `sim:nvidia/*` and `sim:amd/*` and to tell a limited backend apart op by op.

| Workload | Ops | Kind | Targets |
| --- | ---: | --- | --- |
| `lib-fft-linalg` | 38 | functional | cpu, gpu, sim:nvidia/\*, sim:amd/\* |
| `lib-sparse-embedding` | 54 | functional | same |
| `lib-rnn-conv` | 78 | functional | same |
| `lib-attention-precision` | 70 | functional | same |
| `lib-composite-blocks` | 11 | functional | same |
| `lib-kernels-bench` | 53 metrics | benchmark | gpu only |

## What was run, and what was not

**Ran:** the five functional workloads, on the `cpu` target only (4 shared cores, CPU PyTorch 2.13.0 from conda-forge, numpy 2.5.3),
several times (2, 1 and 4 threads: all PASS against one reference), and their references were recorded there with
`bin/pw record <workload> --target cpu` (`recorded_on: cpu` in each `reference.json`). The benchmark code path ran once on
the CPU at `PW_LIB_SIZE=tiny` (53 metrics produced) to prove it works; those numbers mean nothing and were not recorded.

**Not run:** any GPU (NVIDIA or AMD), any simulated GPU (`sim:*`), any benchmark at real size. Nothing in this table that says
"not run" has been seen to work or fail there.

Of the 251 ops, 236 ran and passed on the CPU and 15 reported `unsupported` there (14 SDPA ops pinned to the memory-efficient
(11, one of them backward) or cuDNN (3) backend, and the half-precision FFT). Those keys are `"unsupported"` in the CPU reference (see "References").

| Op family | GPU library it maps to | Workload | cpu | gpu / sim |
| --- | --- | --- | --- | --- |
| FFT c2c / r2c / c2r, 1D/2D/3D, power-of-two, odd, prime sizes, ortho norm, FFT convolution, hfft | cuFFT / rocFFT | `lib-fft-linalg` | ran | not run |
| FFT in half precision (power-of-two only on GPUs) | cuFFT / rocFFT | `lib-fft-linalg` | ran: `unsupported` | not run |
| Cholesky, QR, SVD, eigh, eigvals, LU + lu_solve, solve, inv, slogdet, triangular solve, lstsq, pinv, matrix_exp, matrix_power, norms | cuSOLVER / rocSOLVER, cuBLAS / rocBLAS | `lib-fft-linalg` | ran | not run |
| CSR / COO x dense, CSR x CSR, addmm, transposed operand, COO coalesce, sparse sum, SDDMM (`sampled_addmm`) | cuSPARSE / hipSPARSE | `lib-sparse-embedding` | ran | not run |
| Sparse softmax / log_softmax | PyTorch sparse kernels over cuSPARSE | `lib-sparse-embedding` | ran | not run |
| embedding, embedding_bag (sum/mean/max, offsets, per-sample weights, fixed bags), their backward | PyTorch embedding kernels | `lib-sparse-embedding` | ran | not run |
| index_add, index_select, index_put (accumulate), scatter_add, scatter_reduce (5 modes), gather, masked_select, nonzero | PyTorch indexing kernels (atomics) | `lib-sparse-embedding` | ran | not run |
| sort (stable, ties), topk, kthvalue, cumsum/cumprod/cummax, unique (+counts, inverse), bincount, histc, searchsorted | CUB / hipCUB / rocPRIM | `lib-sparse-embedding` | ran | not run |
| LSTM / GRU / RNN: bidirectional, multi-layer, projection, packed sequences, cells, backward, fp16, bf16 | cuDNN / MIOpen RNN | `lib-rnn-conv` | ran | not run |
| conv2d dense / strided / depthwise / grouped / dilated, conv1d, conv3d, transposed 1D/2D/3D | cuDNN / MIOpen convolution | `lib-rnn-conv` | ran | not run |
| channels_last conv, fp16 / bf16 conv, conv backward (input and weight) | cuDNN / MIOpen | `lib-rnn-conv` | ran | not run |
| pooling, interpolate (bilinear, nearest, bicubic), unfold/fold, pixel_shuffle, padding modes | PyTorch native kernels | `lib-rnn-conv` | ran | not run |
| batch_norm (train with running stats, eval, backward), group / instance / layer / rms norm, LRN, fp16 / bf16 norms | cuDNN / MIOpen batch norm, native norm kernels | `lib-rnn-conv` | ran | not run |
| 20 pointwise activations, GLU, softmax / log_softmax / logsumexp, cross entropy, fp16 / bf16 activations | native kernels | `lib-rnn-conv` | ran | not run |
| SDPA math backend: plain, causal, GQA, bool mask, float bias, cross lengths, head dim 40, decode with KV cache, backward | math (cuBLAS) | `lib-attention-precision` | ran | not run |
| SDPA flash backend: fp16, bf16, fp32, causal, GQA, cross lengths, decode, backward | flash-attention | `lib-attention-precision` | ran (CPU flash kernel) | not run |
| SDPA memory-efficient backend, same variants | mem-efficient (CUTLASS) | `lib-attention-precision` | ran: `unsupported` | not run |
| SDPA cuDNN backend | cuDNN attention | `lib-attention-precision` | ran: `unsupported` | not run |
| int8 matmul (`_int_mm`, exact int32), int8 quantise/dequantise, per-tensor and per-channel fake quant, weight-only int8 linear | cuBLASLt int8, native | `lib-attention-precision` | ran | not run |
| dynamic int8 Linear (torch.ao) | fbgemm / onednn (CPU only) | `lib-attention-precision` | ran | expected `unsupported` |
| fp8 e4m3 `_scaled_mm`, fp8 casts | cuBLASLt / hipBLASLt fp8 | `lib-attention-precision` | ran | not run |
| TF32 matmul / conv / bmm, flags restored afterwards | tensor-core TF32 | `lib-attention-precision` | ran (no TF32 on the CPU: plain fp32) | not run |
| fp16 / bf16 long-K accumulation, reductions, softmax with large logits, fp16 overflow / underflow, cast rounding (fp16, bf16, fp8) | tensor-core half paths | `lib-attention-precision` | ran | not run |
| autocast fp16 / bf16: linear, MLP + layer_norm, softmax / loss, conv + bmm + sdpa, backward with fp32 parameters | cuBLAS / cuDNN under autocast | `lib-attention-precision` | ran | not run |
| ViT encoder (patch embed, class token, attention, MLP; fp32 / fp16 / bf16, backward) | the kernel mix of vision transformers | `lib-composite-blocks` | ran | not run |
| UNet step: ResBlock + GroupNorm + timestep embedding + attention + up/down, one DDPM reverse step (fp32 / fp16 / channels_last bf16, gradients) | diffusion-model kernel mix | `lib-composite-blocks` | ran | not run |
| DLRM-style recsys: bottom MLP, 8 embedding bags, dot interaction, top MLP (fp32 / fp16, gradients) | recommender kernel mix | `lib-composite-blocks` | ran | not run |
| Throughput of all of the above families (FFT GFLOP/s, solver GFLOP/s, SpMM GFLOP/s, embedding / scatter GB/s, sort Mkeys/s, LSTM / conv / SDPA / matmul TFLOPS, int8 / fp8 TOPS, ViT / UNet / DLRM samples/s) | all | `lib-kernels-bench` | tiny smoke test only | not run |

Not covered: 2:4 semi-structured sparsity, BSR formats, block-sparse attention, FlashAttention-3 / FlexAttention, `torch.compile`,
NCCL/RCCL collectives, multi-GPU, CUDA graphs, complex-valued solvers, quantised models (GPTQ/AWQ kernels), Triton kernels.

## How an op is checked

Every op is a function of a context (`workloads/_lib/libkernels.py`): the same function runs twice from the same seeded inputs, on the
target device in the op's dtype and on the CPU in float64 (inputs first rounded to the op's dtype, so both see the same numbers).

1. **Run-time check.** The run fails if the device result differs from the float64 result by more than the op's bound, as max
   |error| divided by the reference's rms (integer results must be equal). This catches a backend that is wrong the same way
   every time, which a recorded reference cannot. A failure names the op and the error.
2. **Recorded checksum.** The op contributes to the `output` object: floats as `[mean|y|, rms, y at 3 positions]`, integers as
   `n=.. sum=.. chk=.. head=..` (position-weighted checksum, exact). `reference.json` holds these; `bin/pw` compares them with the
   manifest's `tolerance` (default abs 1e-4, rel 1e-3, plus per-op `fields` for fp16 / bf16 / TF32 / autocast ops).
3. **Unsupported is a value.** An op for which the backend raises "not implemented / not supported / no available kernel" gets
   the string `"unsupported"` in `output` (and a line on stderr). The run still exits 0, so the output shows exactly which ops a
   limited backend lacks; compared with a reference holding numbers, that key fails and only that key. Wrong numbers or any other
   exception fail the run (they are not "unsupported"). A `VirtualGPU error [` line anywhere fails the run, as in the other PyTorch workloads.

Ops with a sign or gauge freedom are reduced to invariants first (|R| and Q R for QR, singular values and U S Vh for SVD,
eigenvalues and V diag(w) V^T for eigh). Tie-prone integer ops use stable sorts. Ops built from random data draw it from CPU
generators with fixed seeds, in a fixed order, never inside an op function.

## Tolerances: how they were derived

`python workloads/_lib/libkernels.py <group> --calibrate` (needs only torch) runs every float op in its dtype, in float64, with 1
thread and once more, and prints per op the abs tolerance each comparison would need at the chosen rel, the max error against
float64, and how much of the default tolerance was used. Findings on the CPU (torch 2.13.0):

- fp32 ops: max error against float64 is 1e-8 to 2e-5 of rms (the worst is `inv`, 2e-5); every checksum entry is inside the default
  tolerance and uses under 1% of it. 1 thread against 4 threads and repeated runs change no entry beyond that. The per-op bounds
  (`bound=` in the group files) are about 10x the measured error: 2e-5 for FFTs (measured 3e-7 to 1e-6), 1e-4 for most solvers
  and fused ops, 1e-3 for gradients, `lstsq`, `pinv` and general `eigvals`.
- fp16 / bf16 ops (including autocast and TF32, which are classed by the precision they compute in): rel is 4 eps of the dtype
  (5e-3 for fp16 and TF32, 5e-2 for bf16, since an element can land one or two ulp from another backend's), abs is 4x the largest
  deviation measured between the dtype run and the float64 run (and 1 thread, and a repeat), and at least one eps of the
  checksum's rms. Bounds on max-error/rms are 2x to 5x the measured (fp16 2e-2, bf16 1e-1; the CPU measured up to 8e-2 for an autocast
  bf16 softmax, whose CPU policy keeps it in bf16, hence that op's 2x bound).
- These come from one CPU. A GPU's kernels reorder sums and may use different internal precision, so **the tolerances are unverified
  on a GPU**; if a first GPU run fails a half-precision op by a little, widen its field and record the evidence here.
- The manifests' `tolerance.fields` are generated from that output (numbers written as `2.0e-03`: PyYAML reads `2e-03` as a string).

## References

The five references were recorded on the `cpu` target. Consequences:

- The 15 ops the CPU cannot run (memory-efficient and cuDNN SDPA, half-precision FFT) are recorded as `"unsupported"`. On a GPU
  where they work, those keys will differ from the reference. Record a GPU reference (`bin/pw record <workload> --target gpu`) for
  `lib-attention-precision` and `lib-fft-linalg` when a GPU is available, and say so in the commit.
- `dynamic_quantized_linear` is CPU-only and `fp8_scaled_mm_e4m3` needs fp8 hardware on a GPU, so limited backends will show
  `unsupported` there, as designed.
- `flash` fp32 is supported by the CPU kernel but not by GPU flash kernels (fp16 / bf16 only): expect `unsupported` for
  `sdpa_flash_fp32_causal` on a GPU (not seen: no GPU was available).

## Running

```bash
PW_PYTHON=$(tools/pw-conda-torch.sh) bin/pw run lib-fft-linalg lib-sparse-embedding lib-rnn-conv \
    lib-attention-precision lib-composite-blocks --target cpu       # 3 s each
bin/pw run lib-rnn-conv --target gpu                                # a Python with a GPU PyTorch
bin/pw run lib-kernels-bench --target gpu --bench --repeat 3 --device "NVIDIA H100 80GB HBM3"
PW_LIB_SIZE=tiny ...                                                # smoke-test the benchmark code path anywhere
```

- Target handling (cpu / gpu / `sim:nvidia/*` / `sim:amd/*`, pantheonsim launch, `PW_PYTHON`, `VGPU_TORCH_*`) is `workloads/_shared/torch_env.sh`,
  as for the `gpt-train-*` workloads; `workloads/_lib/env.sh` adds the discovery of the conda environment below. The run exits 77 when
  there is no PyTorch, no GPU, or no pantheonsim build.
- `tools/pw-conda-torch.sh` makes a CPU-only PyTorch 2.13.0 from conda-forge (micromamba 2.0.5, sha256-checked) for hosts that can reach
  `conda.anaconda.org` but not `download.pytorch.org`; the CPU target of every `lib-*` workload finds its environment
  automatically. Not run end to end here (the existing environment was reused); its dry run, the micromamba download and checksum, and its
  argument handling were run or are unit tested (`tests/test_lib_coverage.py`).
- Benchmark metrics (`lib-kernels-bench`): `*_tflops`, `*_gflops`, `*_gb_s`, `*_mkeys_s`, `*_ops_s`, `*_samples_s`. The flop and byte
  counts are in each group file's `bench()`. An op that cannot be measured is left out and named on stderr, never faked, so a
  record's keys say what ran. Real sizes target a 16 GB or larger GPU; none has been tried.
