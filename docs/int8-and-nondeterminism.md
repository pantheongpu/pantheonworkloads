# int8 kernels and decode nondeterminism: measurements behind three comparisons

The first A10G run (`docs/first-gpu-run-a10g.md`) failed three ONNX workloads against references recorded on the CPU of another host:
`onnx-zoo-bertsquad-int8` (zoo-vector logits 2.23 off, bound 0.25), `kokoro-tts-int8-onnx` (second sentence 70800 samples, reference 71400)
and `moonshine-tiny-en-onnx` (exact text, 1 of 3 runs matched). Hypothesis: the int8 results depend on the CPU's instruction set (u8s8
`MatMulInteger` saturates without VNNI), and Moonshine is GPU nondeterminism. This file records what was measured and the comparison each
workload now uses. Nothing here is a claim about the reference host, which is not identified (see the end).

## Rigs and method

All on-demand, us-east-1, 2026-10-09, onnxruntime / onnxruntime-gpu 1.30.0, numpy 2.5.3, Python 3.12 (uv), CPU provider at 1 thread
(`PW_ORT_THREADS` default), CUDA EP with `use_tf32=0`, commit 1533ddb plus the change in this PR.

| Rig | CPU | ISA relevant to int8 | GPU |
| --- | --- | --- | --- |
| g5.xlarge (`pw-ort-fix-g5`) | AMD EPYC 7R32, 4 vCPU | AVX2, no VNNI, no AVX-512 | NVIDIA A10G, driver 595.91.07 |
| c6i.xlarge (`pw-ort-fix-c6i`) | Intel Xeon Platinum 8375C (Ice Lake), 4 vCPU | AVX2 + AVX512F + AVX512-VNNI | none (CPU target only) |

Each payload was run repeatedly with the real code (`tools/text_tasks.py bertsquad`, `tools/speech_tasks.py tts|asr`) and every output kept.
Deviations below are against the existing `reference.json` of each workload. A Sapphire Rapids (AMX) rig was not launched: the
hypothesis was settled without it (below).

## 1. BERT-Squad int8: the hypothesis holds

| Configuration | Runs | Answers | Span-score deviation from reference | Zoo-vector logit error | Smallest span margin |
| --- | --- | --- | --- | --- | --- |
| EPYC (AVX2), default, CPU provider | 4 | wrong ("ardmore and houses a 90 - centime", ...) | 17.97 | 2.225 | 0.004 |
| EPYC (AVX2), default, CUDA EP (A10G) | 6 | wrong | 17.74 | 2.226 | 0.0002 |
| EPYC, `session.x64quantprecision=1`, CPU provider | 4 | correct (helen marsh, 1887, port of sedgwick, grain) | 0.513 | 0.032 | 1.41 |
| EPYC, `session.x64quantprecision=1`, CUDA EP | 6 | correct | 0.444 | 0.047 | 1.99 |
| Xeon Ice Lake (AVX512-VNNI), default, CPU | 3 | correct | 0.000 | 0.032 | 1.66 |
| Xeon Ice Lake, `x64quantprecision=1`, CPU | 3 | correct | 0.000 | 0.032 | 1.66 |

The runs are deterministic: all repeats in a row are identical. On the AVX2 host without VNNI the int8 model is not slightly off, it is
wrong: the 16-bit saturating intermediate of the u8s8 kernel destroys the answers, the best span has a margin of 0.0002 to 0.004 logit
units, and a 2.2 logit error on the zoo vector is that damage. The CUDA EP gives the same wrong result because MatMulInteger and the
QLinear ops have no CUDA kernel and run on the host CPU inside the session. ONNX Runtime's own switch for this, the session config
`session.x64quantprecision=1` (7-bit activations on AVX2), repairs it, and is a no-op on the VNNI host (identical outputs to the last digit).

What changed:

- `tools/ort_tasks.py` `session()` now sets `session.x64quantprecision=1` (`PW_ORT_X64QUANT=0` turns it off). The same kind of fix as
  `use_tf32=0`: a library default that makes fp or int results host-dependent is switched to the honest setting. It was **not** in the plan
  as written (compare task output instead of pinning the ISA); it was added because without it the answers on an AVX2 host are wrong,
  so no task-level comparison could pass them, and it is measured to be harmless on a VNNI host. Say if you want it removed: then bertsquad
  fails on AVX2 hosts by design.
- Comparison: `answers` and `spans` exact (unchanged); `span_scores` abs 0.75 (was abs 0.1 + rel 0.5 % of about 15, i.e. 0.17 to 0.09):
  0.75 is the smallest round value above the worst measured cross-host deviation 0.513 with a margin (1.5x). The broken
  result misses by 17.7 to 18.0 and still fails. `zoo_logit_err` and `span_margins` are informational (`informational:` in the manifest,
  see `docs/manifest.md`); the run's own zoo-vector gate went from 0.25 to 0.1 (worst measured 0.047; broken 2.2).
- `onnx-zoo-bertsquad-int8-bench` now runs (its gate passes) and is recorded: A10G, g5.xlarge, 5 repeats, 242.99 ms median per question
  (4.12 questions/s), spread 242.75 to 243.51 ms. The int8 nodes run on the 4-vCPU host CPU, so this is mostly a CPU number.

## 2. Kokoro int8: the hypothesis holds only in part

Second-sentence sample count (first sentence: 94200 everywhere) and statistics against the reference (71400 samples):

| Configuration | Runs | n_samples (2nd) | RMS dev | Centroid dev (Hz) | 100 ms envelope dev |
| --- | --- | --- | --- | --- | --- |
| EPYC (AVX2), default, CPU | 4 | 70800 | 0.00007 to 0.00019 | 56 to 82 | 0.0213 to 0.0224 |
| EPYC, default, CUDA EP | 6 | 70800 | 0.00005 to 0.00017 | 49 to 72 | 0.0214 to 0.0221 |
| EPYC, `x64quantprecision=1`, CPU | 4 | 71400 | 0.00015 to 0.00025 | 14 to 32 | 0.0202 to 0.0206 |
| EPYC, `x64quantprecision=1`, CUDA EP | 6 | 71400 | 0.00008 to 0.00028 | 18 to 37 | 0.0200 to 0.0204 |
| Xeon Ice Lake (AVX512-VNNI), CPU, both modes | 6 | 70800 | 0.00006 to 0.00013 | 2 to 27 | 0.0008 to 0.0016 |

The vocoder draws random numbers, so the waveform varies run to run; the statistics above are the stable ones. The sample count is stable
over all runs of a configuration. It is **not** explained by AVX2 saturation alone: the VNNI Xeon, whose BERT output is perfect, also gives 70800,
while the reference and the 7-bit AVX2 mode give 71400. Different int8 kernels (AVX2 u8s8, 7-bit u8s8, AVX512-VNNI, and whatever the reference
host used) round the duration predictor differently, by 600 samples (25 ms, 0.84 %) in the second sentence. Two other observations: the envelope of the second
sentence has a systematic 0.02 difference on the EPYC in both modes (its speech starts a fraction of a window earlier or later: windows 6 and 12 to 14), while the
Xeon matches the reference envelope to 0.0016, so the old bound of 0.01 could never have passed on the EPYC even with the right sample count.

Comparison now: `n_samples` within 1.5 % (smallest round value above the worst measured 0.84 % with a margin of 1.8x; the exact lengths are kept as
the informational `n_samples_exact`), `envelope` abs 0.03 (was 0.01; worst measured 0.0224, margin 1.3x), RMS abs 0.003 and centroid abs 150 Hz unchanged (worst
measured 0.00028 and 82 Hz, so the old bounds hold with a margin of 10x and 1.8x). A wrong duration (a regression of a few percent) or a
wrong phoneme string still fails.

## 3. Moonshine tiny: GPU nondeterminism confirmed, and the CPU adds an ISA effect

Word error rate of the output text against the reference text (case and punctuation ignored; 65 reference words, one word is 0.0154), plus
the smallest top-1/top-2 logit gap seen during the greedy decode:

| Configuration | Runs | Text equal to reference | WER vs reference | Smallest logit gap | Difference |
| --- | --- | --- | --- | --- | --- |
| A10G, default (no 7-bit mode) | 6 | 0 | 0 (4) / 0.0154 (2) | 0.008 to 0.116 | comma after "parent"; "dishonored" for "dishonoured" |
| A10G, 7-bit mode (the code now) | 11 | 2 | 0 (10) / 0.0154 (1) | 0.002 to 0.132 | same two kinds |
| EPYC CPU provider, default | 4 | 0 | 0 | 0.037 | comma after "parent", identical every run |
| EPYC CPU provider, 7-bit mode | 4 | 0 | 0 | 0.069 | comma after "parent", identical every run |
| Xeon Ice Lake CPU provider | 6 | 6 | 0 | 0.004 | none; token ids equal the reference |

The same card, driver, model and inputs give different tokens run to run (11 runs, 7-bit mode): the text equalled the reference twice. Every difference is a
near tie in the greedy argmax (gaps down to 0.002 logit units) between a comma and the next word, or between two spellings. So GPU run-to-run
nondeterminism is real and not explained by the CPU. The CPU provider is deterministic but the EPYC differs from the reference on the same tie
(the comma), while the VNNI Xeon matches the reference tokens exactly. The comma is invisible to a word error rate; the spelling variant costs 0.0154.

Comparison now: `text` by word error rate against the reference text, maximum 0.03 (the smallest round value above the worst measured 0.0154 with a margin of
1.9x; two changed words fail, as does any real regression); `token_ids` and `min_logit_gap` are informational. The run still fails if the WER against the
archive's own transcripts exceeds 0.15 (measured 0.021 to 0.031).

## Results at a glance

| Workload | Old comparison on the A10G host | Now |
| --- | --- | --- |
| `onnx-zoo-bertsquad-int8` | FAIL (wrong answers, zoo error 2.226) | PASS: correct answers, span-score deviation 0.444, zoo error 0.047 |
| `kokoro-tts-int8-onnx` | FAIL (70800 samples; also envelope 0.0221 > 0.01) | PASS (71400 in 7-bit mode; 70800 would pass at 0.84 %) |
| `moonshine-tiny-en-onnx` | FAIL, flaky | PASS in 11 of 11 repeated runs |

Final `bin/pw run --target gpu` on the A10G: all three PASS (moonshine PASS in all 11 repeated runs); `--target cpu` on the EPYC and on the Xeon: all three PASS
against the unchanged references. No reference was re-recorded: the new output keys (`zoo_logit_err`, `span_margins`, `n_samples_exact`, `min_logit_gap`)
are informational, which `bin/pw` drops from both sides before comparing, so the existing references still apply.

## What is not settled

- **The reference host.** `reference.json` says `recorded_on: cpu`, the commit says "development machine, onnxruntime 1.30.0 CPU, 1 thread", and neither names a CPU. The
  development host's CPU is not recorded anywhere in the repo. It reproduces the VNNI Xeon on BERT-Squad (deviation 0.000) and Moonshine (identical tokens),
  but not on the Kokoro sample count (71400 against the Xeon's 70800), so it is some other int8 kernel path (for example AVX-VNNI on a client core). Settling it needs
  `lscpu` of that machine; no test was run on this WSL host (policy).
- AMX (Sapphire Rapids) was not tried; it was not needed to settle the hypothesis for BERT-Squad, and Kokoro's host-to-host spread is already covered by a measured
  0.84 % bound. A fourth kernel path could in principle differ again: re-measure when a new CPU family shows a Kokoro length outside 1.5 %.
- Real regressions are still caught only inside the bounds above; none of the bounds was loosened without a measurement, and `rms`/`centroid_hz` were not touched.

## Cost

g5.xlarge on-demand about 0.3 h (about 1.01 USD/h) and c6i.xlarge about 0.2 h (about 0.17 USD/h), 100 GB and 30 GB gp3 roots: well under 1 USD in all. Both instances,
the key pair and the security group were deleted (instance ids in the PR description).
