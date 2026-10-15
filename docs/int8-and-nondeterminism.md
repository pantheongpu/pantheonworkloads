# int8 kernels and decode nondeterminism: measurements behind three comparisons

The first A10G run (`docs/first-gpu-run-a10g.md`) failed three ONNX workloads against references recorded on the CPU of another host:
`onnx-zoo-bertsquad-int8` (zoo-vector logits 2.23 off, bound 0.25), `kokoro-tts-int8-onnx` (second sentence 70800 samples; the doc first said the reference had 71400, but its `reference.json` has 70800, see the correction in section 4)
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

Second-sentence sample count (first sentence: 94200 everywhere) and statistics against the committed reference (`reference.json`: 70800 samples; an earlier version of this line said 71400, see the correction in section 4):

| Configuration | Runs | n_samples (2nd) | RMS dev | Centroid dev (Hz) | 100 ms envelope dev |
| --- | --- | --- | --- | --- | --- |
| EPYC (AVX2), default, CPU | 4 | 70800 | 0.00007 to 0.00019 | 56 to 82 | 0.0213 to 0.0224 |
| EPYC, default, CUDA EP | 6 | 70800 | 0.00005 to 0.00017 | 49 to 72 | 0.0214 to 0.0221 |
| EPYC, `x64quantprecision=1`, CPU | 4 | 71400 | 0.00015 to 0.00025 | 14 to 32 | 0.0202 to 0.0206 |
| EPYC, `x64quantprecision=1`, CUDA EP | 6 | 71400 | 0.00008 to 0.00028 | 18 to 37 | 0.0200 to 0.0204 |
| Xeon Ice Lake (AVX512-VNNI), CPU, both modes | 6 | 70800 | 0.00006 to 0.00013 | 2 to 27 | 0.0008 to 0.0016 |

The vocoder draws random numbers, so the waveform varies run to run; the statistics above are the stable ones. The sample count is stable
over all runs of a configuration. It is **not** explained by AVX2 saturation alone: the VNNI Xeon, whose BERT output is perfect, also gives 70800,
while the 7-bit AVX2 mode gives 71400 (the committed reference, like the Xeon, has 70800: section 4). Different int8 kernels (AVX2 u8s8, 7-bit u8s8, AVX512-VNNI) round the duration predictor differently, by 600 samples (25 ms, 0.84 %) in the second sentence. Two other observations: the envelope of the second
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

## 4. server2 (Ryzen 9 9950X, Zen 5, AVX-512 VNNI)

A third CPU, measured 2026-10-15 on server2 (a shared on-prem box, see `docs/rigs.md`, "server2 as a CPU test host"): AMD Ryzen 9 9950X, Zen 5, 32 threads, `avx2 avx512f
avx512_vnni avx_vnni avx512_bf16`, no AMX. onnxruntime 1.30.0 CPU wheel, numpy 2.5.3, Python 3.12.3, the `ort-cpu` venv of `tools/ort-env.sh`, `PW_ORT_THREADS=1` (the
default, as in sections 1 to 3), commit `main` plus this change. Each workload was run 3 times with the default (`session.x64quantprecision=1`) and 3 times with
`PW_ORT_X64QUANT=0`. All 18 runs PASS against the committed, unchanged references, and all runs of a workload are identical in the stable keys, in both modes.

| Workload | Default (7-bit on) | `PW_ORT_X64QUANT=0` | Informational values |
| --- | --- | --- | --- |
| `onnx-zoo-bertsquad-int8` | 3 of 3 PASS | 3 of 3 PASS | answers "helen marsh", "1887", "port of sedgwick", "grain" and the spans equal the reference; span-score deviation 0.0000 (all four); `zoo_logit_err` 0.032; smallest span margin 1.66 |
| `kokoro-tts-int8-onnx` | 3 of 3 PASS | 3 of 3 PASS | `n_samples_exact` [94200, 70800] in all 6 runs; RMS deviation 0.00001 to 0.00016; centroid deviation 0.9 to 23.5 Hz; 100 ms envelope deviation 0.0007 to 0.0017 (second sentence 0.0005 to 0.0013) |
| `moonshine-tiny-en-onnx` | 3 of 3 PASS | 3 of 3 PASS | text equal to the reference, token ids equal to the reference, WER 0.021 against the archive's transcripts, smallest top-1/top-2 logit gap 0.004 |

Which class it resembles: the Xeon 8375C (Ice Lake, AVX512-VNNI), on every number that this document compares.

| Quantity | EPYC 7R32 (AVX2) | Xeon 8375C (AVX512-VNNI) | Ryzen 9950X (AVX512-VNNI) |
| --- | --- | --- | --- |
| BERT-Squad answers | wrong (default), right (7-bit) | right | right |
| BERT-Squad zoo error / smallest margin | 2.225 / 0.004 (default) | 0.032 / 1.66 | 0.032 / 1.66 (equal to the Xeon to the last digit) |
| Kokoro second sentence | 70800 (default), 71400 (7-bit) | 70800 | 70800 |
| Kokoro envelope deviation | 0.020 to 0.022 | 0.0008 to 0.0016 | 0.0005 to 0.0017 |
| Moonshine | comma after "parent" | equals the reference, gap 0.004 | equals the reference, gap 0.004 |
| 7-bit mode | repairs BERT-Squad | no-op | no-op (the 18 runs in both modes agree in every stable value) |

So the Zen 5 box reproduces the committed references about as closely as the Xeon does, in both modes, although it is a different vendor and microarchitecture.
That makes the earlier claim "the int8 results depend on the instruction set" stronger (every VNNI machine measured agrees, the only non-VNNI one differs), and
it shows the 7-bit setting does not change the results on a VNNI machine for these three workloads.

**Correction (found while comparing).** This document and `docs/first-gpu-run-a10g.md` said the Kokoro reference has 71400 samples in the second sentence.
The committed `workloads/kokoro-tts-int8-onnx/reference.json` has 70800 (one commit touches it, `9485cdd`, which already has 70800), the value the Xeon, the Zen 5 box and the
EPYC in default mode all produce. What the 7-bit mode on the EPYC gives, 71400, is 0.85 % above the reference, not equal to it. Consequences: the old exact-length compare
could not have failed on the EPYC's 70800, so the A10G run's Kokoro failure was the 100 ms envelope (0.0221 against the old bound 0.01), which section 2 also reports; the 1.5 %
bound for `n_samples` is still justified (it covers the 7-bit mode's 0.85 %), and no reference or tolerance was changed; and the earlier statement that the reference host
"is some other int8 kernel path" because of the sample count does not hold.

**What this says about the reference host, and what it does not.**

- The reference host's output is consistent with a VNNI int8 path: BERT-Squad correct, Moonshine tokens identical, Kokoro length 70800 and envelope within 0.002, as on the Xeon
  and on this Zen 5 box. It is not consistent with the AVX2-without-VNNI behaviour measured on the EPYC (wrong BERT-Squad answers; Moonshine's comma; envelope 0.022 away).
- It does not identify the CPU. The development machine of the first commit may have been any VNNI-capable CPU, including the WSL machine (an i9-12900K: `avx_vnni`, no AVX-512),
  a Xeon or a Zen 4/5. The three outputs here are insensitive to the differences between those: this box has both AVX-VNNI and AVX512-VNNI and ONNX Runtime picked one path, and the
  Xeon (AVX512-VNNI only) agrees with it, so 256-bit AVX-VNNI versus 512-bit AVX512-VNNI is not separated by this data, and the path cannot be forced from outside onnxruntime.
- The values are deterministic and compared with bounds (zero deviation for BERT-Squad, 0.002 envelope for Kokoro), so a match is a necessary condition, not proof: a CPU of a different
  family could match too. Only `lscpu` of the recording machine, or running the three workloads there, settles it, and that was not done (policy: no runs on the WSL machine).

## Results at a glance

| Workload | Old comparison on the A10G host | Now |
| --- | --- | --- |
| `onnx-zoo-bertsquad-int8` | FAIL (wrong answers, zoo error 2.226) | PASS: correct answers, span-score deviation 0.444, zoo error 0.047 |
| `kokoro-tts-int8-onnx` | FAIL (70800 samples; also envelope 0.0221 > 0.01) | PASS (71400 in 7-bit mode, 0.85 % above the reference's 70800; the envelope bound was the real failure) |
| `moonshine-tiny-en-onnx` | FAIL, flaky | PASS in 11 of 11 repeated runs |

Final `bin/pw run --target gpu` on the A10G: all three PASS (moonshine PASS in all 11 repeated runs); `--target cpu` on the EPYC and on the Xeon: all three PASS
against the unchanged references. No reference was re-recorded: the new output keys (`zoo_logit_err`, `span_margins`, `n_samples_exact`, `min_logit_gap`)
are informational, which `bin/pw` drops from both sides before comparing, so the existing references still apply.

## What is not settled

- **The reference host.** `reference.json` says `recorded_on: cpu`, the commit says "development machine, onnxruntime 1.30.0 CPU, 1 thread", and neither names a CPU. The
  development host's CPU is not recorded anywhere in the repo. Section 4 corrects an error in the first version of this paragraph: the committed Kokoro reference
  has 70800 samples, so the reference reproduces the VNNI Xeon (and a Zen 5 VNNI machine) on all three workloads, and it is the AVX2 EPYC that departs from it
  (BERT-Squad answers, the Moonshine comma, the Kokoro envelope). That narrows the reference host to a CPU with a VNNI int8 kernel path; it does not tell AVX-VNNI
  (256-bit) from AVX512-VNNI. Settling it needs `lscpu` of the recording machine; no test was run on the WSL host (policy).
- AMX (Sapphire Rapids) was not tried; it was not needed to settle the hypothesis for BERT-Squad, and Kokoro's host-to-host spread is already covered by a measured
  0.84 % bound. A fourth kernel path could in principle differ again: re-measure when a new CPU family shows a Kokoro length outside 1.5 %.
- Real regressions are still caught only inside the bounds above; none of the bounds was loosened without a measurement, and `rms`/`centroid_hz` were not touched.

## Cost

g5.xlarge on-demand about 0.3 h (about 1.01 USD/h) and c6i.xlarge about 0.2 h (about 0.17 USD/h), 100 GB and 30 GB gp3 roots: well under 1 USD in all. Both instances,
the key pair and the security group were deleted (instance ids in the PR description).
