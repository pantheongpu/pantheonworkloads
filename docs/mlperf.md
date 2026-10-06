# MLPerf Inference: feasibility study

Status: a study plus one plumbing-only workload (`loadgen-plumbing-check`). **No MLPerf benchmark is
ported and nothing in this repository is an MLPerf result.**

Sources read for this (all shallow clones taken 2026-10-06; hashes so the claims can be re-checked):

| What | Where | Revision |
| --- | --- | --- |
| MLPerf Inference reference code | github.com/mlcommons/inference (Apache-2.0, `LICENSE.md`) | `3fbc329939999c13d0a7b5e67fb2092287e06047` |
| Results rules | github.com/mlcommons/policies, `MLPerf_Results_Messaging_Guidelines.adoc` | `45b8b625425295cdde53c4b969149ea6c4c4d084` |
| Inference rules | github.com/mlcommons/inference_policies (Apache-2.0) | `d3eba2f` |
| LoadGen on PyPI | `mlcommons-loadgen` 6.0.17 (same as `loadgen/VERSION.txt`) | cp313 wheel sha256 `79090cf79054bad8142d00b59aa81f40dc6e19cc5ef83ccafca9fce6d8ddaabb` |

**What could not be checked.** The machine this was written on can reach GitHub and PyPI only
(Hugging Face, Zenodo, mlcommons.org, openslr.org, image-net.org, archive.org were refused by the
egress policy). So the licences and terms of the *datasets and model files themselves* below are not
read from their own pages. Where a statement comes from the inference repository's text it is marked
"repo"; where the terms live on a page that was unreachable it says **unverified**. Do not add a
benchmark on the strength of this table: read the terms at the source first.

## 1. The rules that constrain this repository

From `MLPerf_Results_Messaging_Guidelines.adoc` (policies repo):

- Results not reviewed by MLCommons are **unverified** and must say so: the word "unverified" next to
  each score and the footnote "Result not verified by MLCommons Association." Submitting for review is
  restricted to MLCommons Members and Test Partners.
- Disclosures must name the submitting organisation, benchmark name and version, and system under test,
  with the trademark footnote ("The MLPerf name and logo are registered and unregistered trademarks ...").
- "Claimed results must be compliant with that MLPerf benchmark's rules". A run with a toy SUT, other
  settings, or without the compliance runs is not an MLPerf result and must not be called one.
- MLPerf results may only be compared with compatible MLPerf results, never with non-MLPerf numbers.
- The name must not be used "in any company name, product name, service name ... model number ... or domain
  name", and must not be altered. For that reason this repository does not name workloads `mlperf-*`.
- Timing: non-submitters may not publish results for a benchmark version until two weeks after its
  official publication date.
- Open division ("Relaxed constraints for the Open division" and "Open Division" in `inference_rules.adoc`) allows
  other models and pre/post-processing, but the task and validation dataset must match an existing Closed
  benchmark and be substitutable in LoadGen; without MLCommons review the result is still unverified.

Consequence for this repository: LoadGen may be used as a load generator, but anything we record is a
**pantheonworkloads number measured with LoadGen**, labelled as such, never "MLPerf Inference vX Offline".
If an actual MLPerf-labelled number is ever wanted it has to come from an MLCommons-reviewed submission by a
Member, and be cited with the footnotes above. Whether pantheonworkloads' use of MLPerf *names* in docs like
this one is acceptable beyond factual references is a question for a person to settle with MLCommons
(support@mlcommons.org is the contact the submission docs give); nothing here is legal advice.

## 2. Which reference benchmarks are small and open enough

The v6.1 table in the repo README lists 17 models. Access terms, from the benchmark READMEs:

| Benchmark | Size | Model access | Data access | Fit for here |
| --- | --- | --- | --- | --- |
| **whisper** (`speech2text`, whisper-large-v3 on LibriSpeech dev sets) | 1.5B-parameter model | MLCommons R2 downloader, or `git clone huggingface.co/openai/whisper-large-v3` (repo). Licence: openai/whisper repo README says "code and model weights are released under the MIT License" (verified at `86098128`); the large-v3 model card was **unverified** | `inference_librispeech.csv` points at openslr.org dev-clean / dev-other tarballs with MD5s (repo). LibriSpeech terms **unverified** | Best fit to this repo's speech theme and the licence story is cleanest, but large-v3 is not CPU-friendly. A whisper.cpp SUT would make it an *Open*-style experiment, not the reference. |
| **bert** (BERT-Large, SQuAD v1.1) | 340M | Zenodo records (repo links). Licences **unverified** | SQuAD v1.1 dev set via `mlcr` scripts. Terms **unverified** | Smallest classic one; CPU-feasible with ONNX/PyTorch. Needs licence reading first. |
| **resnet50-v1.5** | 25M | Zenodo records (repo) | ImageNet 2012 validation: the dataset's own access terms **unverified**, and expected to need registration | Not addable without agreeing to ImageNet's terms. |
| **yolo v11** | small | PyTorch/ONNX | "COCO safe subset" - the repo says the script keeps "just the images that comply with license agreements" | Plausible; per-image COCO licences vary. |
| **dlrm-v3** | large | MLCommons storage | **Synthetic dataset**, generated by `streaming_synthetic_data.py` (repo) | No third-party data terms, but GPU-sized. |
| **3d-unet** | 19M | Zenodo | KiTS19 CT scans: terms **unverified** | Skip. |
| **llama3.1-8b** | 8B | Meta's gated licence. The repo says: "One has to accept the MLCommons Llama 3.1 License Confidentiality Notice" or request access from Meta and use a Hugging Face token | CNN/DailyMail via MLCommons downloader | **Excluded:** needs an agreement we cannot show, and a confidentiality notice. |
| **llama2-70b**, **llama3.1-405b**, **pointpainting (Waymo)** | huge | repo: hosted "**exclusively by MLCommons Members**" behind a confidentiality notice | same | **Excluded.** |
| **stable-diffusion-xl** | 3.5B | Hugging Face stabilityai/stable-diffusion-xl-base-1.0 snapshot on MLCommons R2; licence **unverified** | COCO 2014 subset, downloaded by script | GPU-sized; licence to read. |
| **deepseek-r1**, **gpt-oss-120b**, **qwen3-vl-235b**, **wan-2.2**, **rgat** (IGBH, 547M nodes), **e2e-rag** (~283 GB of assets), **edge-agentic** | 27B to 671B or huge datasets | various | deepseek-r1 README lists dataset licences (MIT, CC BY 4.0, "CC"); the rest **unverified** | Not small. |

Summary: nothing in the suite is both tiny and obviously free of access agreements. Excluded outright for
terms: llama2-70b, llama3.1-405b, llama3.1-8b, pointpainting/Waymo. Needs terms read before a decision:
ImageNet, KiTS19, SQuAD, LibriSpeech, COCO, the SDXL and Whisper model cards. The code itself is Apache-2.0.

## 3. LoadGen

`mlcommons-loadgen` (6.0.17, built from `loadgen/` in the inference repo, Apache-2.0) is a C++ library with
Python bindings, installable as a wheel (`pip install mlcommons-loadgen`; the wheel metadata has an empty
licence field, so the licence is the repository's). It knows nothing about models or data: the benchmark
supplies a **QSL** (query sample library: sample count, load/unload callbacks) and a **SUT** (a callback that
receives `QuerySample`s and answers with `QuerySamplesComplete`). Scenarios (from `test_settings.h`):

- **SingleStream**: one sample at a time; result is a latency percentile.
- **MultiStream**: N samples per query, next query after the previous completes.
- **Server**: single-sample queries with Poisson arrivals at `server_target_qps`; PASS if the latency
  percentile is under `server_target_latency_ns`.
- **Offline**: all samples in one query; result is samples/s.

Modes: `AccuracyOnly` (every sample at least once, responses written to `mlperf_log_accuracy.json` as hex
for a model-specific scorer), `PerformanceOnly`, `FindPeakPerformance` (Server), `SubmissionRun`.
Outputs: `mlperf_log_summary.txt` (with `Result is : VALID/INVALID`), `mlperf_log_detail.txt`, accuracy log.
Run lengths are governed by `min_duration_ms` and `min_query_count`; LoadGen's own defaults are far longer
than a smoke test wants, and MLPerf's real settings come from `mlperf.conf` plus per-benchmark `user.conf`.

What `workloads/loadgen-plumbing-check` does (run on this machine, CPU target, 6.6 s): a pure-Python SUT answers each
of 256 samples with the 4 bytes of an 8x8 integer matrix-vector product. Offline in AccuracyOnly mode must
return all 256 samples with the right bytes; Offline and Server in PerformanceOnly mode must be called
`VALID` by LoadGen against deliberately small settings (100 ms, 1024 queries). The reference output is a fixed
string; **no throughput or latency is recorded** (the SUT is a toy and the host is shared). It proves the
bindings, scenarios and accuracy log work in this repo's runner, and nothing more.

## 4. Recommended plan

1. **Keep the plumbing check** as the CI-able proof that LoadGen integration works (done; the only part verified).
2. **Whisper + LoadGen** is the natural next step and fits the whisper.cpp work: a LoadGen QSL over a small
   set of LibriSpeech utterances and a SUT that calls `whisper-cli` / the whisper.cpp library, scored by WER,
   Offline and SingleStream. Prerequisites before writing it: read LibriSpeech's licence at openslr.org and
   the whisper model card, pin dataset MD5s from `inference_librispeech.csv`, and decide a subset size. Label
   it "LoadGen-driven whisper.cpp tiny.en run", not an MLPerf whisper result: the reference model is large-v3,
   and a different model is at most an Open-division experiment which also needs MLCommons review.
3. **BERT-Large/SQuAD** second, only after SQuAD and the Zenodo checkpoints' terms are read.
4. Do **not** pursue the gated and member-only benchmarks. Do not use the MLPerf name for workload names.
5. For GPU numbers, record them through `bin/pw run --bench` under this repo's own workload names, and
   cite LoadGen as the load generator. If an MLCommons-reviewed result ever matters, link to the official
   published entry rather than reproducing it.
