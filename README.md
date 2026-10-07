# pantheonworkloads

Open-source AI workloads (LLM inference and more) with pinned models and reproducible benchmarks, run on real GPUs and on pantheonsim's simulated NVIDIA and AMD GPUs to check functional correctness.

## Two kinds of numbers, kept apart

| | Where it runs | What it records | What it is for |
| --- | --- | --- | --- |
| **Functional** | [pantheonsim](https://github.com/pantheongpu/pantheonsim) profile, or a real GPU | pass / fail against a reference output, within a stated tolerance | "does the simulated GPU model behave like the real one for this workload?" |
| **Benchmark** | real GPU only | tokens/s (more to come), with device, driver, runtime and model revision | "how fast is this workload on this card?" This repo is where those numbers are kept. |

The simulator executes on a CPU, so its wall-clock time says nothing about GPU speed.
Functional results therefore never carry performance numbers, and benchmark files are
refused for simulated targets.

## Layout

```
workloads/<name>/manifest.yaml   what the workload is: source, model pointer, licence, tolerance
workloads/<name>/run.sh          runs it; prints one JSON object on its last line (docs/workload-contract.md)
workloads/<name>/reference.json  expected output, recorded on a trusted target (absent until recorded)
bin/pw                           the runner: pw list | validate | run | record | matrix | bench-table
bench/<name>/*.json              recorded benchmark numbers from real GPUs (committed)
results/                         functional run output (git-ignored)
docs/                            manifest, workload contract, benchmarks, trace schema (draft)
```

## Quick start

```bash
bin/pw validate                                   # check every manifest
bin/pw run selftest --target cpu                  # no GPU, no model: exercises the runner
bin/pw run smollm2-135m-ollama --target sim:nvidia/h100   # needs ollama and a pantheonsim build
bin/pw matrix results/                            # one table from every run's results.tsv
bin/pw run smollm2-135m-ollama --target gpu --bench --repeat 5   # real GPU: record benchmark numbers
bin/pw bench-table                                # everything recorded under bench/
```

`--target` is `cpu`, `gpu` (whatever the host has) or `sim:<vendor>/<profile>`
(for example `sim:amd/mi300x`). For `sim:` targets `PANTHEONSIM_DIR` must point at a
built pantheonsim checkout (its `build/vgpu`).

Results go to `results/<run-id>/`: `results.tsv` (workload, result, seconds, detail, the same
columns pantheonsim's workload matrix reads), `machine.txt`, and one `<workload>.json` per workload.

## Models and licences

The repository stores **pointers** to models (id, revision, checksum), never weights, and each
manifest records the model's licence. Models are fetched by the workload's own script.
Check a model's licence before adding it: the terms differ a lot between model families and
change between versions.

## Status

Early scaffold. What exists and what does not:

- Runner, manifest validation, results format, a no-GPU `selftest`: exist and are tested (`tests/`).
- `smollm2-135m-ollama`: written, **not yet run** and no reference recorded. See its manifest.
- `llamacpp-*` (llama.cpp functional + `llama-bench` workloads, `docs/llamacpp.md`, `docs/models.md`): see the status there; model licences/revisions are **unverified**.
- `pytorch-microsuite`, `gpt2-small-pytorch`, `bert-base-uncased-pytorch`, `resnet18-randinit-pytorch`: written,
  **not yet run** (no PyTorch or Hugging Face access where they were written), no references. See `docs/models.md`.
- Benchmark recording (`--bench`, `bench-table`): exists and is tested with the selftest, but **no real GPU has
  run it yet**, so `bench/` is empty. See `docs/benchmarks.md`.
- Trace capture: designed only (`docs/trace-schema.md` is a draft).
- **Coverage workloads that need no model download** (seeded random weights; they test kernels, graphs and
  architecture code paths, **not** the quality of any named pretrained model): `arch-*` (18 open LLM / vision /
  speech architectures, `docs/arch-coverage.md`), `lib-*` (251 GPU-library ops: FFT, linear algebra, sparse, RNN,
  convolution, attention, precision; `docs/lib-coverage.md`), `llamacpp-synth-*` (synthetic GGUFs across six
  architectures and many quantization types; `docs/llamacpp-synth.md`). References were recorded **on the CPU
  only**; tolerances are unverified on any GPU or simulated GPU.
- **Pretrained models whose licence and weights were read from source and pinned by sha256**: Silero VAD (MIT),
  PP-OCRv3 via RapidOCR (Apache-2.0), ONNX zoo MNIST and MobileNetV2, spaCy `en_core_web_sm` (MIT), and nine vision
  models (YuNet MIT, SFace, PP-HumanSeg, NanoDet, YOLOX and CRNN from OpenCV Zoo, Apache-2.0; ONNX zoo SSD-MobileNetV1
  MIT, ShuffleNet-v2 BSD-3-Clause, EfficientNet-Lite4 MIT; plus a `-bench` workload each). CPU references
  recorded; the `-bench` workloads have not run on a GPU. See `docs/pretrained-reachable.md`.
- `whisper-cpp-tiny-en` (functional) and `whisper-cpp-bench-tiny-en` (GPU benchmark): written; the CPU build of
  whisper.cpp v1.9.5 was built and run here, but the tiny.en model could not be downloaded (Hugging Face blocked),
  so **no transcript has been produced and no reference or benchmark is recorded**. See their manifests and `docs/models.md`.
- `loadgen-plumbing-check`: runs MLCommons LoadGen against a toy CPU SUT; reference recorded on the cpu target.
  **Not an MLPerf result.** Feasibility notes on MLPerf Inference: `docs/mlperf.md`.
