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
- Benchmark recording (`--bench`, `bench-table`): exists and is tested with the selftest, but **no real GPU has
  run it yet**, so `bench/` is empty. See `docs/benchmarks.md`.
- Trace capture: designed only (`docs/trace-schema.md` is a draft).
