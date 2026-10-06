# Recording benchmark numbers

Benchmark numbers are for **real GPUs only**. The simulator runs on a CPU, so `--bench` is refused
for `sim:` targets, and a workload that reports `metrics` on a simulated target fails.

```bash
bin/pw run smollm2-135m-ollama --target gpu --bench --repeat 5 [--device "NVIDIA H100 80GB HBM3"]
bin/pw bench-table            # markdown table of everything recorded under bench/
```

`--bench` runs the workload `--repeat` times (default 3) and writes
`bench/<workload>/<UTC time>-<device>.json` only when **every** run passed its functional check.
A number from a run that produced the wrong answer is never recorded.

## Record format (`pw-bench/0`)

```json
{
  "schema": "pw-bench/0",
  "workload": "smollm2-135m-ollama",
  "date": "2026-10-06T12:00:00Z",
  "environment": {
    "target": "gpu",
    "device": "NVIDIA H100 80GB HBM3",
    "driver": "…",
    "runtime": "ollama",
    "runtime_version": "ollama version is …",
    "model": {"id": "smollm2:135m", "revision": "…", "licence": "…", "source_url": "…"},
    "host_os": "Linux-…",
    "repo_commit": "…",
    "repo_dirty": false
  },
  "repeats": 5,
  "metrics": {"decode_tokens_per_s": {"median": 0.0, "values": [0.0, 0.0, 0.0, 0.0, 0.0]}}
}
```

(The numbers above are placeholders, not measurements.)

## Rules for numbers worth keeping

- **The device must be known.** NVIDIA cards are detected with `nvidia-smi`; for AMD and anything
  else pass `--device`. Without a device the run is refused, because an unattributed number is useless.
- **Pin the model.** A manifest with `model.revision: null` records `null`; pin it first or the
  number cannot be reproduced.
- **A dirty tree is recorded** (`repo_dirty`). Do not commit results from one.
- **One machine per record.** Compare cards by comparing records, never by averaging them.
- Record clocks, power limits and the like by hand in the commit message when they were changed
  from the defaults; the runner does not read them yet.

## What is not done yet

- No real GPU has run anything here, so no benchmark has been recorded, and `smollm2-135m-ollama`
  itself has not been run (ollama is not installed on the machine this was written on).
- Metrics beyond decode tokens/s (prompt processing, time to first token, memory) are not collected.
- AMD devices are not auto-detected, and driver/ROCm versions are not collected for them.
