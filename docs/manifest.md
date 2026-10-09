# Workload manifest (`manifest.yaml`)

Every workload directory has one. `bin/pw validate` checks them against these rules
(`tools/validate.py`).

| Key | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Same as the directory name. Lower-case letters, digits, `-`. |
| `description` | yes | One line. |
| `kind` | yes | `functional` (checked against a reference) or `benchmark`. |
| `runtime` | yes | What executes the model: `ollama`, `llama.cpp`, `pytorch`, `none`, ... |
| `targets` | yes | List of target patterns it supports: `cpu`, `gpu`, `sim:nvidia/*`, `sim:amd/*`. |
| `runtime_version` | no | A pinned runtime version, recorded in benchmark records instead of asking `<runtime> --version` (for runtimes with no such command on PATH, e.g. a pinned llama.cpp build). |
| `model` | when a model is used (`model: null` for a runtime workload that downloads none, e.g. a micro-benchmark suite) | `id` (where it comes from), `revision` (digest or commit; `null` only while unpinned), `licence` (SPDX id or the licence's name), `source_url`. |
| `compare` | for `functional` | How a result is compared to `reference.json`: `exact` text, or `tolerance` (top-level `tolerance: {abs, rel}`). Numbers match within `abs + rel*|reference|`, lists element-wise, strings and booleans exactly. `output` may be an object: keys are compared one by one, and `tolerance.fields: {<key>: {abs, rel}}` gives a key its own bounds, or `{wer: X}` for a text key: the word error rate of the output against the reference text (case and punctuation ignored, `tools/wer.py`) must be at most X. Put token ids in a string ("318 262") to have them compared exactly beside floats. |
| `informational` | no | List of top-level `output` keys that are recorded (a reference documents them) but never compared; `bin/pw` appends `informational differs: <keys>` to the PASS detail when they differ from the reference. Use it for a measurement that explains a result, not for one that decides it (docs/int8-and-nondeterminism.md). |
| `unsupported_ok` | no | List of top-level `output` keys (or a mapping `{<target pattern>: [keys]}` for ops only some targets lack, e.g. `sim:nvidia/t4`). The lib-* workloads report an op the backend has no kernel for as the string `"unsupported"`. When the *reference* says `unsupported` and the target returns numbers, that key is unchecked (counted in the run's detail: `unchecked: N ops`). When the reference has numbers and the target says `unsupported`, the run FAILS, unless the key is listed here (an op that real GPUs lack too, e.g. FlashAttention in fp32). |
| `restricted` | no | `true` for a model under a non-permissive or gated licence (Llama community licence, Gemma terms, OpenRAIL++-M, SD3 community licence, anything gated). Needs `notes` saying which licence applies and what was read. A restricted workload is allowed in `bin/pw list/run/validate` and is marked `restricted` in the coverage table, but is **left out of every default selection**: `bin/pw list --default`, `bin/pw matrix` (unless `--include-restricted`) and anything that runs "all workloads" (use `default_names()` in `bin/pw`, or `validate.is_restricted`). Name it explicitly to run it. Default `false`. |
| `timeout_s` | no | Default 600. |
| `notes` | no | Anything a reader should know before trusting a result. |

A manifest with `model.revision: null` is **unpinned**: `validate` warns, and results from it
are not reproducible until the revision is filled in.
