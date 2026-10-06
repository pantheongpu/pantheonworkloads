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
| `compare` | for `functional` | How a result is compared to `reference.json`: `exact` text, or `tolerance` (top-level `tolerance: {abs, rel}`). Numbers match within `abs + rel*|reference|`, lists element-wise, strings and booleans exactly. `output` may be an object: keys are compared one by one, and `tolerance.fields: {<key>: {abs, rel}}` gives a key its own bounds. Put token ids in a string ("318 262") to have them compared exactly beside floats. |
| `unsupported_ok` | no | List of top-level `output` keys. The lib-* workloads report an op the backend has no kernel for as the string `"unsupported"`. When the *reference* says `unsupported` and the target returns numbers, that key is unchecked (counted in the run's detail: `unchecked: N ops`). When the reference has numbers and the target says `unsupported`, the run FAILS, unless the key is listed here (an op that real GPUs lack too, e.g. FlashAttention in fp32). |
| `timeout_s` | no | Default 600. |
| `notes` | no | Anything a reader should know before trusting a result. |

A manifest with `model.revision: null` is **unpinned**: `validate` warns, and results from it
are not reproducible until the revision is filled in.
