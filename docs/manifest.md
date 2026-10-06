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
| `model` | when a model is used | `id` (where it comes from), `revision` (digest or commit; `null` only while unpinned), `licence` (SPDX id or the licence's name), `source_url`. |
| `compare` | for `functional` | How a result is compared to `reference.json`: `exact` text, or `tolerance` (with `abs`/`rel` numeric bounds). |
| `timeout_s` | no | Default 600. |
| `notes` | no | Anything a reader should know before trusting a result. |

A manifest with `model.revision: null` is **unpinned**: `validate` warns, and results from it
are not reproducible until the revision is filled in.
