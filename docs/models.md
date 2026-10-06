# Models

Weights are never committed; manifests point at them. A licence is recorded only after reading the model's
own card or repository. When that was not possible the manifest says `UNVERIFIED` in `model.licence` and
`bin/pw validate` warns.

**Status of this list: huggingface.co was not reachable from the machine that wrote it (the proxy answered
403 to CONNECT), so no model card could be read, no file downloaded or hashed, and no revision pinned.**
GitHub was reachable, which is how llama.cpp's MIT licence was read.

| Model | Workload | Licence | State |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct (GGUF) | `llamacpp-smollm2-135m`, `llamacpp-bench-smollm2-135m` | UNVERIFIED (expected Apache-2.0) | file name and URL are from memory; revision and sha256 not pinned |
| Mistral-7B-v0.3 | `llamacpp-bench-mistral-7b-v03` (benchmark only, real GPU) | UNVERIFIED (believed Apache-2.0; the original repo may also require accepting terms on Hugging Face) | no default download; you supply the GGUF |
| Qwen2.5-0.5B-Instruct, TinyLlama-1.1B, SmolLM2-360M | not added | not read | candidates (expected Apache-2.0); add after the cards are read |
| Llama family (Meta) | not included: needs the user to accept the licence | Llama community licence (gated) | not fetched; bring your own GGUF with `PW_MODEL_FILE` and add a manifest that cites the licence you accepted |

To pin a model: read its card, put the licence and the commit hash in the manifest, download the file,
`sha256sum` it into `workloads/<name>/model.sha256`, and use the commit-pinned `resolve/<hash>/...` URL.
