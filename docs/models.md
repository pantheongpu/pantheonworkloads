## Speech and LoadGen workloads (whisper.cpp, MLCommons LoadGen)

Added by the `wl-speech-mlperf` branch. Weights and audio are never committed; they are fetched at run
time by `tools/whisper-assets.sh` and checked against pinned checksums.

| Item | Version pinned | Licence and where it was verified | Checksum |
| --- | --- | --- | --- |
| whisper.cpp (runtime) | tag `v1.9.5`, commit `d1be6fde11ac6e0407606b4e42fe72d34add8037` | MIT, `LICENSE` at that commit ("Copyright (c) 2023-2026 The ggml authors") | commit hash checked by `tools/build-whisper-cpp.sh` |
| Whisper tiny.en weights, ggml format (`ggml-tiny.en.bin`, 75 MiB) | `ggerganov/whisper.cpp` on Hugging Face (HF commit not pinned) | The model is OpenAI's: openai/whisper `README.md` at `86098128c0b4f24f0e2aa2994de830614b474227` says "Whisper's code and model weights are released under the MIT License". The ggml conversion's own model card was **not read** (Hugging Face was unreachable where this was written) | sha1 `c78c86eb1a8faa21b369bcd33207cc90d64ae9df` from whisper.cpp `models/README.md` at v1.9.5. **No sha256 recorded yet**: the file could not be downloaded to compute it |
| Test clip `jfk.wav` (11 s, 16 kHz mono, 352078 bytes) | `samples/jfk.wav` at the same commit, fetched from raw.githubusercontent.com | **Not verified.** The whisper.cpp repo states no provenance or licence for the recording (`samples/README.md` only says the folder holds "audio files used for testing"). The words are from a 1961 US presidential address, but the recording's copyright status is unknown to this repo. Therefore it is fetched, not committed | sha256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| `mlcommons-loadgen` (pip) | `6.0.17` | Apache-2.0: `LICENSE.md` of github.com/mlcommons/inference at `3fbc329939999c13d0a7b5e67fb2092287e06047` (the wheel's own metadata licence field is empty) | cp313 manylinux x86_64 wheel sha256 `79090cf79054bad8142d00b59aa81f40dc6e19cc5ef83ccafca9fce6d8ddaabb` (informational; installs are not hash-pinned because wheels differ per platform) |

MLPerf benchmark models and datasets are **not** used by any workload; see `docs/mlperf.md` for what
was found about their terms (several are member-only or gated, several unverified).
