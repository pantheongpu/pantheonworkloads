# Pretrained models whose weights and licence could be read

Earlier workloads had to mark model licences UNVERIFIED because model hubs (Hugging Face and others) are blocked
where they were written. The models below were chosen the other way round: only where **both the weights and
the licence text were fetched and read** from hosts that are reachable (PyPI, `raw.githubusercontent.com`,
`media.githubusercontent.com` for git-LFS objects, GitHub release downloads). Weights are never committed:
each workload's `model.sha256` pins them and `tools/ort_assets.py` downloads and checks them at run time.

Written 2026-10-06. The licences were read on that date at the pins below.

| Model | What it does | Licence, and where it was read | Pin | Size | Workloads | State |
| --- | --- | --- | --- | --- | --- | --- |
| Silero VAD (`silero_vad.onnx`) | voice activity detection, 16 kHz | **MIT**: `LICENSE` of snakers4/silero-vad at commit `5cd7945676eb32225748052e2e6a0580e4686a08` (tag v6.2.3) and `silero_vad-6.2.3.dist-info/licenses/LICENSE` in the PyPI wheel. (The wheel's README badge text says "CC BY-NC 4.0" but links to the MIT licence; the LICENSE file is MIT.) | sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`, identical in the git tag and the wheel | 2.3 MB | `silero-vad-onnx`, `silero-vad-onnx-bench` | cpu reference recorded and re-run PASS |
| PP-OCRv3 det + rec, PP-OCR v2.0 cls (ONNX, converted from PaddleOCR by RapidOCR) | text detection, orientation, recognition | **Apache-2.0**: RapidOCR `README.md` "License / Models" section at commit `f65c7da00e72c19c258245e8e0e5f33af14488be` ("derived from official PaddleOCR models ... distributed under the Apache License, Version 2.0 ... converted artifacts redistributed under the same terms"); `LICENSE` of RapidAI/RapidOCR and of PaddlePaddle/PaddleOCR (both Apache-2.0, `main` on 2026-10-06); wheel `METADATA` `License: Apache-2.0`. The README links a `MODEL_LICENSES.md` that does **not exist** (404) at that commit, so per-file attribution was not available. | PyPI `rapidocr_onnxruntime==1.2.3`; sha256 `3439588c...` (det), `897a3ede...` (rec), `e47acedf...` (cls), full values in `workloads/ppocr-rapidocr/model.sha256` | 13.7 MB | `ppocr-rapidocr`, `ppocr-rapidocr-bench` | cpu reference recorded and re-run PASS |
| MNIST-12 (ONNX model zoo) | digit classifier | **MIT**: "License" section of `validated/vision/classification/mnist/README.md` in onnx/models at commit `c5cae0f1942c8d7727372c7b1f6b485aa39e4421` (the repo's own `LICENSE` is Apache-2.0) | `mnist-12.onnx` sha256 `5c688690...` (equals its git-LFS oid); test tarball `a53a59dc...` | 26 KB | `onnx-zoo-mnist` | cpu reference recorded and re-run PASS; matches the zoo's own expected logits (diff 0) |
| MobileNetV2-12 (ONNX model zoo) | ImageNet classifier | **Apache-2.0**: "License" section of `.../mobilenet/README.md` at the same onnx/models commit, and the repo `LICENSE`. Says nothing about the ImageNet training data's terms. | `mobilenetv2-12.onnx` sha256 `c0c3f76d...` (= git-LFS oid); tarball `5f83b422...` | 14 MB | `onnx-zoo-mobilenetv2`, `onnx-zoo-mobilenetv2-bench` | cpu reference recorded and re-run PASS; matches the zoo's expected logits (diff 0) |
| spaCy `en_core_web_sm` 3.8.0 | tokenizer, POS tagger, parser, NER (English) | **MIT**: `en_core_web_sm-3.8.0.dist-info/LICENSE` (Copyright 2021 ExplosionAI GmbH) and `meta.json` `"license": "MIT"`, inside the wheel. Its `LICENSES_SOURCES` lists training data: OntoNotes 5 ("commercial (licensed by Explosion)"), WordNet 3.0 (own licence, text in the file), ClearNLP guideline (citation only): read those before building on the model. | release `en_core_web_sm-3.8.0` of explosion/spacy-models; wheel sha256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85` | 12.8 MB | `spacy-en-core-web-sm`, `spacy-en-core-web-sm-bench` | cpu reference recorded and re-run PASS |

Inputs that are not models: the Silero clip (`tests/data/test.wav`, 60 s) and the RapidOCR screenshot
(`python/tests/test_files/en.jpg`) come from those MIT / Apache-2.0 repositories at the pinned commits, but
**neither repository states the recording's or the screenshot's own provenance**, so they are fetched and
never committed. The MNIST digit and MobileNet tensor come from the zoo's own test tarballs. The spaCy
sentences were written for this repo.

## How they run

- Runtime: onnxruntime 1.30.0 (MIT) for the first four; spaCy 3.8.16 for the last. `tools/ort-env.sh` /
  `tools/spacy-env.sh` create venvs under `~/.cache/pantheonworkloads/venvs` (`ort-cpu`, `ort-gpu`,
  `spacy-cpu`, `spacy-gpu`) from `tools/ort-requirements.txt` / `tools/spacy-requirements.txt`, or use
  `PW_ORT_PYTHON` / `PW_SPACY_PYTHON`. No PyTorch is needed (so no conda helper was written).
- **Targets: cpu and gpu only.** No `sim:*` target: nothing in `/home/user/pantheonsim` (read-only look) shows
  onnxruntime or spaCy running on the simulator, so none was claimed.
- The gpu target never falls back to the CPU: `tools/ort_tasks.py` picks the first of CUDA, ROCm, MIGraphX that
  onnxruntime offers, checks the session really uses it, and otherwise exits 77. The ort venv for gpu installs
  `onnxruntime-gpu==1.30.0`, which also needs CUDA and cuDNN libraries on the machine (not verified here); for
  AMD point `PW_ORT_PYTHON` at an onnxruntime build with the ROCm or MIGraphX provider. RapidOCR 1.2.3 can only
  use CUDA, so the OCR workloads are a SKIP on AMD. spaCy's GPU path (cupy) is NVIDIA only.
- Functional outputs: Silero segments as sample offsets (exact; the mean probability within 0.002); OCR lines
  (exact; mean score within 0.01); zoo class ids exact and logits within 0.01 + 0.1 %, and each run also fails if the
  logits differ from the zoo's own expected output by more than 1e-3; spaCy tokens, POS, dependency
  labels/heads and entities exact. Silero's segment post-processing is a re-write of `get_speech_timestamps_from_probs`
  (v6.2.3, defaults, no maximum speech length) and was cross-checked against the upstream function on 300
  random probability sequences (0 differences). Exact comparisons are expected to hold on GPUs for these small
  models, but a GPU that flips one window or one logit across a threshold would show up as a failure to
  investigate, not to tolerate.
- Benchmarks (`*-bench`, gpu only, metrics only on real targets): Silero windows/s and real-time factor
  (batch 1, so per-call latency), PP-OCRv3 images/s end to end, MobileNetV2 images/s at batch 32 on host arrays,
  spaCy words/s. **None has been run on a GPU: no GPU is available where this was written, so `bench/` is empty.**
  Their code paths were exercised on the CPU with the provider check bypassed (numbers not recorded).

## What was run

On the cpu target (4 shared cores, onnxruntime 1 thread): the five functional workloads were run, their
references recorded with `bin/pw record --target cpu` (recorder: the CPU backend of onnxruntime 1.30.0 /
spaCy 3.8.16, Python 3.13.16, numpy 2.5.3) and re-run to PASS. On the gpu target here every workload is a clean
SKIP (no NVIDIA GPU). Unit tests: `tests/test_ort_tools.py`.

## Investigated and not added

| Candidate | Finding |
| --- | --- |
| `silero-vad` PyPI package itself | needs PyTorch for its Python API; its ONNX file is what is used directly. |
| `rapidocr` 3.9.2 (PP-OCRv6 small, 31 MB wheel) | newer, same Apache-2.0 statement; not added to keep one OCR model. The v1.2.3 wheel was used because it is small and pinned. |
| `onnx` wheel bundled test data | 476 files, but node-test graphs of single operators, not pretrained models. |
| `mediapipe` 1.1.0 wheel | LICENSE (Apache-2.0) and NOTICE are there, but no `.tflite`/`.task` model files are bundled (they are downloaded from Google storage, blocked). |
| `vosk` wheel / models | the wheel has only the runtime; models live on `alphacephei.com`, blocked (CONNECT 403). |
| `piper` | the binary release downloads from GitHub, but voices are on Hugging Face (blocked). |
| `en_core_web_sm` from PyPI | not on PyPI (index has no such distribution); the GitHub release asset is used instead. |
| other onnx/models entries (ResNet-50 etc.) | same licence mechanism as MobileNetV2 and reachable; not added for size (98 MB) and because per-model README licence lines would each need reading. |
| onnx/models via `raw.githubusercontent.com` | serves a 130-byte LFS pointer, not weights; `media.githubusercontent.com/media/...` serves the real file (sha256 equals the pointer's oid). |
