# Pretrained models whose weights and licence could be read

Earlier workloads had to mark model licences UNVERIFIED because model hubs (Hugging Face and others) are blocked
where they were written. The models below were chosen the other way round: only where **both the weights and
the licence text were fetched and read** from hosts that are reachable (PyPI, `raw.githubusercontent.com`,
`media.githubusercontent.com` for git-LFS objects, GitHub release downloads). Weights are never committed:
each workload's `model.sha256` pins them and `tools/ort_assets.py` downloads and checks them at run time.

Written 2026-10-06; the speech / audio models (second half of the table) were added 2026-10-07. The licences were read on those dates at the pins below.

| Model | What it does | Licence, and where it was read | Pin | Size | Workloads | State |
| --- | --- | --- | --- | --- | --- | --- |
| Silero VAD (`silero_vad.onnx`) | voice activity detection, 16 kHz | **MIT**: `LICENSE` of snakers4/silero-vad at commit `5cd7945676eb32225748052e2e6a0580e4686a08` (tag v6.2.3) and `silero_vad-6.2.3.dist-info/licenses/LICENSE` in the PyPI wheel. (The wheel's README badge text says "CC BY-NC 4.0" but links to the MIT licence; the LICENSE file is MIT.) | sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`, identical in the git tag and the wheel | 2.3 MB | `silero-vad-onnx`, `silero-vad-onnx-bench` | cpu reference recorded and re-run PASS |
| PP-OCRv3 det + rec, PP-OCR v2.0 cls (ONNX, converted from PaddleOCR by RapidOCR) | text detection, orientation, recognition | **Apache-2.0**: RapidOCR `README.md` "License / Models" section at commit `f65c7da00e72c19c258245e8e0e5f33af14488be` ("derived from official PaddleOCR models ... distributed under the Apache License, Version 2.0 ... converted artifacts redistributed under the same terms"); `LICENSE` of RapidAI/RapidOCR and of PaddlePaddle/PaddleOCR (both Apache-2.0, `main` on 2026-10-06); wheel `METADATA` `License: Apache-2.0`. The README links a `MODEL_LICENSES.md` that does **not exist** (404) at that commit, so per-file attribution was not available. | PyPI `rapidocr_onnxruntime==1.2.3`; sha256 `3439588c...` (det), `897a3ede...` (rec), `e47acedf...` (cls), full values in `workloads/ppocr-rapidocr/model.sha256` | 13.7 MB | `ppocr-rapidocr`, `ppocr-rapidocr-bench` | cpu reference recorded and re-run PASS |
| MNIST-12 (ONNX model zoo) | digit classifier | **MIT**: "License" section of `validated/vision/classification/mnist/README.md` in onnx/models at commit `c5cae0f1942c8d7727372c7b1f6b485aa39e4421` (the repo's own `LICENSE` is Apache-2.0) | `mnist-12.onnx` sha256 `5c688690...` (equals its git-LFS oid); test tarball `a53a59dc...` | 26 KB | `onnx-zoo-mnist` | cpu reference recorded and re-run PASS; matches the zoo's own expected logits (diff 0) |
| MobileNetV2-12 (ONNX model zoo) | ImageNet classifier | **Apache-2.0**: "License" section of `.../mobilenet/README.md` at the same onnx/models commit, and the repo `LICENSE`. Says nothing about the ImageNet training data's terms. | `mobilenetv2-12.onnx` sha256 `c0c3f76d...` (= git-LFS oid); tarball `5f83b422...` | 14 MB | `onnx-zoo-mobilenetv2`, `onnx-zoo-mobilenetv2-bench` | cpu reference recorded and re-run PASS; matches the zoo's expected logits (diff 0) |
| spaCy `en_core_web_sm` 3.8.0 | tokenizer, POS tagger, parser, NER (English) | **MIT**: `en_core_web_sm-3.8.0.dist-info/LICENSE` (Copyright 2021 ExplosionAI GmbH) and `meta.json` `"license": "MIT"`, inside the wheel. Its `LICENSES_SOURCES` lists training data: OntoNotes 5 ("commercial (licensed by Explosion)"), WordNet 3.0 (own licence, text in the file), ClearNLP guideline (citation only): read those before building on the model. | release `en_core_web_sm-3.8.0` of explosion/spacy-models; wheel sha256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85` | 12.8 MB | `spacy-en-core-web-sm`, `spacy-en-core-web-sm-bench` | cpu reference recorded and re-run PASS |
| Moonshine tiny English (int8 ONNX; sherpa-onnx export of Useful Sensors' model) | speech recognition, 16 kHz | **MIT**: `LICENSE` inside the sherpa-onnx archive (MIT, Copyright 2024 Useful Sensors, sha256 `29f60769...`), and the "License" section of usefulsensors/moonshine `README.md` at commit `234f60faa0eb388b01cdf7e60aca232af37aefda` ("The models are MIT by default ... the only exceptions are the legacy non-streaming models for languages other than English"). | k2-fsa/sherpa-onnx release tag `asr-models` (rolling tag): `sherpa-onnx-moonshine-tiny-en-int8.tar.bz2` sha256 `d5fe6ec4...`; every member used pinned in `workloads/moonshine-tiny-en-onnx/model.sha256` | 108 MB archive, 124 MB used | `moonshine-tiny-en-onnx`, `moonshine-tiny-en-onnx-bench` | cpu reference recorded and re-run PASS; transcripts identical to sherpa-onnx 1.13.8 |
| WeSpeaker ResNet34, VoxCeleb (ONNX) | speaker embedding (256-d), 16 kHz | **CC-BY-4.0**: wenet-e2e/wespeaker `docs/pretrained.md` at commit `9fecd6cb4f47475d01761d87c826298dff4ef18c`, "Model License": the pretrained models follow their dataset's licence, "the pretrained model on VoxCeleb follows Creative Commons Attribution 4.0 International License"; the code is Apache-2.0 (README badge). The weights file comes from the sherpa-onnx release (metadata `url` = WeSpeaker's `voxceleb_resnet34.onnx`); the release text itself was not readable here. VoxCeleb's own terms were not read. | release tag `speaker-recongition-models` (sic): `wespeaker_en_voxceleb_resnet34.onnx` sha256 `5ef208a9...` | 26.5 MB | `wespeaker-resnet34-onnx`, `wespeaker-resnet34-onnx-bench` | cpu reference recorded and re-run PASS |
| Kokoro v0.19 int8 (ONNX; hexgrad/Kokoro-82M "kLegacy", sherpa-onnx export) | text-to-speech, English, 24 kHz, 11 voices | **Apache-2.0**: `LICENSE` inside the sherpa-onnx archive (full Apache-2.0 text, sha256 `cfc7749b...`); the archive's README points to huggingface.co/hexgrad/Kokoro-82M, which is blocked here and was not read. | release tag `tts-models` (rolling): `kokoro-int8-en-v0_19.tar.bz2` sha256 `c9f0dd39...`; members pinned in `model.sha256` | 103 MB archive, 140 MB used | `kokoro-tts-int8-onnx`, `kokoro-tts-int8-onnx-bench` | cpu reference recorded and re-run PASS (run-to-run noise in the vocoder: compared by statistics, see below) |
| GTCRN (streaming ONNX, `gtcrn_simple.onnx`) | speech enhancement / denoising, 16 kHz, 0.5 MB | **MIT**: `LICENSE` of Xiaobin-Rong/gtcrn at commit `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73` (Copyright 2024 Rong Xiaobin). The README has no separate statement for the weights (`gtcrn_simple.onnx` is the ONNX export of `model_trained_on_dns3.tar` made by `stream/gtcrn_stream.py`). | `raw.githubusercontent.com` at that commit: `stream/onnx_models/gtcrn_simple.onnx` sha256 `b4718df6...` | 0.5 MB | `gtcrn-enhance-onnx`, `gtcrn-enhance-onnx-bench` | cpu reference recorded and re-run PASS; output equals the repo's own `enh.wav` to 16-bit rounding |
| Streaming Zipformer2 keyword spotter, GigaSpeech, 3.3 M parameters (ONNX, icefall via sherpa-onnx) | keyword spotting / wake words (custom keyword lists) | **Apache-2.0**, as the model's own card states: `README.md` inside the sherpa-onnx archive, front matter `license: Apache License 2.0` (sha256 `74e42d37...`); icefall and sherpa-onnx are Apache-2.0. **Open point:** the terms of the training data (GigaSpeech) were not found (its repository README states no data licence; `TERMS_OF_ACCESS` is 404 at commit `44289e4bb6d65291456cf94de495ea83668d1df1`). | release tag `kws-models` (rolling): `sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2` sha256 `f170013b...` | 17.6 MB archive, 14 MB used | `kws-zipformer-gigaspeech-onnx`, `kws-zipformer-gigaspeech-onnx-bench` | cpu reference recorded and re-run PASS; detections identical to sherpa-onnx 1.13.8 |
| Zipformer small audio tagger, AudioSet, 527 classes (ONNX, icefall via sherpa-onnx) | audio classification / sound-event tagging, 16 kHz | **Apache-2.0**, as the model's own card states: `README.md` inside the sherpa-onnx archive, front matter `license: apache-2.0` (sha256 `a00e3521...`); the card text is the Hugging Face model card, which could not be read directly. AudioSet's terms (clips from YouTube; labels CC-BY-4.0) were not read. | release tag `audio-tagging-models` (rolling): `sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2` sha256 `07e2fafc...` | 111 MB archive, 92 MB used | `zipformer-audio-tagging-onnx`, `zipformer-audio-tagging-onnx-bench` | cpu reference recorded and re-run PASS; top-3 classes and scores identical to sherpa-onnx 1.13.8 |

Inputs that are not models (first group): the Silero clip (`tests/data/test.wav`, 60 s) and the RapidOCR screenshot
(`python/tests/test_files/en.jpg`) come from those MIT / Apache-2.0 repositories at the pinned commits, but
**neither repository states the recording's or the screenshot's own provenance**, so they are fetched and
never committed. The MNIST digit and MobileNet tensor come from the zoo's own test tarballs. The spaCy
sentences were written for this repo.

Inputs of the speech / audio workloads: the Moonshine, KWS and audio-tagging clips are the `test_wavs` shipped inside
the sherpa-onnx archives (LibriSpeech-style read English for ASR and KWS; short sound-event clips for tagging), the
speaker workload uses the Silero clip and GTCRN's `mix.wav`, GTCRN uses its repository's `mix.wav` / `enh.wav`. **None of
those sources states the recordings' provenance or licence**, so they are fetched at run time and never committed; the
TTS workload needs no audio (two hand-typed phoneme strings), and the model-free coverage workloads synthesise theirs.

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

### Speech / audio workloads (added 2026-10-07)

- Runtime: onnxruntime directly, **not** sherpa-onnx, so the gpu target uses the same provider check as the rest (the
  sherpa-onnx PyPI wheels, `sherpa-onnx` + `sherpa-onnx-core` 1.13.8, Apache-2.0, are CPU builds; CUDA builds are published elsewhere
  (not checked)). Feature extraction (Kaldi fbank, STFT/iSTFT), greedy / keyword beam search, the keyword graph and
  tokenisation are numpy re-writes in `tools/speech_tasks.py`, written after the sherpa-onnx C++ sources and the
  upstream scripts. Same venvs and `tools/ort-env.sh` as before; no new Python packages.
- Archive assets: `tools/ort_assets.py` now supports members of a release archive (`MEMBERS`): the archive is pinned by
  sha256 in `model.sha256` next to each member, downloaded once, the pinned members extracted into the cache (each
  checked) and the archive deleted, so only the needed files stay on disk.
- Checked against sherpa-onnx 1.13.8 on the development machine (CPU, not part of the workloads): Moonshine transcripts
  and token counts, the keyword detections (keyword and firing chunk), the audio-tagging top-3 and scores, and the
  speaker model's embeddings when fed the same features (cosine 1.0000000, max diff 2e-6). The numpy fbank agrees with
  kaldi-native-fbank 1.22.3 to 1e-4 (log-mel). GTCRN's output equals the repository's own `enh.wav`. Kokoro's output, read
  back by Moonshine tiny, gives the intended sentences ("The quick brown fox jumps over the lazy dog.", "Hello world,
  this is a test.").
- Preprocessing choices worth knowing: the speaker workload follows WeSpeaker's own `infer_onnx.py` (Hamming window,
  `snip_edges`, mean subtraction), **not** sherpa-onnx, which for this file computes features differently and skips the
  mean subtraction (same-recording cosine 0.88 / different 0.52 there, 0.84 / 0.03 with WeSpeaker's recipe on the
  clips used).
- Comparison: ASR transcripts and token ids, KWS detections and tagging class ids exact; embeddings abs 0.01, cosines abs
  0.003; TTS: the Kokoro vocoder draws random numbers, so two runs on the same CPU differ by up to 0.05 per sample
  (0.3 peak). The workload compares sample counts (exact), RMS (abs 0.003), spectral centroid (abs 150 Hz) and the RMS in
  100 ms windows (abs 0.01), whose observed run-to-run spread was 0.0001, 30 Hz and 0.002. GTCRN: sample count exact,
  RMS in 0.5 s windows abs 0.003. These bounds are reasoned from CPU runs only; no GPU has run them.
- int8 caveat: Moonshine and Kokoro are only available as int8 here. A CUDA/ROCm provider lacks kernels for some integer ops,
  so onnxruntime may place those nodes on its CPU provider inside the same session; the session check (GPU provider first)
  passes, but GPU timings of those two would be a mixed placement. The other four are fp32.
- Benchmarks: the `*-bench` workloads (gpu only) report real-time factor and latency (plus embeddings/s, frames/s, clips/s);
  their code paths were run on the CPU with the provider check bypassed (not recorded; e.g. Moonshine RTF 17, GTCRN 13,
  speaker 25, Kokoro 0.47, 1 thread). **None has been run on a GPU.**

## What was run

On the cpu target (4 shared cores, onnxruntime 1 thread): the five functional workloads (and, for the speech / audio
additions, the six listed above) were run, their
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

### Speech / audio candidates (2026-10-07)

| Candidate | Finding |
| --- | --- |
| openWakeWord pretrained models (dscripka/openWakeWord) | README at commit `368c03716d1e92591906a84949bc477f3a834455`, "License": the code is Apache-2.0 but "all of the included pre-trained models are licensed under ... CC BY-NC-SA 4.0 ... due to the inclusion of datasets with unknown or restrictive licensing". Non-commercial: not added. (Only its Google speech-embedding backbone is Apache-2.0, which is not a wake-word model on its own.) |
| CED audio tagging (sherpa-onnx `sherpa-onnx-ced-*-audio-tagging-2024-04-19`) | the archives carry no licence file or statement (README: "converted from https://github.com/RicherMans/CED"); RicherMans/CED `LICENSE` at commit `4b2149dc9ba1ed25c6348083f82dba03f00a5022` is **GPL-3.0**: copyleft, not in the permissive list. Not added. The Zipformer tagger above was used instead. |
| FunASR / SenseVoice (sherpa-onnx SenseVoice archives) | modelscope/FunASR `MODEL_LICENSE` at commit `66d7a4c264a5993a2a63ed00c1f402c296ee521a` is the custom "FunASR Model Open Source License Agreement v1.1" (attribution, retain model names, community-conduct clause, termination). Custom terms: not added. The sherpa archive's own `LICENSE` is a 71-byte file whose text was not read. |
| GigaAM (Russian), reverb diarization, TeleSpeech | sherpa-onnx's docs say GigaAM's licence is `GigaAM License_NC.pdf`, Revai/reverb-diarization is "non-commercial", and TeleSpeech uses the "TeleSpeech model community licence". NC or custom: not added (read in k2-fsa/sherpa docs at commit `c02f72ca1540163a54019e845127fa52d5de175b`; the licence documents themselves were not fetched). |
| TEN VAD | sherpa-onnx's VAD docs describe a "modified version" licence (TEN-framework/ten-vad `LICENSE`, not fetched): custom, not added. Silero VAD (MIT) is already in. |
| pyannote segmentation 3.0 (sherpa-onnx `sherpa-onnx-pyannote-segmentation-3-0`) | the archive has an MIT `LICENSE`, but pyannote distributes the model behind Hugging Face's gated access (user conditions; not verifiable here, Hugging Face is blocked). Gated: not added. |
| NeMo TitaNet small, 3D-Speaker ERes2Net (sherpa-onnx speaker models) | reachable (40 MB each) but no licence statement was read (NeMo and ModelScope cards are on blocked hosts), and WeSpeaker already covers speaker embeddings. Not added. |
| VITS LJSpeech (`vits-ljs.tar.bz2`) | card says `apache-2.0` but the weights are a third-party re-upload (`yo2266911/mb-vits-models`, Hugging Face) of the original VITS LJSpeech checkpoint, whose repository (jaywalnut310/vits) is MIT: unclear which terms govern the file, and it needs a lexicon/phonemiser. Not added (Kokoro covers TTS). |
| Piper voices (`vits-piper-*`) | per-voice dataset licences live in model cards on Hugging Face (blocked); the archives bundle the espeak-ng data directory (its licence was not read). Not added. |
| kokoro-onnx v1.0 (`thewh1teagle/kokoro-onnx` release `model-files-v1.0`: 92 MB int8 model, 28 MB voices) | reachable, but a second Kokoro adds little and its Python package needs espeak-ng/phonemizer; v0.19 from sherpa-onnx was used. Not added. |
| Whisper tiny.en ONNX (sherpa-onnx) | same model as the existing whisper.cpp workload; not duplicated. |
| `pocketsphinx` 5.1.1 wheel (29 MB, bundles the en-us acoustic and language models) | a CPU-only C library with no GPU path, so it would not exercise a GPU; not added. |
| streaming Zipformer English ASR, other sherpa-onnx ASR/TTS models | not examined: the six above already cover ASR, KWS, TTS, speaker embedding, enhancement and audio tagging. |
