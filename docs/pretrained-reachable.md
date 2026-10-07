# Pretrained models whose weights and licence could be read

Earlier workloads had to mark model licences UNVERIFIED because model hubs (Hugging Face and others) are blocked
where they were written. The models below were chosen the other way round: only where **both the weights and
the licence text were fetched and read** from hosts that are reachable (PyPI, `raw.githubusercontent.com`,
`media.githubusercontent.com` for git-LFS objects, GitHub release downloads). Weights are never committed:
each workload's `model.sha256` pins them and `tools/ort_assets.py` downloads and checks them at run time.

Written 2026-10-06; the text-model rows (second table), the vision section and the speech / audio models were added 2026-10-07. The licences were read on those dates at the pins below.

| Model | What it does | Licence, and where it was read | Pin | Size | Workloads | State |
| --- | --- | --- | --- | --- | --- | --- |
| Silero VAD (`silero_vad.onnx`) | voice activity detection, 16 kHz | **MIT**: `LICENSE` of snakers4/silero-vad at commit `5cd7945676eb32225748052e2e6a0580e4686a08` (tag v6.2.3) and `silero_vad-6.2.3.dist-info/licenses/LICENSE` in the PyPI wheel. (The wheel's README badge text says "CC BY-NC 4.0" but links to the MIT licence; the LICENSE file is MIT.) | sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`, identical in the git tag and the wheel | 2.3 MB | `silero-vad-onnx`, `silero-vad-onnx-bench` | cpu reference recorded and re-run PASS |
| PP-OCRv3 det + rec, PP-OCR v2.0 cls (ONNX, converted from PaddleOCR by RapidOCR) | text detection, orientation, recognition | **Apache-2.0**: RapidOCR `README.md` "License / Models" section at commit `f65c7da00e72c19c258245e8e0e5f33af14488be` ("derived from official PaddleOCR models ... distributed under the Apache License, Version 2.0 ... converted artifacts redistributed under the same terms"); `LICENSE` of RapidAI/RapidOCR and of PaddlePaddle/PaddleOCR (both Apache-2.0, `main` on 2026-10-06); wheel `METADATA` `License: Apache-2.0`. The README links a `MODEL_LICENSES.md` that does **not exist** (404) at that commit, so per-file attribution was not available. | PyPI `rapidocr_onnxruntime==1.2.3`; sha256 `3439588c...` (det), `897a3ede...` (rec), `e47acedf...` (cls), full values in `workloads/ppocr-rapidocr/model.sha256` | 13.7 MB | `ppocr-rapidocr`, `ppocr-rapidocr-bench` | cpu reference recorded and re-run PASS |
| MNIST-12 (ONNX model zoo) | digit classifier | **MIT**: "License" section of `validated/vision/classification/mnist/README.md` in onnx/models at commit `c5cae0f1942c8d7727372c7b1f6b485aa39e4421` (the repo's own `LICENSE` is Apache-2.0) | `mnist-12.onnx` sha256 `5c688690...` (equals its git-LFS oid); test tarball `a53a59dc...` | 26 KB | `onnx-zoo-mnist` | cpu reference recorded and re-run PASS; matches the zoo's own expected logits (diff 0) |
| MobileNetV2-12 (ONNX model zoo) | ImageNet classifier | **Apache-2.0**: "License" section of `.../mobilenet/README.md` at the same onnx/models commit, and the repo `LICENSE`. Says nothing about the ImageNet training data's terms. | `mobilenetv2-12.onnx` sha256 `c0c3f76d...` (= git-LFS oid); tarball `5f83b422...` | 14 MB | `onnx-zoo-mobilenetv2`, `onnx-zoo-mobilenetv2-bench` | cpu reference recorded and re-run PASS; matches the zoo's expected logits (diff 0) |
| spaCy `en_core_web_sm` 3.8.0 | tokenizer, POS tagger, parser, NER (English) | **MIT**: `en_core_web_sm-3.8.0.dist-info/LICENSE` (Copyright 2021 ExplosionAI GmbH) and `meta.json` `"license": "MIT"`, inside the wheel. Its `LICENSES_SOURCES` lists training data: OntoNotes 5 ("commercial (licensed by Explosion)"), WordNet 3.0 (own licence, text in the file), ClearNLP guideline (citation only): read those before building on the model. | release `en_core_web_sm-3.8.0` of explosion/spacy-models; wheel sha256 `1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85` | 12.8 MB | `spacy-en-core-web-sm`, `spacy-en-core-web-sm-bench` | cpu reference recorded and re-run PASS |
| YuNet face detection 2023mar | face + 5 landmarks | **MIT**: `models/face_detection_yunet/LICENSE` ("MIT License, Copyright (c) 2020 Shiqi Yu") and the folder README "License" section, opencv/opencv_zoo commit `47534e27c9851bb1128ccc0102f1145e27f23f98` (`main` on 2026-10-07) | `face_detection_yunet_2023mar.onnx` sha256 `8f2383e4...` (= git-LFS oid) | 0.23 MB | `yunet-face-detection`, `-bench` | cpu reference recorded and re-run PASS |
| SFace 2021dec | face embedding (128-d) | **Apache-2.0**: `models/face_recognition_sface/LICENSE` (Apache 2.0 text) and README "License" section, same commit | `face_recognition_sface_2021dec.onnx` sha256 `0ba9fbfa...` | 38.7 MB | `sface-face-embedding`, `-bench` | cpu reference recorded and re-run PASS |
| PP-HumanSeg 2023mar | person segmentation | **Apache-2.0**: `models/human_segmentation_pphumanseg/LICENSE` ("Copyright (c) 2021 PaddlePaddle Authors", Apache 2.0 text) and README, same commit | `human_segmentation_pphumanseg_2023mar.onnx` sha256 `552d8a98...` | 6.2 MB | `pphumanseg-person-segmentation`, `-bench` | cpu reference recorded and re-run PASS |
| NanoDet-Plus-m 416 (2022nov) | COCO object detection | **Apache-2.0**: `models/object_detection_nanodet/LICENSE` and README, same commit | `object_detection_nanodet_2022nov.onnx` sha256 `4b82da99...` | 3.8 MB | `nanodet-object-detection`, `-bench` | cpu reference recorded and re-run PASS |
| YOLOX-S 640 (2022nov) | COCO object detection | **Apache-2.0**: `models/object_detection_yolox/LICENSE` and README, same commit | `object_detection_yolox_2022nov.onnx` sha256 `c5c2d13e...` | 35.9 MB | `yolox-object-detection`, `-bench` | cpu reference recorded and re-run PASS |
| CRNN English (2021sep) | word recognition (CTC) | **Apache-2.0**: `models/text_recognition_crnn/LICENSE` and README, same commit | `text_recognition_CRNN_EN_2021sep.onnx` sha256 `a84b1f6e...` | 33.8 MB | `crnn-text-recognition`, `-bench` | cpu reference recorded and re-run PASS |
| SSD-MobileNetV1-12 (ONNX model zoo) | COCO object detection, NMS in graph | **MIT**: "License" section of `validated/vision/object_detection_segmentation/ssd-mobilenetv1/README.md` at onnx/models commit `c5cae0f1942c8d7727372c7b1f6b485aa39e4421` (the section names the licence, there is no licence text beside the model; the repo `LICENSE` is Apache-2.0) | `ssd_mobilenet_v1_12.onnx` sha256 `b8fba5e4...` (= git-LFS oid) | 29.5 MB | `onnx-zoo-ssd-mobilenetv1`, `-bench` | cpu reference recorded and re-run PASS |
| ShuffleNet-v2-12 (ONNX model zoo) | ImageNet classifier | **BSD-3-Clause**: "License" section of `.../shufflenet/README.md` ("BSD 3-Clause License"), same onnx/models commit; same caveats as above | `shufflenet-v2-12.onnx` sha256 `ea69821b...`; tarball `ec61f826...` | 9.2 MB | `onnx-zoo-shufflenet-v2`, `-bench` | cpu reference recorded and re-run PASS; matches the zoo's expected logits (diff 0) |
| EfficientNet-Lite4-11 (ONNX model zoo) | ImageNet classifier | **MIT**: "License" section of `.../efficientnet-lite4/README.md` ("MIT License"), same onnx/models commit; same caveats | `efficientnet-lite4-11.onnx` sha256 `d1116899...`; tarball `1914de66...` | 51.9 MB | `onnx-zoo-efficientnet-lite4`, `-bench` | cpu reference recorded and re-run PASS; matches the zoo's expected output (diff 4e-9) |

Test inputs of the vision workloads (not models; fetched, never committed): the three scikit-image sample
images `astronaut.png`, `chelsea.png`, `coffee.png` at scikit-image tag v0.22.0 (commit
`441fe68b95a86d4ae2a351311a0c39a4232b6521`, the last releases that still ship them in the repo; sha256 `88431cd9...`,
`596aa1e7...`, `cc02f8ca...`). Their licence statements were read in the docstrings of `skimage/data/_fetchers.py` at
that commit: astronaut "No known copyright restrictions, released into the public domain" (NASA photo of Eileen Collins),
chelsea and coffee "No copyright restrictions. CC0 by the photographer"; the repository itself is BSD-3-Clause. The
CRNN workload draws its words in code (Pillow's bundled default font). The classifiers use the zoo's own test tensors.
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

## Text models (second batch, 2026-10-07)

Same rule: weights and licence text were both fetched. Two further hosts turned out to be reachable and are used:
**the npm registry** (`registry.npmjs.org`, immutable versions, tarball integrity published) and PyPI file
URLs (`files.pythonhosted.org/packages/...`, content-addressed). A third, `storage.googleapis.com`, serves some
public buckets (see "Language-model weights" below). Hugging Face is still blocked.

| Model | What it does | Licence, and where it was read | Pin | Size | Workloads | State |
| --- | --- | --- | --- | --- | --- | --- |
| BiDAF-9 (ONNX model zoo) | extractive question answering (SQuAD v1.1) | **MIT**: SPDX header and "License" section of `validated/text/machine_comprehension/bidirectional_attention_flow/README.md` in onnx/models at commit `c5cae0f1942c8d7727372c7b1f6b485aa39e4421` (the repo `LICENSE` is Apache-2.0). Says nothing about SQuAD's terms. | `bidaf-9.tar.gz` sha256 `c74387ee...` (= git-LFS oid); `bidaf/bidaf.onnx` inside it `dfc317b5...` | 39 MB (onnx 43.5 MB) | `onnx-zoo-bidaf` | cpu reference recorded and re-run PASS; answers all 16 of the zoo's own samples exactly as the zoo expects |
| BERT-Squad-12 int8 (ONNX model zoo) | extractive question answering, BERT-base fine-tuned on SQuAD v1.1, quantized by Intel Neural Compressor | **Apache-2.0**: SPDX header and "License" section of `.../bert-squad/README.md` at the same commit. Says nothing about SQuAD / BERT pretraining data. The vocabulary is **not** in the zoo: it is read from the MiniLM package below (same 30,522-entry uncased vocabulary), and the answers come out right. | `bertsquad-12-int8.tar.gz` sha256 `6c33f44e...` (= LFS oid); `bertsquad-12-int8.onnx` inside it `24325bdf...` | 106 MB (onnx 124.6 MB) | `onnx-zoo-bertsquad-int8`, `onnx-zoo-bertsquad-int8-bench` | cpu reference recorded and re-run PASS; logits within 0.032 of the zoo's stored output |
| all-MiniLM-L6-v2 (fp32 ONNX), **third-party npm repackaging** | sentence embeddings, 384d | **Apache-2.0, second-hand**: `README.md` of the npm package `@xcidos/genesis-memory-model` 0.1.0-alpha.1 ("The bundled model and tokenizer are redistributed from all-MiniLM-L6-v2 under Apache-2.0", pinned Hugging Face revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`) and its `package.json` licence `Apache-2.0`. The upstream model card could **not** be read (Hugging Face blocked), so this is the redistributor's statement. | tarball sha256 `3462d151...` (equals the npm sha512 integrity); `onnx/model.onnx` `6fd5d72f...`, `tokenizer.json` `be50c362...`, `README.md` `af6b01d8...` | 83 MB (onnx 90 MB) | `minilm-l6-v2-onnx`, `minilm-l6-v2-onnx-bench` | cpu reference recorded and re-run PASS; token ids of "the cat sat on the mat." are the known `101 1996 4937 2938 2006 1996 13523 1012 102` |
| GloVe `glove-wiki-gigaword-50` (Stanford, via gensim-data) | 400,000 word vectors, 50d (Wikipedia 2014 + Gigaword 5) | **PDDL-1.0 (public domain dedication)**: `README.md` of stanfordnlp/GloVe at commit `52f7dbc512be3f72e9c11e7cbb19dfdfa4651e9c` ("Pre-trained word vectors are made available under the Public Domain Dedication and License") and `list.json` of RaRe-Technologies/gensim-data at commit `bd93f51f7536dbf84f194cd8a81c0644b7b95b41` (`license: opendatacommons.org/licenses/pddl/`). The training data (Wikipedia is CC BY-SA, Gigaword 5 is LDC-licensed) is not discussed there. | release tag `glove-wiki-gigaword-50`; sha256 `5c55f989...` (md5 equals `list.json`) | 69 MB | `glove-wiki-gigaword-50-knn`, `glove-wiki-gigaword-50-knn-bench` | cpu reference recorded and re-run PASS; king - man + woman gives queen, paris - france + italy gives rome |
| spaCy `en_core_web_md` 3.8.0 | English tagger, parser, NER and 20,000 static 300d vectors | **MIT**: `en_core_web_md-3.8.0.dist-info/LICENSE` (Copyright 2021 ExplosionAI GmbH) and `meta.json` in the wheel. `LICENSES_SOURCES` as for `en_core_web_sm` (OntoNotes 5 licensed by Explosion, WordNet 3.0, ClearNLP citation) plus Explosion vectors (CC0). | release `en_core_web_md-3.8.0`; wheel sha256 `5e6329fe...` | 33 MB | `spacy-en-core-web-md`, `spacy-en-core-web-md-bench` | cpu reference recorded and re-run PASS |
| spaCy `ru_core_news_sm`, `uk_core_news_sm`, `nb_core_news_sm`, `xx_ent_wiki_sm` 3.8.0 | Russian, Ukrainian, Norwegian Bokmal pipelines (tagger, parser, NER) and a 3-label multilingual NER | **MIT** for all four: `<pkg>-3.8.0.dist-info/LICENSE` in each wheel (identical text) and `meta.json`. `LICENSES_SOURCES` read for each: ru Nerus (MIT), uk Ukr-Synth (MIT), nb UD Norwegian Bokmaal + NorNE (both CC0), xx WikiNER (**CC BY 4.0**, attribution required). `license` of every 3.8.0 pipeline read from `meta/*.json` of explosion/spacy-models at commit `ca6f473afda3c4943d3919d3e39406c1b3f48b85` (see the rejected list). | wheel sha256 `69978d47...` (ru), `d20adb50...` (uk), `086f2cfc...` (nb), `6f3c4b85...` (xx) | 15 + 15 + 12 + 11 MB | `spacy-multilingual-sm`, `spacy-multilingual-sm-bench` | cpu reference recorded and re-run PASS |
| py3langid 0.4.0 (langid.py fork), 139-language identifier | language identification (linear model over byte n-grams) | **BSD-3-Clause**: `py3langid-0.4.0.dist-info/licenses/LICENSE` and `METADATA` (`License-Expression: BSD-3-Clause`) in the PyPI wheel. The repo's `TRAINING.md` lists the training text (Wikipedia leads, Leipzig news, Tatoeba, CC-100) whose own terms were **not** read. | wheel sha256 `c25bd038...` | 4.6 MB | `py3langid-wheel` (cpu only) | cpu reference recorded and re-run PASS; 10 of 10 languages right |
| SentencePiece 0.2.2 and its upstream test model `test_model.model` (unigram, 1000 pieces) | subword tokenizer | **Apache-2.0**: `LICENSE` of google/sentencepiece at tag v0.2.2 = commit `e0cce7d37b065b5140349dbe12c6bcf6192fdd78`, and the wheel's `License-Expression: Apache-2.0`. The repo does not say what text the test model was trained on. | model sha256 `4884d27b...`; wheel (cp313, manylinux x86-64) sha256 `64b656f0...` | 0.25 MB + 1.4 MB | `sentencepiece-test-model` (cpu only) | cpu reference recorded and re-run PASS |

All eight functional workloads were also run from an **empty cache** (every file downloaded through
`tools/ort_assets.py`, checked, extracted, the archive deleted) and passed again.

New helper code: `tools/ort_assets.py` gained `fetch_unpacked` (download, check sha256, extract only the listed
members with path-traversal and link checks, delete the archive) and `check_members` (member files pinned as
`<archive>::<path>` lines in `model.sha256` are hashed on every use); `tools/wordpiece.py` is a pure-Python BERT
WordPiece tokenizer (so no tokenizer package is needed); `tools/text_tasks.py` holds the payloads. Unit tests:
`tests/test_text_tools.py`. Wheels (spaCy pipelines, py3langid, sentencepiece) are **unpacked into the cache and put on
`sys.path` / loaded with `spacy.load(path)`**: nothing is pip-installed and no venv was created or changed.

## How they run

- Runtime: onnxruntime 1.30.0 (MIT) for all the ONNX models; spaCy 3.8.16 for the last. `tools/ort-env.sh` /
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

- Vision workloads (`tools/ort_vision.py`): OpenCV's `dnn` module is **not** used; each model's pre- and
  post-processing (YuNet / NanoDet / YOLOX decoding and NMS, SFace alignment, CTC decoding, ...) is re-written
  in numpy from the model's documented usage and runs on the same onnxruntime session as on every target. Checks
  made once by hand against OpenCV while writing them: YuNet finds the same face as `cv2.FaceDetectorYN` (box
  within about 1 px); the SFace alignment matches `cv2.FaceRecognizerSF.alignCrop` (0.0005 grey levels mean
  difference) and the embedding has cosine 1.0 to OpenCV's. Class ids, counts and strings are compared exactly;
  boxes within 2 to 3 px, scores within 0.02 to 0.03, embeddings within 0.05, as written in each manifest. These
  tolerances are reasoned (a GPU provider may use TF32 or another reduction order, ~1e-3 relative), **not measured
  on any GPU**. As a weak substitute the references were also re-run on the CPU with 4 intra-op threads (another
  reduction order) and all nine still PASS.
- Not comparable across targets in the way a real accuracy claim would be: the CRNN workload reads words drawn in
  code (an easy case; with a looser crop "hello" came out as "shellor"), the SFace workload has one face and no
  identification claim, PP-HumanSeg marks about half of the cat photo as "person" (an out-of-domain input; the
  workload records the foreground fraction, not a correctness claim), and NanoDet's weaker duplicate boxes are
  not recorded because their number is not stable.

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

Text-model workloads, in addition: onnxruntime 1.30.0 for BiDAF, BERT-Squad, MiniLM and GloVe (the GloVe search
is a two-node ONNX graph, MatMul then TopK, built in `tools/text_tasks.py`, so it runs on the target's execution
provider); spaCy 3.8.16 for the three spaCy workloads (the Russian and Ukrainian lemmatizers need pymorphy3, which is
not installed, so the lemmatizer is excluded for all); numpy only for py3langid and sentencepiece, which therefore
list **cpu as their only target** (there is no GPU code path, and a `gpu` label would run numpy on the CPU).
Exact comparisons: answer spans, answer texts, token ids, nearest-neighbour words, POS/dependency/entity strings,
languages, SentencePiece ids. Tolerances: embedding and cosine values 2e-3 (MiniLM), GloVe scores 1e-4, spaCy
vector similarities 1e-4, BERT int8 span scores 0.1 + 0.5 %, langid log-scores 0.01 + 1e-4. The int8 BERT model
(DynamicQuantizeLinear, MatMulInteger) and BiDAF's string tensors may have operators with no CUDA kernel that
onnxruntime places on the CPU even under the CUDA provider; the provider check only looks at the first provider,
and none of this was run on a GPU. Benchmarks added (gpu only, **none run on a GPU, `bench/` is empty**):
BERT-Squad questions/s (batch 1, 256 tokens), MiniLM sentences/s (36 sentences, tokenization included), GloVe
queries/s (256 queries against 400k words), spaCy words/s for `en_core_web_md` and for each of the four small
pipelines. Their code paths were exercised on the CPU with the gpu check bypassed (numbers not recorded).

## What was run

On the cpu target (4 shared cores, onnxruntime 1 thread): the five functional workloads (and, for the speech / audio
additions, the six listed above) were run, their
references recorded with `bin/pw record --target cpu` (recorder: the CPU backend of onnxruntime 1.30.0 /
spaCy 3.8.16, Python 3.13.16, numpy 2.5.3) and re-run to PASS. On the gpu target here every workload is a clean
SKIP (no NVIDIA GPU). The nine vision workloads of 2026-10-07 were recorded the same way (recorder: onnxruntime 1.30.0 CPU
provider, 1 intra-op thread, opencv-python 5.0.0.93, Pillow 12.3.0) and re-run to PASS; their `-bench` workloads
(batch 1 for the fixed-batch exports) only had their code paths exercised on the CPU with the provider check
bypassed (numbers not recorded); `bench/` is still empty. Unit tests: `tests/test_ort_tools.py`.

The eight text-model functional workloads were recorded the same way, on the CPU only, with the same runtimes
(onnxruntime 1.30.0, spaCy 3.8.16, numpy 2.5.3, Python 3.13.16; py3langid 0.4.0, sentencepiece 0.2.2) and re-run to
PASS from an empty cache. `--target gpu` is a clean SKIP for all of them; nothing was run on a GPU or a simulated
GPU. Unit tests: `tests/test_ort_tools.py` (updated for archive members and npm / PyPI URLs) and
`tests/test_text_tools.py`.

## Investigated and not added

| Candidate | Finding |
| --- | --- |
| `silero-vad` PyPI package itself | needs PyTorch for its Python API; its ONNX file is what is used directly. |
| `rapidocr` 3.9.2 (PP-OCRv6 small, 31 MB wheel) | newer, same Apache-2.0 statement; not added to keep one OCR model. The v1.2.3 wheel was used because it is small and pinned. |
| `onnx` wheel bundled test data | 476 files, but node-test graphs of single operators, not pretrained models. |
| `mediapipe` 1.1.0 wheel | LICENSE (Apache-2.0) and NOTICE are there, but no `.tflite`/`.task` model files are bundled (they are downloaded from Google storage). **Correction, 2026-10-07:** `storage.googleapis.com/mediapipe-models/` *is* reachable (see the text-model entries below), so "blocked" was true of the other buckets tried, not of this one. |
| `vosk` wheel / models | the wheel has only the runtime; models live on `alphacephei.com`, blocked (CONNECT 403). |
| `piper` | the binary release downloads from GitHub, but voices are on Hugging Face (blocked). |
| `en_core_web_sm` from PyPI | not on PyPI (index has no such distribution); the GitHub release asset is used instead. |
| other onnx/models entries | superseded by the 2026-10-07 rows above. READMEs read at `c5cae0f1...`: licence sections of ResNet (Apache 2.0, resnet18-v1-7 46.8 MB and resnet50-v1-12 102.6 MB), SqueezeNet (Apache 2.0, squeezenet1.1-7 5.0 MB), DenseNet-121 (MIT, 32.7 MB), UltraFace (MIT, 1.3 MB), Tiny-YOLOv2 (MIT, 63.5 MB), FCN-ResNet50 (MIT, 141 MB), SSD (Apache 2.0, 80 MB), RetinaNet (BSD-3, 228 MB), YOLOv4 (MIT, 257 MB), ArcFace (Apache 2.0, 261 MB), VGG (Apache 2.0, 553 MB) all name a permissive licence. Not added: the ones over 100 MB or ~140 MB on size (FCN, RetinaNet, YOLOv4, ArcFace, VGG, ResNet-50) and the rest only to keep this set small (same mechanism, would add nothing new in kind; ResNet-18, SqueezeNet, DenseNet, UltraFace, Tiny-YOLOv2 are the obvious next ones). No onnx/models README read said NC, GPL or research-only. |
| onnx/models via `raw.githubusercontent.com` | serves a 130-byte LFS pointer, not weights; `media.githubusercontent.com/media/...` serves the real file (sha256 equals the pointer's oid). |
| opencv_zoo `facial_expression_recognition` (MobileNet-based FER) | its README says "All files in this directory are licensed under Apache 2.0 License (./LICENSE)", but **there is no LICENSE file in that folder** at `47534e27...` (every other model folder has one). The licence text could not be read, so it was not added. |
| opencv_zoo `deblurring_nafnet` (MIT, 91.7 MB), `inpainting_lama` (Apache-2.0, 92.6 MB) | licence files read and permissive; not added: size close to the 100 MB limit and disk is tight here (no other reason). |
| opencv_zoo `image_classification_ppresnet` (Apache-2.0, 102.6 MB), `person_reid_youtureid` (Apache-2.0, 106.9 MB) | licence files read; over the 100 MB size limit. |
| opencv_zoo `image_segmentation_efficientsam` (Apache-2.0, 48 MB), `object_tracking_vittrack` (Apache-2.0, 0.7 MB), `optical_flow_estimation_raft` (BSD-3-Clause for RAFT, MIT for the PINTO0309 conversion, 64 MB) | licence files read and permissive; not added: they need prompts, a video track or a frame pair, and no licensed input of that kind was available (the sample images in `example_outputs/` are annotated outputs, not inputs, and `benchmark/data` is fetched from Google Drive, blocked). |
| opencv_zoo `handpose_estimation_mediapipe`, `palm_detection_mediapipe`, `pose_estimation_mediapipe`, `person_detection_mediapipe` (Apache-2.0, 4 to 12 MB) | licence files read and permissive; not added: they are stages of a cascade (detector, then an ROI-cropped landmark model) and no licensed hand or full-body image was reachable to feed them honestly. Pose was the one kind in the brief left uncovered for this reason. |
| opencv_zoo `license_plate_detection_yunet` (Apache-2.0), `face_image_quality_assessment_ediffiqa` (**CC-BY-4.0**, 7 MB), `edge_detection_dexined` (MIT, 47 MB), `image_classification_mobilenet` (Apache-2.0), `text_detection_ppocr` (Apache-2.0), `qrcode_wechatqrcode` (Apache-2.0, Caffe models) | licence files read and permissive; not added: no licensed licence-plate image; eDifFIQA needs aligned faces and a quality reference; DexiNed and PP-OCR detection were left out for scope (OCR is covered by `ppocr-rapidocr` and `crnn-text-recognition`, MobileNetV2 by `onnx-zoo-mobilenetv2`); WeChat QR models are Caffe, not ONNX. |
| opencv_zoo `.int8` / `.int8bq` / `fp16` variants | not added: quantised kernels differ between execution providers, so cross-target comparison would need a different tolerance story. |
| Hugging Face, download.pytorch.org (torchvision weights) | not reachable from here (see the top of this page), so torchvision, timm and Ultralytics models were not considered. |

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

### Investigated and not added (text models, 2026-10-07)

| Candidate | Finding |
| --- | --- |
| onnx/models `gpt2-10`, `gpt2-lm-head-10`, `gpt2-lm-head-bs-12` (GPT-2), `roberta-base-11`, `roberta-sequence-classification-9`, `t5-encoder-12`, `t5-decoder-with-lm-head-12`, `bertsquad-8/10/12` (fp32) | READMEs at the pinned commit say Apache 2.0 (read), and the files are reachable via `media.githubusercontent.com`, but they are **439-665 MB each** (BERT-Squad fp32 436 MB, RoBERTa 499 MB): over the 150 MB per model target and the disk budget here. These are the best candidates if a bigger budget is acceptable. |
| onnx/models `bidaf-11-int8` (12 MB) | MIT, same README; the same model as the BiDAF-9 workload quantized, not added to keep one. |
| onnx/models `Natural_Language_Processing/*_transformers` and `Generative_AI/*_transformers` (TurnkeyML exports: electra 54 MB, esm 30 MB, gptneox 1.8 MB, distilbert 265 MB, ...) | no per-model README or licence: only `turnkey_stats.yaml` (author: transformers) and the repo-wide Apache-2.0 `LICENSE`; the root README says they were exported from timm / torchvision / transformers, whose models carry their own, differing licences. Whether the weights are pretrained is not stated. Not added (a README silent on the licence). |
| spaCy 3.8.0 pipelines with non-permissive `license` in `meta/*.json` (explosion/spacy-models `ca6f473a...`) | GNU GPL 3.0: `ca_`, `es_`, `pl_core_news_*`, `es_dep_news_trf`; LGPL-LR: `fr_core_news_*`, `fr_dep_news_trf`; CC BY-NC-SA 3.0: `el_`, `it_core_news_*`; CC BY-SA 4.0 (share-alike, scope for a model unclear): `da_`, `fi_`, `hr_`, `ja_`, `ko_`, `lt_`, `mk_`, `nl_`, `pt_`, `ro_`, `sl_`, `sv_core_news_*`, `ja_core_news_trf` (3.0); CC BY-SA 3.0: `xx_sent_ud_sm`. The wheel `LICENSE` and `LICENSES_SOURCES` were read only for the pipelines that were added (and `en_core_web_sm` earlier); the rest are rejected on `meta.json` and its `sources` list. |
| spaCy `de_core_news_sm` (MIT), `zh_core_web_sm` (MIT), `nb_core_news_md`, `ru_core_news_md`, `uk_core_news_md` | permissive by `meta.json` (de: TIGER "commercial, licensed by Explosion" + WikiNER CC BY; zh: OntoNotes "licensed by Explosion") but wheels not downloaded or read: not added to keep the set small. `en_core_web_lg` and the `*_trf` pipelines are far larger (not downloaded). |
| `lingua-language-detector` 2.2.0 | the x86-64 wheel is 170 MB, over the size limit (licence not read). |
| `pocketsphinx` 5.1.1 (29 MB wheel with an en-us model) | a speech recogniser; its licence text and model files were not read in this pass. |
| gensim-data: `glove-wiki-gigaword-100/200/300`, `glove-twitter-25/50/100/200`, `word2vec-ruscorpora-300`, `fasttext-wiki-news-subwords-300`, `conceptnet-numberbatch-17-06-300`, `word2vec-google-news-300` | `list.json` licences: PDDL for the GloVe ones (134 MB to 795 MB, `glove-twitter-25` is 109 MB), CC BY 4.0 for ruscorpora (208 MB), CC BY-SA 3.0 for fastText (1.0 GB), a licence file link for ConceptNet (1.2 GB), `not found` for word2vec-google-news-300 (1.7 GB). The 50d wiki-gigaword vectors were the smallest permissive one. |
| MediaPipe text models on `storage.googleapis.com/mediapipe-models/` (`language_detector` 0.3 MB, `average_word_classifier` 0.8 MB, `universal_sentence_encoder` 6 MB, `bert_classifier` 26 MB, `bert_embedder` 26 MB; TFLite) | reachable and small, with object generations to pin, but **no licence text for the models could be found**: the MediaPipe repo (Apache-2.0) and the samples' READMEs are silent about them, and the model-card PDFs in the `mediapipe-assets` bucket (prefix `Model`) are all for vision models. Also needs a TFLite runtime. Not added. |
| `@ryanstark24/sfgraph-models` 1.1.3 (npm, quantized MiniLM ONNX, 23 MB) | package says MIT but its README and `MODEL_INFO.md` never state the model's own licence (they point at the blocked Hugging Face page). Not added. |
| `@energetic-ai/model-embeddings-en`, `@cup319/mmpl-model` (bge-base-zh int8, 103 MB), `@ternlight/base` / `@ternlight/mini` (ternary distilled encoder compiled into a wasm), `@mailwoman/neural-weights-en-us` (**AGPL-3.0-only OR commercial**) | not read beyond npm metadata, or not an ONNX/GGUF model the runtimes here can run (ternlight: weights live inside a wasm), or AGPL (excluded by rule). |

### Language-model weights (LLMs): what is reachable

Probed on 2026-10-07. **Hugging Face, hf-mirror, ModelScope, Zenodo, archive.org and `dl.fbaipublicfiles.com` could not
be connected to (curl status 000).** No small GGUF or safetensors language model was found on GitHub, but GitHub was
not enumerated: `github.com/<o>/<r>/tree/...` pages and the REST API are refused, so only repositories whose file lists
could be read through `git` were examined (onnx/models, gensim-data, spacy-models, sentencepiece, GloVe, py3langid,
the genesis repo; a partial clone of `microsoft/onnxruntime-genai`, whose test models would be randomly initialised
anyway, was abandoned as too slow). Two hosts do carry real LLM weights:

- **Google Cloud Storage, `mediapipe-models` bucket** (public, listable at
  `https://storage.googleapis.com/storage/v1/b/mediapipe-models/o?prefix=text_summarizer/`): `text_summarizer/200m/.../summarization_quant_200m_2modes.litertlm`
  (117.6 MB), `text_proofreader/200m/.../proofread_quant_200m.litertlm` (117.6 MB) and
  `text_embedder/embedding_gemma/int4int8/.../embedding_gemma.task` (183.8 MB). The path names say Gemma. They are
  **not added**: no licence text sits next to them, Gemma-family weights are, to my knowledge, under Google's custom
  Gemma Terms of Use and Prohibited Use Policy (not read here, and not one of the permissive licences this repo
  accepts), and the LiteRT-LM / `.task` formats need runtimes that are not installed. They are in the size range and
  would be the first thing to try if the licence is read and accepted.
- **npm**: `slaunt-model-tinyllama-1b-part1` / `-part2` (0.1.0, each 161 MB unpacked; "Slaunt Audit's bundled TinyLlama
  1.1B GGUF model" split into parts, `license: Apache-2.0` in `package.json`). TinyLlama 1.1B is a real pretrained LLM, but
  the whole model is several hundred MB across at least two parts (the other parts were not listed or downloaded), over
  the disk budget, and the packages' contents and the weights' licence text were **not read**. Other npm packages that
  matched ("GGUF" in the description) are runtimes or downloaders (`@joelrd01/llama-cpp-qwen2.5-0.5b` fetches the model
  from Hugging Face on install and carries no weights).
- **conda-forge** and **PyPI**: `conda.anaconda.org` answers (the `repodata.json.zst` request returned 200) but conda-forge
  was **not searched** for weights packages. On PyPI only sizes were looked at (`gpt4all` and `llama-cpp-python` list no
  suitable wheel, `ctransformers` 9.9 MB, `sentence-transformers` 0.7 MB): runtimes by size alone, contents not inspected.
  Neither search was exhaustive.

So: **no small LLM weights with a readable permissive licence were found.** The only LLM-like weights reachable under 150 MB
are the Gemma-family files above, and the smallest permissive-looking LLM (TinyLlama via npm) is too large and unread.
