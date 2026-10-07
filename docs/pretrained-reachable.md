# Pretrained models whose weights and licence could be read

Earlier workloads had to mark model licences UNVERIFIED because model hubs (Hugging Face and others) are blocked
where they were written. The models below were chosen the other way round: only where **both the weights and
the licence text were fetched and read** from hosts that are reachable (PyPI, `raw.githubusercontent.com`,
`media.githubusercontent.com` for git-LFS objects, GitHub release downloads). Weights are never committed:
each workload's `model.sha256` pins them and `tools/ort_assets.py` downloads and checks them at run time.

Written 2026-10-06; the text-model rows (second table) were added 2026-10-07. The licences were read on those dates at the pins below.

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

On the cpu target (4 shared cores, onnxruntime 1 thread): the five functional workloads were run, their
references recorded with `bin/pw record --target cpu` (recorder: the CPU backend of onnxruntime 1.30.0 /
spaCy 3.8.16, Python 3.13.16, numpy 2.5.3) and re-run to PASS. On the gpu target here every workload is a clean
SKIP (no NVIDIA GPU). Unit tests: `tests/test_ort_tools.py`.

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
| other onnx/models entries (ResNet-50 etc.) | same licence mechanism as MobileNetV2 and reachable; not added for size (98 MB) and because per-model README licence lines would each need reading. |
| onnx/models via `raw.githubusercontent.com` | serves a 130-byte LFS pointer, not weights; `media.githubusercontent.com/media/...` serves the real file (sha256 equals the pointer's oid). |

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
