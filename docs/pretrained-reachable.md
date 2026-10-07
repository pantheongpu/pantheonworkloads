# Pretrained models whose weights and licence could be read

Earlier workloads had to mark model licences UNVERIFIED because model hubs (Hugging Face and others) are blocked
where they were written. The models below were chosen the other way round: only where **both the weights and
the licence text were fetched and read** from hosts that are reachable (PyPI, `raw.githubusercontent.com`,
`media.githubusercontent.com` for git-LFS objects, GitHub release downloads). Weights are never committed:
each workload's `model.sha256` pins them and `tools/ort_assets.py` downloads and checks them at run time.

Written 2026-10-06; the vision section below was added on 2026-10-07. The licences were read on those dates at the pins below.

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

Inputs that are not models: the Silero clip (`tests/data/test.wav`, 60 s) and the RapidOCR screenshot
(`python/tests/test_files/en.jpg`) come from those MIT / Apache-2.0 repositories at the pinned commits, but
**neither repository states the recording's or the screenshot's own provenance**, so they are fetched and
never committed. The MNIST digit and MobileNet tensor come from the zoo's own test tarballs. The spaCy
sentences were written for this repo.

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

## What was run

On the cpu target (4 shared cores, onnxruntime 1 thread): the five functional workloads were run, their
references recorded with `bin/pw record --target cpu` (recorder: the CPU backend of onnxruntime 1.30.0 /
spaCy 3.8.16, Python 3.13.16, numpy 2.5.3) and re-run to PASS. On the gpu target here every workload is a clean
SKIP (no NVIDIA GPU). The nine vision workloads of 2026-10-07 were recorded the same way (recorder: onnxruntime 1.30.0 CPU
provider, 1 intra-op thread, opencv-python 5.0.0.93, Pillow 12.3.0) and re-run to PASS; their `-bench` workloads
(batch 1 for the fixed-batch exports) only had their code paths exercised on the CPU with the provider check
bypassed (numbers not recorded); `bench/` is still empty. Unit tests: `tests/test_ort_tools.py`.

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
