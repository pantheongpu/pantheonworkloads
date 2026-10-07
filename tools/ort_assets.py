#!/usr/bin/env python3
"""Fetch, checksum-verified, the pinned files the ort-* and spacy-* workloads need. Nothing is committed.

    ort_assets.py <workload-dir> <asset>...     prints each local path, one per line

Every asset has a URL here (pinned to a commit) and its sha256 in <workload-dir>/model.sha256
("<sha256>  <asset name>", sha256sum format). Exit 77 when a file cannot be had here (offline,
blocked host) or does not match its checksum. PW_CACHE: cache directory (default
~/.cache/pantheonworkloads). Run it with `python3 -I` (it only needs the standard library).
"""
import hashlib
import os
import pathlib
import subprocess
import sys

SKIP = 77

# Commits the URLs are pinned to (see docs/pretrained-reachable.md for what was read at each).
SILERO = "5cd7945676eb32225748052e2e6a0580e4686a08"      # snakers4/silero-vad tag v6.2.3
ONNX_MODELS = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"  # onnx/models main on 2026-10-06
RAPIDOCR = "f65c7da00e72c19c258245e8e0e5f33af14488be"    # RapidAI/RapidOCR main on 2026-10-06
OPENCV_ZOO = "47534e27c9851bb1128ccc0102f1145e27f23f98"  # opencv/opencv_zoo main on 2026-10-07
SKIMAGE = "441fe68b95a86d4ae2a351311a0c39a4232b6521"     # scikit-image/scikit-image tag v0.22.0 (still ships the sample images)
_RAW = "https://raw.githubusercontent.com"
_LFS = "https://media.githubusercontent.com/media"   # git-LFS objects (raw.* serves only the pointer)
_ZOO = "validated/vision/classification"

URLS = {
    "silero_vad.onnx": f"{_RAW}/snakers4/silero-vad/{SILERO}/src/silero_vad/data/silero_vad.onnx",
    "silero-test.wav": f"{_RAW}/snakers4/silero-vad/{SILERO}/tests/data/test.wav",
    "mnist-12.onnx": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mnist/model/mnist-12.onnx",
    "mnist-12.tar.gz": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mnist/model/mnist-12.tar.gz",
    "mobilenetv2-12.onnx": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mobilenet/model/mobilenetv2-12.onnx",
    "mobilenetv2-12.tar.gz": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mobilenet/model/mobilenetv2-12.tar.gz",
    "en_core_web_sm-3.8.0-py3-none-any.whl": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    "ocr-en.jpg": f"{_RAW}/RapidAI/RapidOCR/{RAPIDOCR}/python/tests/test_files/en.jpg",
}
# OpenCV Zoo weights are git-LFS objects; each model folder has its own LICENSE at that commit.
for _name, _dir in {
    "face_detection_yunet_2023mar.onnx": "face_detection_yunet",
    "face_recognition_sface_2021dec.onnx": "face_recognition_sface",
    "human_segmentation_pphumanseg_2023mar.onnx": "human_segmentation_pphumanseg",
    "object_detection_nanodet_2022nov.onnx": "object_detection_nanodet",
    "object_detection_yolox_2022nov.onnx": "object_detection_yolox",
    "text_recognition_CRNN_EN_2021sep.onnx": "text_recognition_crnn",
}.items():
    URLS[_name] = f"{_LFS}/opencv/opencv_zoo/{OPENCV_ZOO}/models/{_dir}/{_name}"
# ONNX model zoo (same commit as MNIST / MobileNetV2 above)
_DET = "validated/vision/object_detection_segmentation"
for _name, _path in {
    "shufflenet-v2-12.onnx": f"{_ZOO}/shufflenet/model/shufflenet-v2-12.onnx",
    "shufflenet-v2-12.tar.gz": f"{_ZOO}/shufflenet/model/shufflenet-v2-12.tar.gz",
    "efficientnet-lite4-11.onnx": f"{_ZOO}/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
    "efficientnet-lite4-11.tar.gz": f"{_ZOO}/efficientnet-lite4/model/efficientnet-lite4-11.tar.gz",
    "ssd_mobilenet_v1_12.onnx": f"{_DET}/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx",
}.items():
    URLS[_name] = f"{_LFS}/onnx/models/{ONNX_MODELS}/{_path}"
# Sample images (public domain / CC0 per skimage/data/_fetchers.py at that tag)
for _name in ("astronaut", "chelsea", "coffee"):
    URLS[f"{_name}.png"] = f"{_RAW}/scikit-image/scikit-image/{SKIMAGE}/skimage/data/{_name}.png"


def parse_sums(text):
    """sha256sum format -> {name: sha256}. Blank lines and # comments are ignored."""
    sums = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            sums[name.strip().lstrip("*")] = digest.lower()
    return sums


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, want, cache):
    """Return the cached path of `name` after downloading it (curl, so the proxy settings apply)
    and checking it against `want`; raise RuntimeError with a reason otherwise."""
    dest = pathlib.Path(cache) / want[:16] / name
    if not dest.exists():
        if name not in URLS:
            raise RuntimeError(f"no URL known for {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        p = subprocess.run(["curl", "-fsSL", "--retry", "2", "-m", "600", "-o", str(part), URLS[name]],
                           capture_output=True, text=True)
        if p.returncode != 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"cannot download {URLS[name]}: {p.stderr.strip()[:120]}")
        part.rename(dest)
    got = sha256(dest)
    if got != want:
        raise RuntimeError(f"{dest} has sha256 {got}, pinned {want}; delete it to re-download")
    return dest


def main(argv):
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    sums = parse_sums((pathlib.Path(argv[1]) / "model.sha256").read_text())
    cache = pathlib.Path(os.environ.get("PW_CACHE") or pathlib.Path.home() / ".cache" / "pantheonworkloads") / "ort-assets"
    for name in argv[2:]:
        if name not in sums:
            print(f"{name} has no checksum in {argv[1]}/model.sha256", file=sys.stderr)
            return 2
        try:
            print(fetch(name, sums[name], cache))
        except RuntimeError as e:
            print(f"SKIP: {e}")
            return SKIP
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
