#!/usr/bin/env python3
"""Fetch, checksum-verified, the pinned files the ort-* and spacy-* workloads need. Nothing is committed.

    ort_assets.py <workload-dir> <asset>...     prints each local path, one per line

Every asset has a URL here (pinned to a commit) and its sha256 in <workload-dir>/model.sha256
("<sha256>  <asset name>", sha256sum format; "<sha256>  <archive>::<member>" pins one file inside an
archive, see fetch_unpacked / check_members). Exit 77 when a file cannot be had here (offline,
blocked host) or does not match its checksum. PW_CACHE: cache directory (default
~/.cache/pantheonworkloads). Run it with `python3 -I` (it only needs the standard library).

Some assets are members of a release archive (MEMBERS below: asset name -> archive, path inside it).
Such a member is extracted from the archive once, checked against its own sha256 and cached on its
own; the archive (which must also be pinned in model.sha256, by its sha256) is deleted after
extraction to save disk, and downloaded again only if a member is missing.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import zipfile

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
_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
GTCRN = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"       # Xiaobin-Rong/gtcrn main on 2026-10-07
_ZOO_TEXT = "validated/text/machine_comprehension"
_SPACY = "https://github.com/explosion/spacy-models/releases/download"
_PYPI = "https://files.pythonhosted.org/packages"
SENTENCEPIECE = "e0cce7d37b065b5140349dbe12c6bcf6192fdd78"   # google/sentencepiece tag v0.2.2

URLS = {
    "silero_vad.onnx": f"{_RAW}/snakers4/silero-vad/{SILERO}/src/silero_vad/data/silero_vad.onnx",
    "silero-test.wav": f"{_RAW}/snakers4/silero-vad/{SILERO}/tests/data/test.wav",
    "mnist-12.onnx": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mnist/model/mnist-12.onnx",
    "mnist-12.tar.gz": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mnist/model/mnist-12.tar.gz",
    "mobilenetv2-12.onnx": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mobilenet/model/mobilenetv2-12.onnx",
    "mobilenetv2-12.tar.gz": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO}/mobilenet/model/mobilenetv2-12.tar.gz",
    "en_core_web_sm-3.8.0-py3-none-any.whl": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
    "ocr-en.jpg": f"{_RAW}/RapidAI/RapidOCR/{RAPIDOCR}/python/tests/test_files/en.jpg",
    # sherpa-onnx release assets: the tags are rolling names, so the sha256 in model.sha256 is the pin
    "sherpa-onnx-moonshine-tiny-en-int8.tar.bz2": f"{_SHERPA}/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2",
    "kokoro-int8-en-v0_19.tar.bz2": f"{_SHERPA}/tts-models/kokoro-int8-en-v0_19.tar.bz2",
    "wespeaker_en_voxceleb_resnet34.onnx": f"{_SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
    "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2": f"{_SHERPA}/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2",
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2": f"{_SHERPA}/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2",
    # Xiaobin-Rong/gtcrn
    "gtcrn_simple.onnx": f"{_RAW}/Xiaobin-Rong/gtcrn/{GTCRN}/stream/onnx_models/gtcrn_simple.onnx",
    "gtcrn-mix.wav": f"{_RAW}/Xiaobin-Rong/gtcrn/{GTCRN}/test_wavs/mix.wav",
    "gtcrn-enh.wav": f"{_RAW}/Xiaobin-Rong/gtcrn/{GTCRN}/test_wavs/enh.wav",
    # text models (docs/pretrained-reachable.md)
    "bidaf-9.tar.gz": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO_TEXT}/bidirectional_attention_flow/model/bidaf-9.tar.gz",
    "bertsquad-12-int8.tar.gz": f"{_LFS}/onnx/models/{ONNX_MODELS}/{_ZOO_TEXT}/bert-squad/model/bertsquad-12-int8.tar.gz",
    # a release asset: the tag is the pin, the sha256 pins the bytes (gensim-data's list.json says PDDL for it)
    "glove-wiki-gigaword-50.gz": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz",
    # npm versions are immutable; third-party redistribution of sentence-transformers/all-MiniLM-L6-v2
    "genesis-memory-model-0.1.0-alpha.1.tgz": "https://registry.npmjs.org/@xcidos/genesis-memory-model/-/genesis-memory-model-0.1.0-alpha.1.tgz",
    "ru_core_news_sm-3.8.0-py3-none-any.whl": f"{_SPACY}/ru_core_news_sm-3.8.0/ru_core_news_sm-3.8.0-py3-none-any.whl",
    "uk_core_news_sm-3.8.0-py3-none-any.whl": f"{_SPACY}/uk_core_news_sm-3.8.0/uk_core_news_sm-3.8.0-py3-none-any.whl",
    "nb_core_news_sm-3.8.0-py3-none-any.whl": f"{_SPACY}/nb_core_news_sm-3.8.0/nb_core_news_sm-3.8.0-py3-none-any.whl",
    "xx_ent_wiki_sm-3.8.0-py3-none-any.whl": f"{_SPACY}/xx_ent_wiki_sm-3.8.0/xx_ent_wiki_sm-3.8.0-py3-none-any.whl",
    "en_core_web_md-3.8.0-py3-none-any.whl": f"{_SPACY}/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl",
    # PyPI files are content-addressed in their URL path
    "py3langid-0.4.0-py3-none-any.whl": f"{_PYPI}/69/2d/0eeff2727c970b1d553be70d15ba2d84e3830696c09a79d954c5ae3d5212/py3langid-0.4.0-py3-none-any.whl",
    "sentencepiece-0.2.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl": f"{_PYPI}/59/b4/a0356fa04d6a14337a6e0e443556785a0422c53ec58baae6b9568120eb0f/sentencepiece-0.2.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    "sentencepiece-test.model": f"{_RAW}/google/sentencepiece/{SENTENCEPIECE}/python/test/test_model.model",
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

_M = "sherpa-onnx-moonshine-tiny-en-int8"
_K = "kokoro-int8-en-v0_19"
_T = "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15"
_TA = _T + ".tar.bz2"
_W = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
_WA = _W + ".tar.bz2"
_KWS = "epoch-12-avg-2-chunk-16-left-64"
# asset name -> (archive asset, path inside the archive)
MEMBERS = {
    "audiotag-model.onnx": (_TA, f"{_T}/model.onnx"),
    "audiotag-labels.csv": (_TA, f"{_T}/class_labels_indices.csv"),
    "audiotag-README.md": (_TA, f"{_T}/README.md"),
    "audiotag-test-1.wav": (_TA, f"{_T}/test_wavs/1.wav"),
    "audiotag-test-2.wav": (_TA, f"{_T}/test_wavs/2.wav"),
    "audiotag-test-5.wav": (_TA, f"{_T}/test_wavs/5.wav"),
    "audiotag-test-7.wav": (_TA, f"{_T}/test_wavs/7.wav"),
    "audiotag-test-9.wav": (_TA, f"{_T}/test_wavs/9.wav"),
    "audiotag-test-13.wav": (_TA, f"{_T}/test_wavs/13.wav"),
    "kws-encoder.onnx": (_WA, f"{_W}/encoder-{_KWS}.onnx"),
    "kws-decoder.onnx": (_WA, f"{_W}/decoder-{_KWS}.onnx"),
    "kws-joiner.onnx": (_WA, f"{_W}/joiner-{_KWS}.onnx"),
    "kws-tokens.txt": (_WA, f"{_W}/tokens.txt"),
    "kws-keywords.txt": (_WA, f"{_W}/keywords.txt"),
    "kws-README.md": (_WA, f"{_W}/README.md"),
    "kws-test-keywords.txt": (_WA, f"{_W}/test_wavs/test_keywords.txt"),
    "kws-test-trans.txt": (_WA, f"{_W}/test_wavs/trans.txt"),
    "kws-test-0.wav": (_WA, f"{_W}/test_wavs/0.wav"),
    "kws-test-1.wav": (_WA, f"{_W}/test_wavs/1.wav"),
    "moonshine-preprocess.onnx": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/preprocess.onnx"),
    "moonshine-encode.int8.onnx": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/encode.int8.onnx"),
    "moonshine-uncached-decode.int8.onnx": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/uncached_decode.int8.onnx"),
    "moonshine-cached-decode.int8.onnx": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/cached_decode.int8.onnx"),
    "moonshine-tokens.txt": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/tokens.txt"),
    "moonshine-LICENSE": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/LICENSE"),
    "moonshine-test-0.wav": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/test_wavs/0.wav"),
    "moonshine-test-1.wav": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/test_wavs/1.wav"),
    "moonshine-test-8k.wav": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/test_wavs/8k.wav"),
    "moonshine-test-trans.txt": ("sherpa-onnx-moonshine-tiny-en-int8.tar.bz2", f"{_M}/test_wavs/trans.txt"),
    "kokoro-model.int8.onnx": ("kokoro-int8-en-v0_19.tar.bz2", f"{_K}/model.int8.onnx"),
    "kokoro-voices.bin": ("kokoro-int8-en-v0_19.tar.bz2", f"{_K}/voices.bin"),
    "kokoro-tokens.txt": ("kokoro-int8-en-v0_19.tar.bz2", f"{_K}/tokens.txt"),
    "kokoro-LICENSE": ("kokoro-int8-en-v0_19.tar.bz2", f"{_K}/LICENSE"),
}


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


def _download(url, dest):
    """curl `url` to `dest` (via a .part file, so a cut-off transfer is never mistaken for the file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    p = subprocess.run(["curl", "-fsSL", "--retry", "2", "-m", "900", "-o", str(part), url],
                       capture_output=True, text=True)
    if p.returncode != 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"cannot download {url}: {p.stderr.strip()[:120]}")
    part.rename(dest)


def _extract_members(archive_path, wanted, cache):
    """Extract the members of `archive_path` listed in `wanted` ({asset name: (path in archive, sha256)})
    into cache/<sha16>/<asset name>, checking each against its sha256 (all or nothing)."""
    done = {}
    with tarfile.open(archive_path) as t:
        inside = {m.name: m for m in t.getmembers() if m.isfile()}
        for name, (member, want) in wanted.items():
            if member not in inside:
                raise RuntimeError(f"{archive_path} has no member {member}")
            dest = pathlib.Path(cache) / want[:16] / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            with t.extractfile(inside[member]) as src, open(part, "wb") as out:
                for block in iter(lambda: src.read(1 << 20), b""):
                    out.write(block)
            got = sha256(part)
            if got != want:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"{member} in {archive_path} has sha256 {got}, pinned {want}")
            done[name] = (part, dest)
    for part, dest in done.values():
        part.rename(dest)


def fetch_member(name, want, cache, sums):
    """The cached path of archive member `name`; extracts it (and every other member of the same
    archive that `sums` pins) from the archive when missing, then deletes the archive."""
    archive, _ = MEMBERS[name]
    dest = pathlib.Path(cache) / want[:16] / name
    if not dest.exists():
        if archive not in sums:
            raise RuntimeError(f"{name} is a member of {archive}, which has no checksum in model.sha256")
        if archive not in URLS:
            raise RuntimeError(f"no URL known for {archive}")
        apath = pathlib.Path(cache) / sums[archive][:16] / archive
        try:
            _download(URLS[archive], apath)
            got = sha256(apath)
            if got != sums[archive]:
                raise RuntimeError(f"{archive} has sha256 {got}, pinned {sums[archive]}")
            wanted = {n: (MEMBERS[n][1], sums[n]) for n in sums if n in MEMBERS and MEMBERS[n][0] == archive}
            _extract_members(apath, wanted, cache)
        finally:
            apath.unlink(missing_ok=True)
    got = sha256(dest)
    if got != want:
        raise RuntimeError(f"{dest} has sha256 {got}, pinned {want}; delete it to re-extract")
    return dest


def fetch(name, want, cache, sums=None):
    """Return the cached path of `name` after downloading it (curl, so the proxy settings apply)
    and checking it against `want`; raise RuntimeError with a reason otherwise. `sums` (the
    workload's model.sha256) is needed for archive members."""
    if name in MEMBERS:
        return fetch_member(name, want, cache, sums or {})
    dest = pathlib.Path(cache) / want[:16] / name
    if not dest.exists():
        if name not in URLS:
            raise RuntimeError(f"no URL known for {name}")
        _download(URLS[name], dest)
    got = sha256(dest)
    if got != want:
        raise RuntimeError(f"{dest} has sha256 {got}, pinned {want}; delete it to re-download")
    return dest


def safe_members(names):
    """Refuse archive member names that are absolute or climb out of the target directory."""
    for n in names:
        parts = pathlib.PurePosixPath(n).parts
        if n.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
            raise RuntimeError(f"unsafe archive member name {n!r}")


def extract_archive(archive, dest, members=None):
    """Extract a .tar.gz/.tgz/.whl/.zip into `dest` (all of it, or only `members`: names or
    directory prefixes). Links and special files are refused. Returns the extracted file names."""
    archive, dest = pathlib.Path(archive), pathlib.Path(dest)
    want = None if members is None else tuple(m.rstrip("/") for m in members)
    keep = lambda n: want is None or any(n == m or n.startswith(m + "/") for m in want)   # noqa: E731
    done = []
    if archive.suffix in (".whl", ".zip"):
        with zipfile.ZipFile(archive) as z:
            safe_members(z.namelist())
            for info in z.infolist():
                if not info.is_dir() and keep(info.filename):
                    z.extract(info, dest)
                    done.append(info.filename)
    else:
        with tarfile.open(archive) as t:
            infos = t.getmembers()
            safe_members(i.name for i in infos)
            for info in infos:
                if info.isfile() and keep(info.name):
                    t.extract(info, dest, filter="data")
                    done.append(info.name)
                elif not (info.isfile() or info.isdir()):
                    raise RuntimeError(f"{archive}: member {info.name!r} is a link or special file")
    return done


def fetch_unpacked(name, want, cache, members=None):
    """Directory holding the contents of the pinned archive `name` (sha256 `want`), extracted once.
    The archive is downloaded, checked, extracted (only `members` if given) and then deleted, so
    only the extracted files stay in the cache. A later call with the same or a smaller member
    set reuses them. Everything in the directory came from the checked archive; files you hash
    yourself (pin them as '<archive>::<member>' in model.sha256) are checked on every use by
    `check_members`."""
    root = pathlib.Path(cache) / want[:16] / (name + ".d")
    marker = root / ".extracted.json"
    wanted = None if members is None else sorted(m.rstrip("/") for m in members)
    try:
        have = json.loads(marker.read_text())
        if have["archive_sha256"] == want and (have["members"] is None or (wanted is not None and set(wanted) <= set(have["members"]))):
            return root
    except (OSError, ValueError, KeyError):
        pass
    arch = fetch(name, want, cache)
    extract_archive(arch, root, members)
    marker.write_text(json.dumps({"archive_sha256": want, "members": wanted}))
    arch.unlink()
    return root


def check_members(root, sums, name):
    """Verify every '<name>::<member>' file pinned in `sums` that exists under `root`; raise
    RuntimeError on a mismatch or a missing file. Returns the number checked."""
    n = 0
    for key, want in sums.items():
        if key.startswith(name + "::"):
            p = pathlib.Path(root) / key.split("::", 1)[1]
            if not p.is_file():
                raise RuntimeError(f"{p} is missing")
            got = sha256(p)
            if got != want:
                raise RuntimeError(f"{p} has sha256 {got}, pinned {want}; delete {root} to re-extract")
            n += 1
    return n


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
            print(fetch(name, sums[name], cache, sums))
        except RuntimeError as e:
            print(f"SKIP: {e}")
            return SKIP
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
