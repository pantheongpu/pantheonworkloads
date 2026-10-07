#!/usr/bin/env python3
"""Fetch, checksum-verified, the pinned files the ort-* and spacy-* workloads need. Nothing is committed.

    ort_assets.py <workload-dir> <asset>...     prints each local path, one per line

Every asset has a URL here (pinned to a commit) and its sha256 in <workload-dir>/model.sha256
("<sha256>  <asset name>", sha256sum format). Exit 77 when a file cannot be had here (offline,
blocked host) or does not match its checksum. PW_CACHE: cache directory (default
~/.cache/pantheonworkloads). Run it with `python3 -I` (it only needs the standard library).

Some assets are members of a release archive (MEMBERS below: asset name -> archive, path inside it).
Such a member is extracted from the archive once, checked against its own sha256 and cached on its
own; the archive (which must also be pinned in model.sha256, by its sha256) is deleted after
extraction to save disk, and downloaded again only if a member is missing.
"""
import hashlib
import os
import pathlib
import subprocess
import sys
import tarfile

SKIP = 77

# Commits the URLs are pinned to (see docs/pretrained-reachable.md for what was read at each).
SILERO = "5cd7945676eb32225748052e2e6a0580e4686a08"      # snakers4/silero-vad tag v6.2.3
ONNX_MODELS = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"  # onnx/models main on 2026-10-06
RAPIDOCR = "f65c7da00e72c19c258245e8e0e5f33af14488be"    # RapidAI/RapidOCR main on 2026-10-06
_RAW = "https://raw.githubusercontent.com"
_LFS = "https://media.githubusercontent.com/media"   # git-LFS objects (raw.* serves only the pointer)
_ZOO = "validated/vision/classification"
_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
GTCRN = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"       # Xiaobin-Rong/gtcrn main on 2026-10-07

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
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2": f"{_SHERPA}/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2",
    # Xiaobin-Rong/gtcrn
    "gtcrn_simple.onnx": f"{_RAW}/Xiaobin-Rong/gtcrn/{GTCRN}/stream/onnx_models/gtcrn_simple.onnx",
    "gtcrn-mix.wav": f"{_RAW}/Xiaobin-Rong/gtcrn/{GTCRN}/test_wavs/mix.wav",
    "gtcrn-enh.wav": f"{_RAW}/Xiaobin-Rong/gtcrn/{GTCRN}/test_wavs/enh.wav",
}

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
