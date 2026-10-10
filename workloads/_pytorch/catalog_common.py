"""Helpers shared by the model-catalog workloads that load a Hugging Face snapshot (imported beside common.py and pinned.py)."""
import os
import sys
import wave

import numpy as np

import common
import pinned

CLIP_URL = ("https://raw.githubusercontent.com/ggml-org/whisper.cpp/d1be6fde11ac6e0407606b4e42fe72d34add8037/samples/jfk.wav")
CLIP_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
CLIP_TEXT = ("and so my fellow americans ask not what your country can do for you ask what you can do for your country")
SMALL_FILES = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "*.yaml", "*.py", "tokenizer*/*", "*.vocab", "*.bpe"]


def cache_root():
    return os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads"))


def fetch_snapshot(model_id, revision, extra=()):
    """Download the pinned files of the workload (what model.sha256 lists, plus small configs), verify every sha256
    (a mismatch fails the run; PW_HF_REVISION skips the pin), return the local directory."""
    from huggingface_hub import snapshot_download
    rev = os.environ.get("PW_HF_REVISION") or revision
    sums_path = os.path.join(os.environ["PW_WORKLOAD_DIR"], "model.sha256")
    sums = pinned.read_sums(sums_path)
    allow = [*SMALL_FILES, *extra, *sums]
    path = common.hf_load(lambda m, r, c: snapshot_download(m, revision=r, cache_dir=c, allow_patterns=allow), model_id, rev)
    if not os.environ.get("PW_HF_REVISION"):
        pinned.verify_snapshot(path, sums_path)
    return path


def load_clip():
    """(float32 mono samples at 16 kHz, wav path): the 11 s jfk.wav of whisper.cpp at the pinned commit, sha256 checked."""
    wav_path = common.hf_load(lambda m, r, c: pinned.fetch_asset(
        CLIP_URL, CLIP_SHA256, os.path.join(cache_root(), "assets", "jfk.wav")), "jfk.wav", None)
    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit("jfk.wav is not 16 kHz mono 16-bit")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return data, wav_path


def normalise_text(text):
    """Lower case, letters and spaces only: what the speech workloads compare."""
    import re
    return " ".join(re.sub(r"[^a-z ]+", " ", text.lower()).split())


def word_error_rate(hyp, ref):
    h, r = hyp.split(), ref.split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))
