"""Shared body of the catalog's speech-recognition workloads, imported by their main.py.

The input is the 11 s clip jfk.wav of whisper.cpp (pinned commit, sha256 checked, fetched at run time, never stored
here: its provenance is stated nowhere, see docs/models.md). The output is the transcript, lower case letters and
spaces only, compared by word error rate (manifest `tolerance.fields.text.wer`); the run also fails when the word
error rate against the known words of the speech exceeds PW_ASR_WER_MAX (default 0.15), whatever a reference says.
Engines:
  transformers-pipeline   transformers.pipeline("automatic-speech-recognition") on the pinned snapshot
  nemo                    NVIDIA NeMo: ASRModel.restore_from(<.nemo file of the snapshot>).transcribe([wav])
Bench mode (PW_ASR_MODE=bench): seconds of audio transcribed per second of compute. Never run on a GPU yet.
"""
import glob
import os
import sys
import time

import torch

import catalog_common as cc
import common


def run(model_id, revision, engine, label=None, dtype="float16", language=None):
    mode = os.environ.get("PW_ASR_MODE", "functional")
    if common.DEVICE == "cpu":
        common.skip("this speech model is run on a GPU only")
    common.setup()
    path = cc.fetch_snapshot(model_id, revision)
    audio, wav_path = cc.load_clip()
    seconds = len(audio) / 16000.0
    t0 = time.perf_counter()
    if engine == "transformers-pipeline":
        from transformers import pipeline
        pipe = pipeline("automatic-speech-recognition", model=path, dtype=getattr(torch, dtype), device_map=os.environ.get("PW_DEVICE_MAP") or "auto")

        def go():
            return pipe({"raw": audio, "sampling_rate": 16000})["text"]
    elif engine == "nemo":
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            common.skip(f"NeMo is not installed ({e}); pip install -r workloads/_pytorch/requirements-speech.txt")
        files = sorted(glob.glob(os.path.join(path, "*.nemo")))
        if not files:
            sys.exit(f"no .nemo file in {path}")
        model = nemo_asr.models.ASRModel.restore_from(files[0], map_location=common.DEVICE).eval()

        def go():
            res = model.transcribe([wav_path])
            first = res[0] if isinstance(res, (list, tuple)) else res
            return getattr(first, "text", first) if not isinstance(first, str) else first
    else:
        sys.exit(f"unknown engine {engine}")
    load_s = time.perf_counter() - t0
    raw = go()
    text = cc.normalise_text(raw if isinstance(raw, str) else str(raw))
    wer = cc.word_error_rate(text, cc.CLIP_TEXT)
    limit = float(os.environ.get("PW_ASR_WER_MAX", "0.15"))
    if wer > limit:
        sys.exit(f"word error rate {wer:.3f} against the known speech exceeds {limit}: {text!r}")
    out = {"text": text}
    metrics = {}
    if mode == "bench":
        for _ in range(2):
            go()
        common.sync()
        t = time.perf_counter()
        n = 5
        for _ in range(n):
            go()
        common.sync()
        metrics = {"audio_seconds_per_s": round(n * seconds / (time.perf_counter() - t), 2)}
        out = f"{label or model_id} bench ok"
    common.finish(out, f"{common.device_name()}, torch {torch.__version__}, engine {engine}, load {load_s:.0f} s, wer {wer:.3f}, text {text!r}", metrics)
