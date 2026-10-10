"""Shared body of the catalog's text-to-speech workloads, imported by their main.py.

One fixed sentence is synthesised and the waveform is checked, never compared sample by sample: output = the clip's
duration rounded to a half second, its RMS level (3 decimals) and whether every sample is finite. The run fails when
the clip is shorter than a second, silent or clipped. Engines:
  chatterbox   the `chatterbox-tts` package: ChatterboxTTS.from_local(<pinned snapshot>, device).generate(text)
  f5-tts       the `f5-tts` package: F5TTS(ckpt_file=..., vocab_file=...).infer(ref_file, ref_text, gen_text) with the
               pinned jfk.wav clip as the reference voice
Both sample, so the generator is seeded (torch.manual_seed) but the result is only expected to be stable in these
coarse numbers. Bench mode (PW_TTS_MODE=bench): seconds of audio produced per second of compute. Run on an A10G (stage 2a).
"""
import glob
import os
import sys
import time

import numpy as np
import torch

import catalog_common as cc
import common

TEXT = "The quick brown fox jumps over the lazy dog."


def run(model_id, revision, engine, label=None):
    mode = os.environ.get("PW_TTS_MODE", "functional")
    if common.DEVICE == "cpu":
        common.skip("this speech model is run on a GPU only")
    common.setup()
    path = cc.fetch_snapshot(model_id, revision)
    t0 = time.perf_counter()
    if engine == "chatterbox":
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as e:
            common.skip(f"chatterbox-tts is not installed ({e}); pip install -r workloads/_pytorch/requirements-chatterbox.txt")
        model = ChatterboxTTS.from_local(path, common.DEVICE)
        sr = model.sr

        def go():
            torch.manual_seed(0)
            return model.generate(TEXT).squeeze().float().cpu().numpy()
    elif engine == "f5-tts":
        try:
            from f5_tts.api import F5TTS
        except ImportError as e:
            common.skip(f"f5-tts is not installed ({e}); pip install -r workloads/_pytorch/requirements-f5.txt")
        _, ref = cc.load_clip()
        ckpts = sorted(glob.glob(os.path.join(path, "F5TTS_v1_Base", "model_*.safetensors")) + glob.glob(os.path.join(path, "F5TTS_Base", "model_*.safetensors")))
        vocabs = sorted(glob.glob(os.path.join(path, "F5TTS_v1_Base", "vocab.txt")) + glob.glob(os.path.join(path, "F5TTS_Base", "vocab.txt")))
        if not ckpts:
            sys.exit(f"no F5-TTS checkpoint found under {path}")
        own = os.path.join(os.path.dirname(ckpts[0]), "vocab.txt")   # the checkpoint's own directory first (the two vocabularies are identical, measured)
        model = F5TTS(ckpt_file=ckpts[0], vocab_file=own if os.path.exists(own) else (vocabs[0] if vocabs else ""), device=common.DEVICE)
        sr = 24000

        def go():
            wav, _sr, _spec = model.infer(ref_file=ref, ref_text="And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
                                          gen_text=TEXT, seed=0)
            return np.asarray(wav, dtype=np.float32)
    else:
        sys.exit(f"unknown engine {engine}")
    load_s = time.perf_counter() - t0
    wav = go()
    dur = len(wav) / sr
    if not np.isfinite(wav).all():
        sys.exit("non-finite samples")
    rms = float(np.sqrt(np.mean(np.square(wav))))
    if dur < 1.0 or rms < 1e-3 or float(np.abs(wav).max()) > 1.5:
        sys.exit(f"implausible clip: {dur:.2f} s, rms {rms:.4f}, peak {float(np.abs(wav).max()):.2f}")
    if os.environ.get("PW_OUT"):
        import wave
        with wave.open(os.path.join(os.environ["PW_OUT"], "tts.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
    out = {"duration_s": round(dur * 2) / 2, "rms": round(rms, 3), "finite": True}
    metrics = {}
    if mode == "bench":
        go()
        common.sync()
        t = time.perf_counter()
        n = 3
        total = 0.0
        for _ in range(n):
            total += len(go()) / sr
        common.sync()
        metrics = {"audio_seconds_per_s": round(total / (time.perf_counter() - t), 2)}
        out = f"{label or model_id} bench ok"
    common.finish(out, f"{common.device_name()}, torch {torch.__version__}, engine {engine}, load {load_s:.0f} s, {dur:.2f} s of audio, rms {rms:.3f}", metrics)
