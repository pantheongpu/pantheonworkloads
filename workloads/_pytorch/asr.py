"""Shared body of the catalog's speech-recognition workloads, imported by their main.py.

The input is the 11 s clip jfk.wav of whisper.cpp (pinned commit, sha256 checked, fetched at run time, never stored
here: its provenance is stated nowhere, see docs/models.md). The output is the transcript, lower case letters and
spaces only, compared by word error rate (manifest `tolerance.fields.text.wer`); the run also fails when the word
error rate against the known words of the speech exceeds PW_ASR_WER_MAX (default 0.15), whatever a reference says.
Engines:
  transformers-pipeline   transformers.pipeline("automatic-speech-recognition") on the pinned snapshot
  qwen3-asr               transformers' native Qwen3ASRForConditionalGeneration on Qwen's own checkpoint, which is in the layout of Qwen's own
                          `qwen-asr` package (transformers 4.57): config.json nests everything under `thinker_config`, the audio encoder's model_type is
                          `qwen3_asr_audio_encoder` (native: `qwen3_asr_encoder`) and the weights are named `thinker.*`. Loaded here with that config and a
                          key mapping (without them every weight is UNEXPECTED and the model is random: measured on an A10G). Prompt: the processor's chat
                          template with the audio, greedy generate(); the output "language English<asr_text>..." is parsed by the processor. The pipeline
                          cannot drive this model (beam search by default, audio features passed as input_ids).
  nemo                    NVIDIA NeMo: ASRModel.restore_from(<.nemo file of the snapshot>).transcribe([wav])
Bench mode (PW_ASR_MODE=bench): seconds of audio transcribed per second of compute. Run on an A10G (stage 2a).
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
    elif engine == "qwen3-asr":
        import json
        from transformers import AutoProcessor, Qwen3ASRConfig, Qwen3ASRForConditionalGeneration
        processor = AutoProcessor.from_pretrained(path)
        thinker = json.load(open(os.path.join(path, "config.json")))["thinker_config"]
        thinker["audio_config"]["model_type"] = "qwen3_asr_encoder"
        key_mapping = {r"^thinker\.model\.": "model.language_model.", r"^thinker\.audio_tower\.proj1\.": "model.multi_modal_projector.linear_1.",
                       r"^thinker\.audio_tower\.proj2\.": "model.multi_modal_projector.linear_2.", r"^thinker\.audio_tower\.": "model.audio_tower.",
                       r"^thinker\.lm_head\.": "lm_head."}
        model = Qwen3ASRForConditionalGeneration.from_pretrained(path, config=Qwen3ASRConfig(**thinker), dtype=getattr(torch, dtype), key_mapping=key_mapping,
                                                                 device_map=os.environ.get("PW_DEVICE_MAP") or "auto").eval()

        def go():
            messages = [{"role": "user", "content": [{"type": "audio", "audio": audio}]}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, model.dtype)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False, num_beams=1)
            return processor.decode(out[0, inputs["input_ids"].shape[1]:], return_format="transcription_only")
    elif engine == "nemo":
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            common.skip(f"NeMo is not installed ({e}); pip install -r workloads/_pytorch/requirements-nemo.txt")
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
