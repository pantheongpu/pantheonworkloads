#!/usr/bin/env python3
"""Speech / audio payloads of the ONNX workloads: pretrained models run directly on onnxruntime
(no sherpa-onnx at run time, so the gpu target uses the same execution-provider check as the other
ort-* workloads and never silently runs on the CPU).

    python3 -I tools/speech_tasks.py <task> [--bench]

Tasks: asr (Moonshine tiny English, int8), speaker (WeSpeaker ResNet34 speaker embedding),
tts (Kokoro v0.19 int8, phoneme input), enhance (GTCRN speech enhancement), kws (streaming Zipformer2
keyword spotter), tag (Zipformer AudioSet tagging).
Environment and output contract: tools/ort_tasks.py and docs/workload-contract.md.

The feature extraction / decoding here is written for this repo (numpy only) after reading the
reference implementations (sherpa-onnx csrc, GTCRN stream/gtcrn_stream.py); on the development machine
each task was checked against sherpa-onnx 1.13.8's own output (see docs/pretrained-reachable.md).
"""
import importlib.util
import os
import pathlib
import sys
import time
import wave

HERE = pathlib.Path(__file__).resolve().parent


def _ort_tasks():
    spec = importlib.util.spec_from_file_location("ort_tasks", HERE / "ort_tasks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ot = _ort_tasks()
Skip = ot.Skip


# ---------------------------------------------------------------- shared audio helpers

def read_wav(path):
    """Mono 16-bit PCM wav -> (float32 samples in [-1, 1), sample rate)."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth()) != (1, 2):
            raise ValueError("expected mono 16-bit PCM")
        rate = w.getframerate()
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0, rate


def word_error_rate(ref, hyp):
    """Word-level edit distance / number of reference words (case, punctuation and hyphens ignored)."""
    import re
    norm = lambda t: re.sub(r"[^a-z0-9' ]", " ", t.lower())
    r, h = norm(ref).split(), norm(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


# ---------------------------------------------------------------- Moonshine tiny (ASR)

def moonshine_text(token_ids, tokens):
    """Token ids -> text. tokens.txt is a SentencePiece vocabulary: '▁' marks a word start and
    '<0xNN>' is one byte of a UTF-8 sequence."""
    out = bytearray()
    for i in token_ids:
        piece = tokens[i]
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            out.append(int(piece[3:5], 16))
        else:
            out += piece.replace("▁", " ").encode("utf-8")
    return out.decode("utf-8", errors="replace").strip()


def read_tokens(path):
    """'<piece><TAB><id>' lines -> list indexed by id."""
    table = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            piece, _, idx = line.rpartition("\t" if "\t" in line else " ")
            table[int(idx)] = piece
    return [table[i] for i in range(len(table))]


class Moonshine:
    """Moonshine v1 tiny, the four ONNX pieces of the sherpa-onnx export: preprocess (raw audio ->
    features), encode, uncached_decode (first token) and cached_decode (key/value caches carried along).
    Greedy search, at most 6 tokens per second of audio, BOS 1 / EOS 2 (as sherpa-onnx does)."""

    def __init__(self, paths):
        pre, enc, unc, cac = paths
        self.pre, self.prov = ot.session(pre)
        self.enc, _ = ot.session(enc)
        self.unc, _ = ot.session(unc)
        self.cac, _ = ot.session(cac)

    def transcribe(self, audio):
        import numpy as np
        feats = self.pre.run(None, {self.pre.get_inputs()[0].name: audio[None, :].astype("float32")})[0]
        flen = np.array([feats.shape[1]], dtype="int32")
        enc = self.enc.run(None, {self.enc.get_inputs()[0].name: feats, self.enc.get_inputs()[1].name: flen})[0]
        max_len = int(enc.shape[1] * 384 / 16000.0 * 6)
        tokens, seq_len = [], 1
        ui = [i.name for i in self.unc.get_inputs()]
        out = self.unc.run(None, {ui[0]: np.array([[1]], dtype="int32"), ui[1]: enc, ui[2]: np.array([seq_len], dtype="int32")})
        logits, states = out[0], out[1:]
        ci = [i.name for i in self.cac.get_inputs()]
        for _ in range(max_len):
            tok = int(np.argmax(logits[0, -1]))
            if tok == 2:
                break
            tokens.append(tok)
            seq_len += 1
            feed = {ci[0]: np.array([[tok]], dtype="int32"), ci[1]: enc, ci[2]: np.array([seq_len], dtype="int32")}
            feed.update(dict(zip(ci[3:], states)))
            out = self.cac.run(None, feed)
            logits, states = out[0], out[1:]
        return tokens


def task_asr(bench):
    names = ["moonshine-preprocess.onnx", "moonshine-encode.int8.onnx", "moonshine-uncached-decode.int8.onnx",
             "moonshine-cached-decode.int8.onnx"]
    paths = ot.assets(*names)
    tok_path, trans_path, w0, w1 = ot.assets("moonshine-tokens.txt", "moonshine-test-trans.txt", "moonshine-test-0.wav", "moonshine-test-1.wav")
    tokens = read_tokens(tok_path)
    refs = dict(line.split(" ", 1) for line in pathlib.Path(trans_path).read_text().splitlines() if line.strip())
    model = Moonshine(paths)
    texts, ids, wers, seconds = [], [], [], 0.0
    clips = []
    for wav, key in ((w0, "0.wav"), (w1, "1.wav")):
        audio, rate = read_wav(wav)
        if rate != 16000:
            raise ValueError(f"{wav}: expected 16 kHz")
        clips.append(audio)
        seconds += len(audio) / 16000
        t = model.transcribe(audio)
        text = moonshine_text(t, tokens)
        ids.append(" ".join(map(str, t)))
        texts.append(text)
        wers.append(word_error_rate(refs[key].lower(), text.lower()))
    wer = sum(wers) / len(wers)
    wer_max = float(os.environ.get("PW_ASR_WER_MAX", "0.15"))
    if wer > wer_max:
        raise RuntimeError(f"word error rate {wer:.3f} against the archive's own transcripts exceeds {wer_max}")
    output = {"text": "|".join(texts), "token_ids": "|".join(ids)}
    detail = f"{len(texts)} clips, {seconds:.1f} s of audio, WER {wer:.3f} vs the archive's transcripts, Moonshine tiny int8, {model.prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "3"))):
            t0 = time.perf_counter()
            for a in clips:
                model.transcribe(a)
            times.append(time.perf_counter() - t0)
        metrics = {"realtime_factor": seconds / ot.med(times), "latency_ms_per_clip": ot.med(times) * 1000 / len(clips)}
    return output, detail, metrics


# ---------------------------------------------------------------- Kaldi-style fbank (for the speaker model)

def window(n, kind):
    """Kaldi window of n samples: 'povey' (Hann ** 0.85) or 'hamming'."""
    import numpy as np
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    if kind == "povey":
        return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))) ** 0.85
    raise ValueError(kind)


def mel_banks(num_bins, n_fft, rate, low, high):
    """Kaldi triangular mel filters, shape (num_bins, n_fft // 2); the Nyquist bin is not used."""
    import numpy as np
    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)
    nyq = rate / 2
    if high <= 0:
        high += nyq
    fft_bins = n_fft // 2
    delta = (mel(high) - mel(low)) / (num_bins + 1)
    freqs = np.arange(fft_bins) * (rate / n_fft)
    mf = mel(freqs)
    banks = np.zeros((num_bins, fft_bins))
    for b in range(num_bins):
        left, centre, right = mel(low) + b * delta, mel(low) + (b + 1) * delta, mel(low) + (b + 2) * delta
        up = (mf - left) / (centre - left)
        down = (right - mf) / (right - centre)
        banks[b] = np.maximum(0.0, np.minimum(up, down))
    return banks


def kaldi_fbank(samples, rate=16000, num_bins=80, low=20.0, high=-400.0, frame_ms=25.0, shift_ms=10.0,
                preemph=0.97, snip_edges=False, window_type="povey"):
    """Kaldi / kaldi-native-fbank log-mel features of float samples already scaled to 16-bit range:
    window (povey or hamming), DC removal and pre-emphasis per frame, power spectrum, no dither, no energy.
    Returns (frames, num_bins) float32. snip_edges=False pads by reflection (sherpa-onnx's default),
    True drops the partial frames (torchaudio.compliance.kaldi.fbank's default)."""
    import numpy as np
    flen, fshift = int(rate * frame_ms / 1000), int(rate * shift_ms / 1000)
    n = len(samples)
    if snip_edges:
        nf = 0 if n < flen else 1 + (n - flen) // fshift
        starts = np.arange(nf) * fshift
    else:
        nf = (n + fshift // 2) // fshift
        starts = np.arange(nf) * fshift + fshift // 2 - flen // 2
    idx = starts[:, None] + np.arange(flen)[None, :]
    if not snip_edges:   # reflect about the edges, kaldi style: -1 -> 0, n -> n - 1
        idx = np.where(idx < 0, -idx - 1, idx)
        idx = np.where(idx >= n, 2 * n - 1 - idx, idx)
    frames = np.asarray(samples, dtype="float64")[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = (frames - preemph * prev) * window(flen, window_type)
    n_fft = 1 << (flen - 1).bit_length()
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel = power[:, :n_fft // 2] @ mel_banks(num_bins, n_fft, rate, low, high).T
    return np.log(np.maximum(mel, np.finfo("float32").eps)).astype("float32")


# ---------------------------------------------------------------- WeSpeaker ResNet34 (speaker embedding)

def speaker_embedding(sess, audio):
    """256-d embedding of a 16 kHz clip, preprocessed as WeSpeaker's own wespeaker/bin/infer_onnx.py does
    (commit 9fecd6c): Kaldi fbank of samples x 32768, 80 mel bins, 25/10 ms, Hamming window, edge frames
    dropped (snip_edges), 20 Hz to Nyquist, no dither, then the mean over time subtracted (CMN)."""
    feats = kaldi_fbank(audio * 32768.0, high=0.0, snip_edges=True, window_type="hamming")
    feats = feats - feats.mean(axis=0, keepdims=True)
    return sess.run(None, {sess.get_inputs()[0].name: feats[None]})[0][0]


def cosine(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def speaker_clips(silero_wav, mix_wav):
    """Three clips: two disjoint 10 s windows of the Silero test recording and the first 9.7 s of the GTCRN
    noisy mixture (a different recording)."""
    a, ra = read_wav(silero_wav)
    b, rb = read_wav(mix_wav)
    if ra != 16000 or rb != 16000:
        raise ValueError("expected 16 kHz inputs")
    return [a[0:160000], a[240000:400000], b]


def task_speaker(bench):
    import numpy as np
    model, silero, mix = ot.assets("wespeaker_en_voxceleb_resnet34.onnx", "silero-test.wav", "gtcrn-mix.wav")
    sess, prov = ot.session(model)
    clips = speaker_clips(silero, mix)
    embs = [speaker_embedding(sess, c) for c in clips]
    cos = [cosine(embs[i], embs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    output = {"cosine_a_a2_b": [round(c, 4) for c in cos],
              "norms": [round(float(np.linalg.norm(e)), 3) for e in embs],
              "embeddings": [[round(float(x), 3) for x in e] for e in embs]}
    detail = f"{len(embs)} clips -> 256-d embeddings; cosine same-recording {cos[0]:.3f}, cross-recording {cos[1]:.3f}/{cos[2]:.3f}, {prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "5"))):
            t0 = time.perf_counter()
            for c in clips:
                speaker_embedding(sess, c)
            times.append(time.perf_counter() - t0)
        secs = sum(len(c) for c in clips) / 16000
        metrics = {"realtime_factor": secs / ot.med(times), "embeddings_per_s": len(clips) / ot.med(times)}
    return output, detail, metrics


# ---------------------------------------------------------------- Kokoro v0.19 int8 (TTS)

# Phoneme strings in the style espeak-ng writes for en-us, typed in by hand (not produced by espeak: the
# workload needs no phonemizer). Moonshine tiny transcribes the audio back to the intended sentences.
KOKORO_SENTENCES = [
    "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˈoʊvɚ ðə lˈeɪzi dˈɔːɡ.",   # "The quick brown fox jumps over the lazy dog."
    "hɛlˈoʊ wˈɜːld ðɪs ɪz ɐ tˈɛst.",                           # "Hello world, this is a test."
]
KOKORO_SPEAKER = 0      # "af" in this release's voices.bin
KOKORO_RATE = 24000


def read_phoneme_tokens(path):
    """tokens.txt of the Kokoro export: '<symbol> <id>' per line; the symbol is one Unicode character."""
    table = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip("\n"):
            sym, _, idx = line.rpartition(" ")
            table[sym if sym else " "] = int(idx)
    return table


def kokoro_ids(phonemes, table):
    """Token ids as sherpa-onnx feeds them: id 0 before and after the phoneme characters."""
    missing = sorted({c for c in phonemes if c not in table})
    if missing:
        raise ValueError(f"no token for {missing}")
    return [0] + [table[c] for c in phonemes] + [0]


def envelope(wave_, rate, win_s=0.1):
    """RMS of consecutive 100 ms windows."""
    import numpy as np
    w = int(rate * win_s)
    return [float(np.sqrt(np.mean(wave_[i:i + w] ** 2))) for i in range(0, len(wave_), w)]


def audio_stats(wave_, rate):
    """Run-to-run stable summary of a waveform: length, RMS, spectral centroid, 100 ms RMS envelope."""
    import numpy as np
    n = len(wave_)
    spec = np.abs(np.fft.rfft(wave_ * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / rate)
    return {"n_samples": n, "rms": round(float(np.sqrt(np.mean(wave_ ** 2))), 5),
            "centroid_hz": round(float((spec * freqs).sum() / spec.sum()), 1),
            "envelope": [round(e, 4) for e in envelope(wave_, rate)]}


def task_tts(bench):
    import numpy as np
    model, voices, toks = ot.assets("kokoro-model.int8.onnx", "kokoro-voices.bin", "kokoro-tokens.txt")
    table = read_phoneme_tokens(toks)
    styles = np.fromfile(voices, dtype="float32").reshape(11, 511, 256)
    sess, prov = ot.session(model)

    def speak(phonemes):
        ids = kokoro_ids(phonemes, table)
        style = styles[KOKORO_SPEAKER, len(ids) - 2][None]    # one style row per phoneme count
        (audio,) = sess.run(None, {"tokens": np.array([ids], dtype="int64"), "style": style,
                                   "speed": np.array([1.0], dtype="float32")})
        return audio.astype("float32")

    audios = [speak(s) for s in KOKORO_SENTENCES]
    stats = [audio_stats(a, KOKORO_RATE) for a in audios]
    output = {k: [st[k] for st in stats] for k in ("n_samples", "rms", "centroid_hz", "envelope")}
    secs = sum(len(a) for a in audios) / KOKORO_RATE
    detail = f"{len(audios)} sentences, {secs:.1f} s of 24 kHz audio, Kokoro v0.19 int8 voice 'af', {prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "3"))):
            t0 = time.perf_counter()
            for s in KOKORO_SENTENCES:
                speak(s)
            times.append(time.perf_counter() - t0)
        metrics = {"realtime_factor": secs / ot.med(times), "latency_ms_per_sentence": ot.med(times) * 1000 / len(KOKORO_SENTENCES)}
    return output, detail, metrics


# ---------------------------------------------------------------- GTCRN (speech enhancement)

N_FFT, HOP = 512, 256


def sqrt_hann(n=N_FFT):
    import numpy as np
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))   # periodic, as torch.hann_window(n).pow(0.5)


def stft(x):
    """torch.stft(x, 512, 256, 512, sqrt-hann, center=True, pad_mode='reflect') as (257, frames, 2) float32."""
    import numpy as np
    pad = N_FFT // 2
    xp = np.pad(np.asarray(x, dtype="float64"), pad, mode="reflect")
    nf = 1 + (len(xp) - N_FFT) // HOP
    frames = xp[np.arange(nf)[:, None] * HOP + np.arange(N_FFT)[None, :]] * sqrt_hann()
    spec = np.fft.rfft(frames, axis=1).T
    return np.stack([spec.real, spec.imag], axis=-1).astype("float32")


def istft(spec, length):
    """Inverse of stft(): overlap-add with window-square normalisation, centre padding removed, `length` samples."""
    import numpy as np
    w = sqrt_hann()
    c = spec[..., 0].astype("float64") + 1j * spec[..., 1].astype("float64")
    frames = np.fft.irfft(c.T, n=N_FFT, axis=1) * w
    nf = frames.shape[0]
    out = np.zeros(HOP * (nf - 1) + N_FFT)
    norm = np.zeros_like(out)
    for i in range(nf):
        out[i * HOP:i * HOP + N_FFT] += frames[i]
        norm[i * HOP:i * HOP + N_FFT] += w ** 2
    out = out / np.maximum(norm, 1e-11)
    return out[N_FFT // 2:N_FFT // 2 + length].astype("float32")


def gtcrn_enhance(sess, x):
    """Run the streaming GTCRN one STFT frame at a time (conv / TRA / inter-frame caches carried along)."""
    import numpy as np
    spec = stft(x)                                   # (257, T, 2)
    conv = np.zeros([2, 1, 16, 16, 33], dtype="float32")
    tra = np.zeros([2, 3, 1, 1, 16], dtype="float32")
    inter = np.zeros([2, 1, 33, 16], dtype="float32")
    out = []
    for t in range(spec.shape[1]):
        frame = spec[None, :, t:t + 1, :]            # (1, 257, 1, 2)
        o, conv, tra, inter = sess.run(None, {"mix": frame, "conv_cache": conv, "tra_cache": tra, "inter_cache": inter})
        out.append(o[0])
    return istft(np.concatenate(out, axis=1), len(x))


def task_enhance(bench):
    import numpy as np
    model, mix, enh = ot.assets("gtcrn_simple.onnx", "gtcrn-mix.wav", "gtcrn-enh.wav")
    x, rate = read_wav(mix)
    ref, _ = read_wav(enh)
    if rate != 16000:
        raise ValueError("expected 16 kHz")
    sess, prov = ot.session(model)
    y = gtcrn_enhance(sess, x)
    n = min(len(y), len(ref))
    err = float(np.abs(y[:n] - ref[:n]).max())
    if err > 0.005:   # the repository's own offline (PyTorch) output, stored as 16-bit
        raise RuntimeError(f"differs from the GTCRN repository's enh.wav by up to {err:.4f}")
    output = {"n_samples": len(y), "rms_in": round(float(np.sqrt(np.mean(x ** 2))), 5),
              "rms_out": round(float(np.sqrt(np.mean(y ** 2))), 5),
              "envelope_out": [round(e, 4) for e in envelope(y, 16000, 0.5)]}
    detail = f"{len(y) / 16000:.1f} s enhanced, max abs diff to the repo's enh.wav {err:.4f}, GTCRN (streaming ONNX), {prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "3"))):
            t0 = time.perf_counter()
            gtcrn_enhance(sess, x)
            times.append(time.perf_counter() - t0)
        nframes = 1 + len(x) // HOP
        metrics = {"realtime_factor": len(x) / 16000 / ot.med(times), "frames_per_s": nframes / ot.med(times)}
    return output, detail, metrics


# ---------------------------------------------------------------- Streaming Zipformer2 keyword spotter

class ContextState:
    """A node of the keyword graph (an Aho-Corasick automaton over BPE token ids), after sherpa-onnx's ContextGraph."""
    __slots__ = ("token", "token_score", "node_score", "output_score", "level", "ac_threshold", "is_end",
                 "phrase", "next", "fail", "output")

    def __init__(self, token, token_score, node_score, output_score, level=0, ac_threshold=0.0, is_end=False, phrase=""):
        self.token, self.token_score, self.node_score, self.output_score = token, token_score, node_score, output_score
        self.level, self.ac_threshold, self.is_end, self.phrase = level, ac_threshold, is_end, phrase
        self.next, self.fail, self.output = {}, None, None


class ContextGraph:
    def __init__(self, token_ids, scores, phrases, thresholds, default_score, default_threshold):
        self.root = ContextState(-1, 0.0, 0.0, 0.0)
        self.root.fail = self.root
        for ids, sc, ph, th in zip(token_ids, scores, phrases, thresholds):
            sc = sc or default_score
            th = th or default_threshold
            node = self.root
            for j, tok in enumerate(ids):
                is_end = j == len(ids) - 1
                if tok not in node.next:
                    node.next[tok] = ContextState(tok, sc, node.node_score + sc, node.node_score + sc if is_end else 0.0,
                                                  j + 1, th if is_end else 0.0, is_end, ph if is_end else "")
                else:
                    child = node.next[tok]
                    child.token_score = max(sc, child.token_score)
                    child.node_score = node.node_score + child.token_score
                    is_end = is_end or child.is_end
                    child.output_score = child.node_score if is_end else 0.0
                    child.is_end = is_end
                    if j == len(ids) - 1:
                        child.phrase, child.ac_threshold = ph, th
                node = node.next[tok]
        self._fill_fail_output()

    def _fill_fail_output(self):
        queue = []
        for child in self.root.next.values():
            child.fail = self.root
            queue.append(child)
        while queue:
            cur = queue.pop(0)
            for tok, child in cur.next.items():
                fail = cur.fail
                if tok in fail.next:
                    fail = fail.next[tok]
                else:
                    fail = fail.fail
                    while tok not in fail.next:
                        fail = fail.fail
                        if fail.token == -1:
                            break
                    if tok in fail.next:
                        fail = fail.next[tok]
                child.fail = fail
                out = fail
                while not out.is_end:
                    out = out.fail
                    if out.token == -1:
                        out = None
                        break
                child.output = out
                child.output_score += 0.0 if out is None else out.output_score
                queue.append(child)

    def forward(self, state, token):
        """(score, next state) after `token`; the strict-mode ForwardOneStep of sherpa-onnx."""
        if token in state.next:
            node = state.next[token]
            score = node.token_score
        else:
            node = state.fail
            while token not in node.next:
                node = node.fail
                if node.token == -1:
                    break
            if token in node.next:
                node = node.next[token]
            score = node.node_score - state.node_score
        return score + node.output_score, node

    @staticmethod
    def matched(state):
        """The keyword node `state` completes, or None."""
        if state.is_end:
            return state
        return state.output


def encode_keywords(lines, tokens):
    """sherpa-onnx keyword lines -> (token ids, boost scores, thresholds, phrases). A line is BPE tokens
    separated by spaces, optionally followed by ':score', '#threshold' and '@phrase' fields. A phrase
    defaults to the tokens joined, '\u2581' written as a space."""
    index = {t: i for i, t in enumerate(tokens)}
    ids, scores, thresholds, phrases = [], [], [], []
    for line in lines:
        if not line.strip():
            continue
        row, score, thr, phrase = [], 0.0, 0.0, ""
        for word in line.split():
            if word in index:
                row.append(index[word])
            elif word[0] == ":":
                score = float(word[1:])
            elif word[0] == "#":
                thr = float(word[1:])
            elif word[0] == "@":
                phrase = word[1:]
            else:
                raise ValueError(f"no token {word!r} in line {line!r}")
        if not phrase:
            phrase = "".join(tokens[i] for i in row).replace("\u2581", " ").strip()
        ids.append(row)
        scores.append(score)
        thresholds.append(thr)
        phrases.append(phrase)
    return ids, scores, thresholds, phrases


class _Hyp:
    __slots__ = ("ys", "timestamps", "ys_probs", "log_prob", "trailing", "state")

    def __init__(self, ys, state, log_prob=0.0):
        self.ys, self.timestamps, self.ys_probs, self.log_prob, self.trailing, self.state = list(ys), [], [], log_prob, 0, state

    def copy(self):
        h = _Hyp(self.ys, self.state, self.log_prob)
        h.timestamps, h.ys_probs, h.trailing = list(self.timestamps), list(self.ys_probs), self.trailing
        return h


class KeywordSpotter:
    """Streaming Zipformer2 transducer + keyword beam search, a re-write (numpy + onnxruntime) of sherpa-onnx's
    KeywordSpotter: 45-frame chunks advancing 32 frames, the encoder's cache tensors carried along, at most
    4 active paths, a keyword fires when its graph node is reached, more than one blank follows and the mean
    token probability reaches the keyword's threshold. After a detection (or 1.5 s of trailing blanks)
    the decoder state and caches are reset."""

    def __init__(self, enc, dec, joiner, tokens, graph, max_active=4, trailing_blanks=1):
        self.enc, self.prov = ot.session(enc)
        self.dec, _ = ot.session(dec)
        self.joiner, _ = ot.session(joiner)
        self.tokens, self.graph = tokens, graph
        self.max_active, self.min_trailing = max_active, trailing_blanks
        meta = self.enc.get_modelmeta().custom_metadata_map
        self.chunk_len, self.chunk_shift = int(meta["T"]), int(meta["decode_chunk_len"])
        self.context = int(self.dec.get_modelmeta().custom_metadata_map["context_size"])
        self.unk = tokens.index("<unk>") if "<unk>" in tokens else -1
        self.in_names = [i.name for i in self.enc.get_inputs()]
        self.reset()

    def reset(self):
        import numpy as np
        self.states = [np.zeros([1 if isinstance(d, str) else d for d in i.shape], dtype="int64" if "int64" in i.type else "float32")
                       for i in self.enc.get_inputs()[1:]]
        self.hyps = {(-1,) * (self.context - 1) + (0,): _Hyp([-1] * (self.context - 1) + [0], self.graph.root)}
        self.frame_offset = 0
        self.trailing = 0
        self.keyword = None

    def _log_softmax(self, x):
        import numpy as np
        m = x.max(axis=1, keepdims=True)
        return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))

    def _decode_frames(self, enc_out):
        import numpy as np
        blanks = [-1] * (self.context - 1) + [0]
        for t in range(enc_out.shape[1]):
            prev = list(self.hyps.values())
            dec_in = np.array([h.ys[-self.context:] for h in prev], dtype="int64")
            dec_out = self.dec.run(None, {self.dec.get_inputs()[0].name: dec_in})[0]
            enc_t = np.repeat(enc_out[:, t, :], len(prev), axis=0)
            logit = self.joiner.run(None, {self.joiner.get_inputs()[0].name: enc_t, self.joiner.get_inputs()[1].name: dec_out})[0]
            logp = self._log_softmax(logit.astype("float32"))
            total = logp + np.array([h.log_prob for h in prev], dtype="float32")[:, None]
            vocab = logp.shape[1]
            order = np.argsort(-total.reshape(-1), kind="stable")[:self.max_active]
            new = {}
            for k in order:
                hi, tok = int(k) // vocab, int(k) % vocab
                h = prev[hi].copy()
                ctx_score = 0.0
                if tok != 0 and tok != self.unk:
                    h.ys.append(tok)
                    h.timestamps.append(t + self.frame_offset)
                    h.ys_probs.append(float(np.exp(logp[hi, tok])))
                    h.trailing = 0
                    ctx_score, h.state = self.graph.forward(h.state, tok)
                    if h.state.token == -1:    # back at the root: forget the decoder history
                        h.ys, h.timestamps, h.ys_probs = list(blanks), [], []
                else:
                    h.trailing += 1
                h.log_prob = float(total[hi, tok]) + ctx_score
                key = tuple(h.ys)
                if key in new:
                    new[key].log_prob = float(np.logaddexp(new[key].log_prob, h.log_prob))
                else:
                    new[key] = h
            best = max(new.values(), key=lambda h: h.log_prob)
            node = self.graph.matched(best.state)
            if node is not None:
                prob = sum(best.ys_probs[:node.level]) / node.level
                if best.trailing > self.min_trailing and prob >= node.ac_threshold:
                    self.keyword = (node.phrase, list(best.ys[-node.level:]), list(best.timestamps[-node.level:]))
                    new = {tuple(blanks): _Hyp(blanks, self.graph.root)}
            self.hyps = new
        best = max(self.hyps.values(), key=lambda h: h.log_prob)
        self.trailing = best.trailing
        self.frame_offset += enc_out.shape[1]

    def run(self, feats):
        """Detections [(keyword, token ids, timestamps in output frames since the last reset, chunk start in
        input frames)] over a (frames, 80) feature matrix, fed like the sherpa-onnx python example does."""
        import numpy as np
        detections, processed = [], 0
        enc_names = [o.name for o in self.enc.get_outputs()]
        while processed + self.chunk_len < len(feats):
            if self.trailing * 4 * 0.01 > 1.5:
                self.reset()
            x = feats[processed:processed + self.chunk_len][None].astype("float32")
            feed = dict(zip(self.in_names, [x] + self.states))
            out = self.enc.run(enc_names, feed)
            self.states = out[1:]
            self._decode_frames(out[0])
            if self.keyword is not None:
                phrase, toks, stamps = self.keyword
                detections.append((phrase, toks, stamps, processed))
                self.reset()
            processed += self.chunk_shift
        return detections


def task_kws(bench):
    names = ["kws-encoder.onnx", "kws-decoder.onnx", "kws-joiner.onnx"]
    enc, dec, joi = ot.assets(*names)
    tok_path, kw_path, test_kw_path, w0, w1 = ot.assets("kws-tokens.txt", "kws-keywords.txt", "kws-test-keywords.txt",
                                                        "kws-test-0.wav", "kws-test-1.wav")
    tokens = read_tokens(tok_path)
    lines = pathlib.Path(kw_path).read_text(encoding="utf-8").splitlines() + pathlib.Path(test_kw_path).read_text(encoding="utf-8").splitlines()
    ids, scores, thresholds, phrases = encode_keywords(lines, tokens)
    graph = ContextGraph(ids, scores, phrases, thresholds, 1.0, 0.25)
    spotter = KeywordSpotter(enc, dec, joi, tokens, graph)
    feats_per_clip, seconds = [], 0.0
    for wav in (w0, w1):
        audio, rate = read_wav(wav)
        if rate != 16000:
            raise ValueError(f"{wav}: expected 16 kHz")
        seconds += len(audio) / 16000
        import numpy as np
        padded = np.concatenate([audio, np.zeros(int(0.66 * 16000), dtype="float32")])   # tail padding, as the sherpa-onnx example does
        feats_per_clip.append(kaldi_fbank(padded, snip_edges=False))

    def detect_all():
        res = []
        for f in feats_per_clip:
            spotter.reset()
            res.append(spotter.run(f))
        return res

    results = detect_all()
    flat = [d for clip in results for d in clip]
    output = {"keywords": "|".join(" ".join(d[0] for d in clip) for clip in results),
              "token_ids": "|".join(",".join(map(str, d[1])) for d in flat),
              "timestamps": "|".join(",".join(map(str, d[2])) for d in flat),
              "chunk_starts": [d[3] for d in flat]}
    detail = f"{len(flat)} keyword detections in {seconds:.1f} s of audio ({len(lines)} keywords loaded): {', '.join(d[0] for d in flat)}; {spotter.prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "3"))):
            t0 = time.perf_counter()
            detect_all()
            times.append(time.perf_counter() - t0)
        metrics = {"realtime_factor": seconds / ot.med(times), "latency_ms_per_clip": ot.med(times) * 1000 / len(feats_per_clip)}
    return output, detail, metrics


# ---------------------------------------------------------------- Zipformer audio tagging (AudioSet, 527 classes)

AUDIOTAG_CLIPS = (1, 2, 5, 7, 9, 13)


def read_labels(path):
    """AudioSet class_labels_indices.csv (index, mid, display_name) -> display names in index order."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    return [r[2] for r in sorted(rows, key=lambda r: int(r[0]))]


def top_k(values, k):
    """Indices of the k largest values, largest first, ties to the lower index."""
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def tag_clip(sess, audio):
    """Sigmoid scores of the 527 AudioSet classes for a 16 kHz clip: Kaldi fbank (samples in [-1, 1)), icefall Zipformer."""
    import numpy as np
    feats = kaldi_fbank(audio, snip_edges=False)
    return sess.run(None, {"x": feats[None], "x_lens": np.array([len(feats)], dtype="int64")})[0][0]


def task_tag(bench):
    model, labels_path = ot.assets("audiotag-model.onnx", "audiotag-labels.csv")
    wavs = ot.assets(*[f"audiotag-test-{n}.wav" for n in AUDIOTAG_CLIPS])
    labels = read_labels(labels_path)
    sess, prov = ot.session(model)
    clips = []
    for w in wavs:
        audio, rate = read_wav(w)
        if rate != 16000:
            raise ValueError(f"{w}: expected 16 kHz")
        clips.append(audio)
    scores = [tag_clip(sess, a) for a in clips]
    if any(len(sc) != len(labels) for sc in scores):
        raise RuntimeError("the model's class count differs from the label file")
    tops = [top_k(list(map(float, sc)), 3) for sc in scores]
    output = {"top3_ids": "|".join(",".join(map(str, t)) for t in tops),
              "top3_scores": [[round(float(sc[i]), 3) for i in t] for sc, t in zip(scores, tops)]}
    seconds = sum(len(a) for a in clips) / 16000
    detail = "; ".join(f"{n}.wav: {labels[t[0]]}" for n, t in zip(AUDIOTAG_CLIPS, tops)) + f"; {seconds:.0f} s of audio, {prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "3"))):
            t0 = time.perf_counter()
            for a in clips:
                tag_clip(sess, a)
            times.append(time.perf_counter() - t0)
        metrics = {"realtime_factor": seconds / ot.med(times), "clips_per_s": len(clips) / ot.med(times)}
    return output, detail, metrics


TASKS = {"asr": task_asr, "speaker": task_speaker, "tts": task_tts, "enhance": task_enhance, "kws": task_kws, "tag": task_tag}


def main(argv):
    ot.TASKS.update(TASKS)
    return ot.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
