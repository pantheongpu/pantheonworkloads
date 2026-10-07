#!/usr/bin/env python3
"""The ort-* workloads' payloads: pretrained ONNX models run on onnxruntime.

    python3 -I tools/ort_tasks.py <task> [--bench]

Tasks: vad (Silero VAD), ocr (PP-OCRv3 through rapidocr_onnxruntime), mnist, mobilenetv2, and the vision
models of tools/ort_vision.py (yunet, sface, pphumanseg, nanodet, yolox, ssdmobilenet, crnn, shufflenet,
efficientnet).
Environment from bin/pw: PW_TARGET (cpu | gpu), PW_WORKLOAD_DIR (its model.sha256 pins the files).
Prints, as the last stdout line, {"output", "detail", "metrics"} (docs/workload-contract.md);
metrics are only set with --bench on the gpu target. Exit 77: cannot run here (no network, no GPU
execution provider). The gpu target never silently runs on the CPU: when no GPU provider is
usable the run is a SKIP.
"""
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import tarfile
import time
import wave

HERE = pathlib.Path(__file__).resolve().parent
GPU_PROVIDERS = ("CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider")


class Skip(Exception):
    """The workload cannot run here (exit 77)."""


def _assets():
    spec = importlib.util.spec_from_file_location("ort_assets", HERE / "ort_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assets(*names):
    """Local paths of the pinned files `names` (downloaded and verified once)."""
    wdir = os.environ["PW_WORKLOAD_DIR"]
    a = _assets()
    sums = a.parse_sums((pathlib.Path(wdir) / "model.sha256").read_text())
    cache = pathlib.Path(os.environ.get("PW_CACHE") or pathlib.Path.home() / ".cache" / "pantheonworkloads") / "ort-assets"
    try:
        return [a.fetch(n, sums[n], cache) for n in names]
    except (RuntimeError, KeyError) as e:
        raise Skip(str(e))


def choose_provider(target, available, gpu_ok=GPU_PROVIDERS):
    """The execution provider for a target: CPU for cpu; for gpu the first GPU provider that
    onnxruntime offers, else Skip (never the CPU)."""
    if target == "cpu":
        return "CPUExecutionProvider"
    if target == "gpu":
        for p in gpu_ok:
            if p in available:
                return p
        raise Skip(f"target gpu: onnxruntime offers no GPU execution provider (has {sorted(available)})")
    raise Skip(f"target {target} is not supported by the ort workloads")


def session(path, log_level=None):
    """(InferenceSession, provider name) for the current target. `log_level` 3 silences
    onnxruntime's warnings (some zoo models trigger a screenful of them)."""
    import onnxruntime as ort
    prov = choose_provider(os.environ.get("PW_TARGET", "cpu"), ort.get_available_providers())
    opts = ort.SessionOptions()
    if log_level is not None:
        opts.log_severity_level = log_level
    if prov == "CPUExecutionProvider":   # fixed thread count: stable results and a polite use of shared hosts
        opts.intra_op_num_threads = opts.inter_op_num_threads = int(os.environ.get("PW_ORT_THREADS", "1"))
    providers = [prov] if prov == "CPUExecutionProvider" else [prov, "CPUExecutionProvider"]
    model = bytes(path) if isinstance(path, (bytes, bytearray)) else str(path)   # a serialized graph or a file name
    s = ort.InferenceSession(model, sess_options=opts, providers=providers)
    if s.get_providers()[0] != prov:
        raise Skip(f"{prov} was requested but the session runs on {s.get_providers()}")
    return s, prov


def med(values):
    return statistics.median(values)


# ---------------------------------------------------------------- Silero VAD

def read_wav16k(path):
    """Mono 16 kHz 16-bit wav -> float32 in [-1, 1) (what silero_vad.read_audio yields)."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError("expected 16 kHz mono 16-bit PCM")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0


def vad_probs(sess, audio):
    """Speech probability of every 512-sample window, as Silero's OnnxWrapper computes it at 16 kHz:
    64 samples of context before each window, a (2, 1, 128) recurrent state carried along."""
    import numpy as np
    state = np.zeros((2, 1, 128), dtype="float32")
    context = np.zeros((1, 64), dtype="float32")
    sr = np.array(16000, dtype="int64")
    probs = []
    for start in range(0, len(audio), 512):
        chunk = audio[start:start + 512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))
        x = np.concatenate([context, chunk[None, :]], axis=1)
        out, state = sess.run(None, {"input": x, "state": state, "sr": sr})
        context = x[:, -64:]
        probs.append(float(out[0, 0]))
    return probs


def speech_segments(probs, n_samples, threshold=0.5, min_speech_ms=250, min_silence_ms=100, pad_ms=30, sr=16000, win=512):
    """Silero's get_speech_timestamps_from_probs (v6.2.3) with its default parameters and no
    maximum speech length, as sample offsets [(start, end), ...]. Re-written, not copied."""
    neg = max(threshold - 0.15, 0.01)
    min_speech = sr * min_speech_ms / 1000
    pad = sr * pad_ms / 1000
    min_silence = sr * min_silence_ms / 1000
    triggered, temp_end, cur, speeches = False, 0, {}, []
    for i, p in enumerate(probs):
        pos = win * i
        if p >= threshold and temp_end:
            temp_end = 0
        if p >= threshold and not triggered:
            triggered, cur = True, {"start": pos}
            continue
        if p < neg and triggered:
            if not temp_end:
                temp_end = pos
            if pos - temp_end < min_silence:
                continue
            cur["end"] = temp_end
            if cur["end"] - cur["start"] > min_speech:
                speeches.append(cur)
            cur, temp_end, triggered = {}, 0, False
    if cur and n_samples - cur["start"] > min_speech:
        cur["end"] = n_samples
        speeches.append(cur)
    for i, s in enumerate(speeches):
        if i == 0:
            s["start"] = int(max(0, s["start"] - pad))
        if i != len(speeches) - 1:
            gap = speeches[i + 1]["start"] - s["end"]
            if gap < 2 * pad:
                s["end"] += int(gap // 2)
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - gap // 2))
            else:
                s["end"] = int(min(n_samples, s["end"] + pad))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - pad))
        else:
            s["end"] = int(min(n_samples, s["end"] + pad))
    return [(s["start"], s["end"]) for s in speeches]


def task_vad(bench):
    model, wav = assets("silero_vad.onnx", "silero-test.wav")
    audio = read_wav16k(wav)
    sess, prov = session(model)
    probs = vad_probs(sess, audio)
    segs = speech_segments(probs, len(audio))
    output = {"segments": " ".join(f"{a}-{b}" for a, b in segs), "n_segments": len(segs),
              "n_windows": len(probs), "mean_prob": round(sum(probs) / len(probs), 4)}
    detail = f"{len(segs)} speech segments in {len(audio) / 16000:.1f} s of audio, {prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "5"))):
            t = time.perf_counter()
            vad_probs(sess, audio)
            times.append(time.perf_counter() - t)
        metrics = {"windows_per_s": len(probs) / med(times), "realtime_factor": len(audio) / 16000 / med(times)}
    return output, detail, metrics


# ---------------------------------------------------------------- PP-OCRv3 via RapidOCR

OCR_MODEL_SHA256 = {   # the files inside the rapidocr_onnxruntime 1.2.3 wheel
    "ch_PP-OCRv3_det_infer.onnx": "3439588c030faea393a54515f51e983d8e155b19a2e8aba7891934c1cf0de526",
    "ch_PP-OCRv3_rec_infer.onnx": "897a3ededb38fee0dae2c1ccee38241f37df202c9509e3abca02e9217c5ee615",
    "ch_ppocr_mobile_v2.0_cls_infer.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
}


def ocr_engine(target):
    import onnxruntime as ort
    import rapidocr_onnxruntime as r
    _assets_mod = _assets()
    mdir = pathlib.Path(r.__file__).parent / "models"
    for name, want in OCR_MODEL_SHA256.items():
        if not (mdir / name).exists() or _assets_mod.sha256(mdir / name) != want:
            raise Skip(f"{name} in the installed rapidocr_onnxruntime does not match the pinned sha256")
    prov = choose_provider(target, ort.get_available_providers(), gpu_ok=("CUDAExecutionProvider",))  # RapidOCR 1.2.3 only knows CUDA
    use_cuda = prov == "CUDAExecutionProvider"
    if use_cuda:   # RapidOCR 1.2.3's kwargs cannot set use_cuda for cls and rec (the key keeps its prefix): patch the config it reads
        from rapidocr_onnxruntime import rapid_ocr_api as api
        orig = api.read_yaml

        def read_yaml_cuda(path):
            cfg = orig(path)
            for part in ("Det", "Cls", "Rec"):
                cfg[part]["use_cuda"] = True
            return cfg
        api.read_yaml = read_yaml_cuda
    eng = r.RapidOCR()
    sessions = [eng.text_detector.infer.session, eng.text_recognizer.session.session]
    for s in sessions:
        if s.get_providers()[0] != prov:
            raise Skip(f"{prov} was requested but a RapidOCR session runs on {s.get_providers()}")
    return eng, prov


def task_ocr(bench):
    (img,) = assets("ocr-en.jpg")
    eng, prov = ocr_engine(os.environ.get("PW_TARGET", "cpu"))
    result, _ = eng(str(img))
    lines = [(r[1], float(r[2])) for r in (result or [])]
    output = {"text": "|".join(t for t, _ in lines), "n_lines": len(lines),
              "mean_score": round(sum(s for _, s in lines) / max(len(lines), 1), 3)}
    detail = f"{len(lines)} text lines, PP-OCRv3 det+cls+rec, {prov}"
    metrics = {}
    if bench:
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "10"))):
            t = time.perf_counter()
            eng(str(img))
            times.append(time.perf_counter() - t)
        metrics = {"images_per_s": 1 / med(times), "latency_ms": med(times) * 1000}
    return output, detail, metrics


# ---------------------------------------------------------------- ONNX model zoo classifiers

def zoo_sample(tar_path, stem):
    """(input, expected output) tensors of test_data_set_0 in a model-zoo tarball, as numpy arrays. The
    tarball's top directory is usually `stem` but not always (efficientnet-lite4-11.tar.gz has
    efficientnet-lite4/), so the member is found by its path below that directory."""
    import onnx
    from onnx import numpy_helper
    arrays = []
    with tarfile.open(tar_path) as t:
        for name in ("input_0", "output_0"):
            tp = onnx.TensorProto()
            member = [m for m in t.getnames() if m.count("/") == 2 and m.endswith(f"/test_data_set_0/{name}.pb")]
            if len(member) != 1:
                raise RuntimeError(f"{tar_path}: expected one test_data_set_0/{name}.pb, found {member}")
            tp.ParseFromString(t.extractfile(member[0]).read())
            arrays.append(numpy_helper.to_array(tp))
    return arrays


def top_k(logits, k):
    """Indices of the k largest values, largest first, ties to the lower index."""
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    return order[:k]


def is_probability(v):
    """True for a softmax-like vector: no negative entry, sum 1."""
    return min(v) >= 0 and abs(sum(v) - 1) < 1e-3


def task_zoo(stem, bench):
    import numpy as np
    model, tar = assets(f"{stem}.onnx", f"{stem}.tar.gz")
    x, want = zoo_sample(tar, stem)
    sess, prov = session(model)
    name = sess.get_inputs()[0].name
    (got,) = sess.run(None, {name: x})
    err = float(np.abs(got - want).max())
    if err > 1e-3 or int(got.argmax()) != int(want.argmax()):
        raise RuntimeError(f"differs from the model zoo's own expected output (max abs diff {err:.3g})")
    logits = got[0].tolist()
    ids = top_k(logits, 5 if len(logits) > 10 else 3)
    if is_probability(logits):   # softmax output: ranks among near-zero probabilities are numerical noise
        ids = [i for i in ids if logits[i] >= 1e-3] or ids[:1]
    output = {"top_ids": " ".join(map(str, ids)), "top_logits": [round(logits[i], 3) for i in ids]}
    detail = f"top class {ids[0]}; max abs diff to the zoo's expected output {err:.2g}; {prov}"
    metrics = {}
    if bench:
        fixed = sess.get_inputs()[0].shape[0]   # a model exported with a fixed batch dimension is timed at that batch
        batch = fixed if isinstance(fixed, int) else int(os.environ.get("PW_ORT_BATCH", "32"))
        xb = np.repeat(x, batch, axis=0)
        for _ in range(5):
            sess.run(None, {name: xb})
        times = []
        for _ in range(int(os.environ.get("PW_ORT_REPEATS", "30"))):
            t = time.perf_counter()
            sess.run(None, {name: xb})
            times.append(time.perf_counter() - t)
        metrics = {"images_per_s": batch / med(times), "batch_latency_ms": med(times) * 1000}
    return output, detail, metrics


TASKS = {
    "vad": task_vad,
    "ocr": task_ocr,
    "mnist": lambda b: task_zoo("mnist-12", b),
    "mobilenetv2": lambda b: task_zoo("mobilenetv2-12", b),
}


def all_tasks():
    """TASKS plus the vision tasks (tools/ort_vision.py, loaded by path: `python3 -I` has no script dir on sys.path)."""
    spec = importlib.util.spec_from_file_location("ort_vision", HERE / "ort_vision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {**TASKS, **mod.register(sys.modules[__name__])}


def main(argv):
    tasks = all_tasks()
    args = [a for a in argv[1:] if a != "--bench"]
    bench = "--bench" in argv[1:]
    if len(args) != 1 or args[0] not in tasks:
        print(f"usage: ort_tasks.py {{{'|'.join(tasks)}}} [--bench]", file=sys.stderr)
        return 2
    if bench and os.environ.get("PW_TARGET") != "gpu":
        print("SKIP: benchmarks are for the real gpu target only")
        return 77
    try:
        output, detail, metrics = tasks[args[0]](bench)
    except Skip as e:
        print(f"SKIP: {e}")
        return 77
    except ImportError as e:
        print(f"SKIP: missing Python package ({e})")
        return 77
    print(json.dumps({"output": output, "detail": detail, "metrics": metrics if os.environ.get("PW_TARGET") == "gpu" else {}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
