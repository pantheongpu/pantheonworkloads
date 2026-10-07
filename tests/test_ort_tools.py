"""Tests for the ort-* workloads' helpers (tools/ort_assets.py, tools/ort_tasks.py). No network,
no onnxruntime, no model: the pure logic, the pinned-file tables and the exit-77 paths.

    python3 -m unittest discover -s tests -v
"""
import hashlib
import io
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ort_assets as assets  # noqa: E402
import ort_tasks as tasks  # noqa: E402
import speech_tasks as speech  # noqa: E402

try:
    import numpy as np
except ImportError:   # the system Python of a CI box may not have numpy: the numeric tests then skip
    np = None

ORT_WORKLOADS = ["silero-vad-onnx", "silero-vad-onnx-bench", "ppocr-rapidocr", "ppocr-rapidocr-bench",
                 "onnx-zoo-mnist", "onnx-zoo-mobilenetv2", "onnx-zoo-mobilenetv2-bench", "spacy-en-core-web-sm", "spacy-en-core-web-sm-bench",
                 "moonshine-tiny-en-onnx", "moonshine-tiny-en-onnx-bench", "wespeaker-resnet34-onnx", "wespeaker-resnet34-onnx-bench",
                 "kokoro-tts-int8-onnx", "kokoro-tts-int8-onnx-bench", "gtcrn-enhance-onnx", "gtcrn-enhance-onnx-bench",
                 "kws-zipformer-gigaspeech-onnx", "kws-zipformer-gigaspeech-onnx-bench",
                 "zipformer-audio-tagging-onnx", "zipformer-audio-tagging-onnx-bench"]
SPEECH_WORKLOADS = ORT_WORKLOADS[-12:]
# sherpa-onnx's release tags are rolling names (no version in them): the sha256 in model.sha256 is the pin
ROLLING_RELEASE_TAGS = {"asr-models", "tts-models", "speaker-recongition-models", "kws-models", "audio-tagging-models"}


class Assets(unittest.TestCase):
    def test_parse_sums(self):
        d = "a" * 64
        self.assertEqual(assets.parse_sums(f"# c\n\n{d}  x.onnx\n{d.upper()}  *y.bin\n"), {"x.onnx": d, "y.bin": d})

    def test_fetch_checks_the_checksum(self):
        with tempfile.TemporaryDirectory() as t:
            src = pathlib.Path(t) / "src.bin"
            src.write_bytes(b"weights")
            good = hashlib.sha256(b"weights").hexdigest()
            old = dict(assets.URLS)
            assets.URLS["w.bin"] = src.as_uri()
            try:
                p = assets.fetch("w.bin", good, pathlib.Path(t) / "cache")
                self.assertEqual(p.read_bytes(), b"weights")
                with self.assertRaises(RuntimeError):
                    assets.fetch("w.bin", "0" * 64, pathlib.Path(t) / "cache2")
                with self.assertRaises(RuntimeError):
                    assets.fetch("unknown.bin", good, pathlib.Path(t) / "cache3")
            finally:
                assets.URLS.clear()
                assets.URLS.update(old)

    def test_corrupt_cache_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            good = hashlib.sha256(b"x").hexdigest()
            dest = pathlib.Path(t) / good[:16] / "w.bin"
            dest.parent.mkdir(parents=True)
            dest.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                assets.fetch("w.bin", good, t)

    def test_every_workload_pins_every_file_it_uses(self):
        for w in ORT_WORKLOADS:
            sums = assets.parse_sums((ROOT / "workloads" / w / "model.sha256").read_text())
            self.assertTrue(sums, w)
            for name, digest in sums.items():
                self.assertTrue(name in assets.URLS or name in assets.MEMBERS, f"{w}: {name} has no URL")
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_urls_are_pinned_to_commits_or_release_tags(self):
        for name, url in assets.URLS.items():
            if "/releases/download/" in url:   # a release asset: the tag is the pin (and the sha256 pins the bytes)
                tag = url.split("/releases/download/")[1].split("/")[0]
                if tag in ROLLING_RELEASE_TAGS:
                    self.assertTrue(url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/"), name)
                else:
                    self.assertRegex(url, r"^https://github\.com/[^/]+/[^/]+/releases/download/[^/]*\d[^/]*/", name)
            else:
                self.assertRegex(url, r"^https://(raw|media)\.githubusercontent\.com/", name)
                self.assertRegex(url, r"[0-9a-f]{40}", name)   # a commit, never a branch

    def test_archive_members_pin_their_archive(self):
        for name, (archive, inside) in assets.MEMBERS.items():
            self.assertIn(archive, assets.URLS, name)
            self.assertNotIn(name, assets.URLS, name)
        for w in ORT_WORKLOADS:
            sums = assets.parse_sums((ROOT / "workloads" / w / "model.sha256").read_text())
            for name in sums:
                if name in assets.MEMBERS:
                    self.assertIn(assets.MEMBERS[name][0], sums, f"{w}: {name} needs its archive pinned")

    def test_same_file_has_the_same_sum_everywhere(self):
        seen = {}
        for w in ORT_WORKLOADS:
            for n, d in assets.parse_sums((ROOT / "workloads" / w / "model.sha256").read_text()).items():
                self.assertEqual(seen.setdefault(n, d), d, n)


class ArchiveMembers(unittest.TestCase):
    """fetch() of a member of a release archive: extracted once, checked, the archive deleted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = pathlib.Path(self.tmp.name)
        self.files = {"pkg/model.onnx": b"weights", "pkg/LICENSE": b"MIT", "pkg/unused.bin": b"x" * 10}
        self.archive = self.t / "a.tar.bz2"
        with tarfile.open(self.archive, "w:bz2") as tf:
            for n, data in self.files.items():
                ti = tarfile.TarInfo(n)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        self.asha = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.sums = {"a.tar.bz2": self.asha, "m.onnx": hashlib.sha256(b"weights").hexdigest(),
                     "LICENSE-m": hashlib.sha256(b"MIT").hexdigest()}
        self.old = (dict(assets.URLS), dict(assets.MEMBERS))
        assets.URLS["a.tar.bz2"] = self.archive.as_uri()
        assets.MEMBERS.update({"m.onnx": ("a.tar.bz2", "pkg/model.onnx"), "LICENSE-m": ("a.tar.bz2", "pkg/LICENSE")})
        self.cache = self.t / "cache"

    def tearDown(self):
        assets.URLS.clear(); assets.URLS.update(self.old[0])
        assets.MEMBERS.clear(); assets.MEMBERS.update(self.old[1])
        self.tmp.cleanup()

    def archive_copies(self):
        return list(self.cache.rglob("a.tar.bz2*"))

    def test_extracts_every_pinned_member_and_deletes_the_archive(self):
        p = assets.fetch("m.onnx", self.sums["m.onnx"], self.cache, self.sums)
        self.assertEqual(p.read_bytes(), b"weights")
        self.assertEqual(self.archive_copies(), [])
        self.assertTrue(any(self.cache.rglob("LICENSE-m")))      # the other pinned member came with it
        self.assertFalse(any(self.cache.rglob("unused.bin")))    # an unpinned member did not
        assets.URLS["a.tar.bz2"] = (self.t / "gone.tar.bz2").as_uri()   # now cached: no download needed
        self.assertEqual(assets.fetch("m.onnx", self.sums["m.onnx"], self.cache, self.sums), p)

    def test_wrong_member_sum_is_refused(self):
        sums = dict(self.sums, **{"m.onnx": "0" * 64})
        with self.assertRaises(RuntimeError):
            assets.fetch("m.onnx", "0" * 64, self.cache, sums)
        self.assertEqual(self.archive_copies(), [])
        self.assertFalse(any(self.cache.rglob("m.onnx")))

    def test_wrong_archive_sum_is_refused(self):
        sums = dict(self.sums, **{"a.tar.bz2": "1" * 64})
        with self.assertRaises(RuntimeError):
            assets.fetch("m.onnx", sums["m.onnx"], self.cache, sums)
        self.assertEqual(self.archive_copies(), [])

    def test_member_without_pinned_archive(self):
        sums = {k: v for k, v in self.sums.items() if k != "a.tar.bz2"}
        with self.assertRaises(RuntimeError):
            assets.fetch("m.onnx", sums["m.onnx"], self.cache, sums)

    def test_missing_member_in_archive(self):
        assets.MEMBERS["m.onnx"] = ("a.tar.bz2", "pkg/nope")
        with self.assertRaises(RuntimeError):
            assets.fetch("m.onnx", self.sums["m.onnx"], self.cache, self.sums)


class Providers(unittest.TestCase):
    def test_cpu(self):
        self.assertEqual(tasks.choose_provider("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]), "CPUExecutionProvider")

    def test_gpu_never_falls_back_to_cpu(self):
        with self.assertRaises(tasks.Skip):
            tasks.choose_provider("gpu", ["CPUExecutionProvider"])
        self.assertEqual(tasks.choose_provider("gpu", ["ROCMExecutionProvider", "CPUExecutionProvider"]), "ROCMExecutionProvider")
        self.assertEqual(tasks.choose_provider("gpu", ["MIGraphXExecutionProvider", "CUDAExecutionProvider"]), "CUDAExecutionProvider")
        with self.assertRaises(tasks.Skip):
            tasks.choose_provider("gpu", ["ROCMExecutionProvider"], gpu_ok=("CUDAExecutionProvider",))

    def test_sim_is_not_supported(self):
        with self.assertRaises(tasks.Skip):
            tasks.choose_provider("sim:nvidia/h100", ["CPUExecutionProvider"])


class Pure(unittest.TestCase):
    def test_top_k_ties_go_to_the_lower_index(self):
        self.assertEqual(tasks.top_k([1.0, 3.0, 3.0, 2.0], 3), [1, 2, 3])

    def test_no_speech(self):
        self.assertEqual(tasks.speech_segments([0.0] * 100, 100 * 512), [])

    def test_one_segment_is_padded(self):
        probs = [0.0] * 20 + [0.9] * 30 + [0.0] * 50   # speech in windows 20..49
        (seg,) = tasks.speech_segments(probs, 100 * 512)
        self.assertEqual(seg[0], 20 * 512 - 480)                 # 30 ms pad at 16 kHz = 480 samples
        self.assertEqual(seg[1], 50 * 512 + 480)                 # speech ends at the first window below the exit threshold, + pad

    def test_short_blip_is_dropped(self):
        probs = [0.0] * 20 + [0.9] * 3 + [0.0] * 50           # 3 windows = 96 ms < 250 ms
        self.assertEqual(tasks.speech_segments(probs, 73 * 512), [])

    def test_short_silence_does_not_split(self):
        probs = [0.0] * 5 + [0.9] * 20 + [0.0] * 2 + [0.9] * 20 + [0.0] * 20   # 2 windows = 64 ms < 100 ms
        self.assertEqual(len(tasks.speech_segments(probs, len(probs) * 512)), 1)

    def test_long_silence_splits(self):
        probs = [0.0] * 5 + [0.9] * 20 + [0.0] * 30 + [0.9] * 20 + [0.0] * 5
        self.assertEqual(len(tasks.speech_segments(probs, len(probs) * 512)), 2)

    def test_speech_running_to_the_end(self):
        probs = [0.0] * 5 + [0.9] * 40
        (seg,) = tasks.speech_segments(probs, 45 * 512 - 100)
        self.assertEqual(seg[1], 45 * 512 - 100)


class SpeechText(unittest.TestCase):
    def test_word_error_rate(self):
        self.assertEqual(speech.word_error_rate("the cat sat", "the cat sat"), 0.0)
        self.assertEqual(speech.word_error_rate("THE CAT, SAT", "the cat sat."), 0.0)    # case and punctuation ignored
        self.assertAlmostEqual(speech.word_error_rate("the cat sat", "the dog sat"), 1 / 3)
        self.assertAlmostEqual(speech.word_error_rate("a b c d", "a d"), 0.5)
        self.assertEqual(speech.word_error_rate("a", ""), 1.0)

    def test_moonshine_text_pieces_and_bytes(self):
        toks = ["<unk>", "\u2581Hel", "lo", "\u2581world", "<0xC3>", "<0xA9>"]
        self.assertEqual(speech.moonshine_text([1, 2, 3], toks), "Hello world")
        self.assertEqual(speech.moonshine_text([3, 4, 5], toks), "world\u00e9")      # two byte tokens form one UTF-8 character

    def test_read_tokens_tab_and_space_separated(self):
        with tempfile.TemporaryDirectory() as t:
            f = pathlib.Path(t) / "tokens.txt"
            f.write_text("<unk>\t0\n<s>\t1\n\u2581a b\t2\n", encoding="utf-8")
            self.assertEqual(speech.read_tokens(f), ["<unk>", "<s>", "\u2581a b"])

    def test_kokoro_ids_pad_with_zero_and_refuse_unknown_symbols(self):
        table = {"a": 5, " ": 16, ".": 4}
        self.assertEqual(speech.kokoro_ids("a a.", table), [0, 5, 16, 5, 4, 0])
        with self.assertRaises(ValueError):
            speech.kokoro_ids("ab", table)

    def test_read_phoneme_tokens_keeps_the_space_symbol(self):
        with tempfile.TemporaryDirectory() as t:
            f = pathlib.Path(t) / "tokens.txt"
            f.write_text("$ 0\n; 1\n  16\n\u02c8 156\n", encoding="utf-8")
            self.assertEqual(speech.read_phoneme_tokens(f), {"$": 0, ";": 1, " ": 16, "\u02c8": 156})


class KeywordGraph(unittest.TestCase):
    TOKENS = ["<blk>", "<sos/eos>", "<unk>", "\u2581HE", "LL", "O", "\u2581WORLD", "\u2581GO", "\u2581HOME"]

    def graph(self, lines, score=1.0, thr=0.25):
        ids, scores, thresholds, phrases = speech.encode_keywords(lines, self.TOKENS)
        return speech.ContextGraph(ids, scores, phrases, thresholds, score, thr), ids

    def test_encode_keywords_phrase_score_and_threshold_fields(self):
        ids, scores, thr, phrases = speech.encode_keywords(
            ["\u2581HE LL O \u2581WORLD", "\u2581GO \u2581HOME :2.5 #0.4 @go_home", ""], self.TOKENS)
        self.assertEqual(ids, [[3, 4, 5, 6], [7, 8]])
        self.assertEqual(scores, [0.0, 2.5])
        self.assertEqual(thr, [0.0, 0.4])
        self.assertEqual(phrases, ["HELLO WORLD", "go_home"])      # default phrase: the tokens joined, word marker -> space
        with self.assertRaises(ValueError):
            speech.encode_keywords(["\u2581HE NOPE"], self.TOKENS)

    def test_walking_a_keyword_collects_boosts_and_matches_at_its_end(self):
        g, ids = self.graph(["\u2581HE LL O \u2581WORLD"], score=1.0)
        state, total = g.root, 0.0
        for k, tok in enumerate(ids[0]):
            sc, state = g.forward(state, tok)
            total += sc
            self.assertEqual(g.matched(state) is not None, k == len(ids[0]) - 1)
        # 1.0 per token, and the last step adds the whole node score (4.0) as sherpa-onnx's output_score does
        self.assertAlmostEqual(total, 8.0)
        self.assertEqual(g.matched(state).phrase, "HELLO WORLD")
        self.assertEqual(g.matched(state).ac_threshold, 0.25)

    def test_a_wrong_token_loses_the_boost_so_far(self):
        g, ids = self.graph(["\u2581HE LL O \u2581WORLD"])
        sc1, s1 = g.forward(g.root, ids[0][0])
        sc2, s2 = g.forward(s1, 7)                          # a token that continues nothing
        self.assertAlmostEqual(sc1 + sc2, 0.0)              # the boost given for the first token is taken back
        self.assertIs(s2, g.root)

    def test_overlapping_keywords_share_a_prefix_and_report_the_shorter_one(self):
        g, _ = self.graph(["\u2581GO", "\u2581GO \u2581HOME @both"])
        _, s = g.forward(g.root, 7)
        self.assertEqual(g.matched(s).phrase, "GO")
        _, s = g.forward(s, 8)
        self.assertEqual(g.matched(s).phrase, "both")


class TaggingText(unittest.TestCase):
    def test_top_k_ties_go_to_the_lower_index(self):
        self.assertEqual(speech.top_k([0.1, 0.9, 0.9, 0.5], 3), [1, 2, 3])

    def test_read_labels_orders_by_index_and_handles_quotes(self):
        with tempfile.TemporaryDirectory() as t:
            f = pathlib.Path(t) / "labels.csv"
            f.write_text('index,mid,display_name\n1,/m/b,"Male speech, man speaking"\n0,/m/a,Speech\n', encoding="utf-8")
            self.assertEqual(speech.read_labels(f), ["Speech", "Male speech, man speaking"])


@unittest.skipIf(np is None, "numpy is not installed here (the ort venv has it)")
class SpeechNumeric(unittest.TestCase):
    def test_window_kinds(self):
        for kind in ("povey", "hamming"):
            w = speech.window(400, kind)
            self.assertEqual(w.shape, (400,))
            self.assertTrue((w >= 0).all())
        self.assertAlmostEqual(float(speech.window(5, "hamming")[0]), 0.08)
        with self.assertRaises(ValueError):
            speech.window(10, "hann")

    def test_fbank_frame_counts(self):
        x = np.random.RandomState(0).randn(16000).astype("float32") * 1000
        self.assertEqual(speech.kaldi_fbank(x, snip_edges=True).shape, (98, 80))      # 1 + (16000 - 400) // 160
        self.assertEqual(speech.kaldi_fbank(x, snip_edges=False).shape, (100, 80))    # (16000 + 80) // 160
        self.assertEqual(speech.kaldi_fbank(x[:300], snip_edges=True).shape, (0, 80))

    def test_fbank_of_a_tone_peaks_in_the_right_mel_bin(self):
        t = np.arange(16000) / 16000.0
        x = (np.sin(2 * np.pi * 1000 * t) * 8000).astype("float32")
        f = speech.kaldi_fbank(x, snip_edges=True)
        banks = speech.mel_banks(80, 512, 16000, 20.0, -400.0)
        centre = int(np.argmax(banks[:, 1000 * 512 // 16000]))
        self.assertLessEqual(abs(int(np.argmax(f.mean(axis=0))) - centre), 1)

    def test_mel_banks_are_triangles(self):
        b = speech.mel_banks(80, 512, 16000, 20.0, -400.0)
        self.assertEqual(b.shape, (80, 256))
        self.assertTrue((b >= 0).all() and (b <= 1).all())
        self.assertTrue((b.sum(axis=1) > 0).all())

    def test_stft_istft_round_trip(self):
        x = np.random.RandomState(1).randn(5000).astype("float32")
        spec = speech.stft(x)
        self.assertEqual(spec.shape, (257, 1 + 5000 // 256, 2))
        y = speech.istft(spec, len(x))
        self.assertEqual(len(y), len(x))
        self.assertLess(float(np.abs(x - y).max()), 1e-4)

    def test_envelope_and_audio_stats(self):
        x = np.concatenate([np.zeros(2400), np.ones(2400) * 0.5]).astype("float32")
        self.assertEqual(speech.envelope(x, 24000), [0.0, 0.5])
        st = speech.audio_stats(x, 24000)
        self.assertEqual(st["n_samples"], 4800)
        self.assertEqual(st["envelope"], [0.0, 0.5])
        self.assertAlmostEqual(st["rms"], 0.35355, places=4)

    def test_cosine(self):
        self.assertAlmostEqual(speech.cosine(np.array([1.0, 0.0]), np.array([1.0, 1.0])), 2 ** -0.5)

    def test_read_wav_rejects_other_formats(self):
        import wave
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t) / "x.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0" * 8)
            with self.assertRaises(ValueError):
                speech.read_wav(p)
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(np.array([0, 16384, -32768], dtype="<i2").tobytes())
            x, rate = speech.read_wav(p)
            self.assertEqual(rate, 8000)
            self.assertEqual(x.tolist(), [0.0, 0.5, -1.0])


class Scripts(unittest.TestCase):
    def run_sh(self, workload, target, **env):
        e = dict(os.environ, PW_TARGET=target, PW_WORKLOAD_DIR=str(ROOT / "workloads" / workload),
                 PW_NO_INSTALL="1", PW_VENV_ROOT=tempfile.mkdtemp(), **env)
        return subprocess.run(["bash", str(ROOT / "workloads" / workload / "run.sh")], env=e, capture_output=True, text=True)

    def test_no_venv_and_no_install_is_a_skip(self):
        p = self.run_sh("onnx-zoo-mnist", "cpu")
        self.assertEqual(p.returncode, 77, p.stderr)
        self.assertTrue(p.stdout.startswith("SKIP:"))

    def test_sim_target_is_a_skip(self):
        self.assertEqual(self.run_sh("silero-vad-onnx", "sim:nvidia/h100").returncode, 77)

    def test_unusable_python_override_is_a_skip(self):
        self.assertEqual(self.run_sh("onnx-zoo-mnist", "cpu", PW_ORT_PYTHON="/nonexistent/python").returncode, 77)

    def test_speech_workloads_skip_without_a_venv(self):
        for w in SPEECH_WORKLOADS:
            if w.endswith("-bench"):
                continue
            p = self.run_sh(w, "cpu")
            self.assertEqual(p.returncode, 77, (w, p.stderr))
            self.assertTrue(p.stdout.startswith("SKIP:"))

    def test_speech_benchmarks_refuse_cpu_and_sim(self):
        for w in SPEECH_WORKLOADS:
            if w.endswith("-bench"):
                self.assertEqual(self.run_sh(w, "sim:nvidia/h100").returncode, 77, w)
        for t in sorted(speech.TASKS):
            p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "speech_tasks.py"), t, "--bench"],
                               env=dict(os.environ, PW_TARGET="cpu"), capture_output=True, text=True)
            self.assertEqual(p.returncode, 77, t)

    def test_speech_tasks_bad_usage(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "speech_tasks.py"), "nope"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)

    def test_speech_tasks_registered_on_the_shared_main(self):
        self.assertEqual(set(speech.TASKS), {"asr", "speaker", "tts", "enhance", "kws", "tag"})

    def test_bench_refuses_cpu(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "ort_tasks.py"), "vad", "--bench"],
                           env=dict(os.environ, PW_TARGET="cpu"), capture_output=True, text=True)
        self.assertEqual(p.returncode, 77)

    def test_bad_usage(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "ort_tasks.py"), "nope"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
