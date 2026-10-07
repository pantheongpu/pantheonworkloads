"""Tests for the ort-* workloads' helpers (tools/ort_assets.py, tools/ort_tasks.py). No network,
no onnxruntime, no model: the pure logic, the pinned-file tables and the exit-77 paths.

    python3 -m unittest discover -s tests -v
"""
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ort_assets as assets  # noqa: E402
import ort_tasks as tasks  # noqa: E402

ORT_WORKLOADS = ["silero-vad-onnx", "silero-vad-onnx-bench", "ppocr-rapidocr", "ppocr-rapidocr-bench",
                 "onnx-zoo-mnist", "onnx-zoo-mobilenetv2", "onnx-zoo-mobilenetv2-bench", "spacy-en-core-web-sm", "spacy-en-core-web-sm-bench"]
TEXT_WORKLOADS = ["onnx-zoo-bidaf", "onnx-zoo-bertsquad-int8", "onnx-zoo-bertsquad-int8-bench", "minilm-l6-v2-onnx", "minilm-l6-v2-onnx-bench",
                  "glove-wiki-gigaword-50-knn", "glove-wiki-gigaword-50-knn-bench", "spacy-en-core-web-md", "spacy-en-core-web-md-bench",
                  "spacy-multilingual-sm", "spacy-multilingual-sm-bench", "py3langid-wheel", "sentencepiece-test-model"]


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
        for w in ORT_WORKLOADS + TEXT_WORKLOADS:
            sums = assets.parse_sums((ROOT / "workloads" / w / "model.sha256").read_text())
            self.assertTrue(sums, w)
            for name, digest in sums.items():
                archive = name.split("::", 1)[0]   # '<archive>::<member>' pins a file inside an archive
                self.assertIn(archive, assets.URLS, f"{w}: {name} has no URL")
                if "::" in name:
                    self.assertIn(archive, sums, f"{w}: member {name} pinned without its archive")
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_urls_are_pinned_to_commits_or_release_tags(self):
        for name, url in assets.URLS.items():
            if "/releases/download/" in url:   # a release asset: the tag is the pin (and the sha256 pins the bytes)
                self.assertRegex(url, r"^https://github\.com/[^/]+/[^/]+/releases/download/[^/]*\d[^/]*/", name)
            elif url.startswith("https://registry.npmjs.org/"):   # npm versions are immutable
                self.assertRegex(url, r"/-/[^/]+-\d+\.\d+\.\d+[^/]*\.tgz$", name)
            elif url.startswith("https://files.pythonhosted.org/packages/"):   # content-addressed path of one release file
                self.assertRegex(url, r"^https://files\.pythonhosted\.org/packages/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]{60}/[^/]+\.whl$", name)
            else:
                self.assertRegex(url, r"^https://(raw|media)\.githubusercontent\.com/", name)
                self.assertRegex(url, r"[0-9a-f]{40}", name)   # a commit, never a branch

    def test_same_file_has_the_same_sum_everywhere(self):
        seen = {}
        for w in ORT_WORKLOADS + TEXT_WORKLOADS:
            for n, d in assets.parse_sums((ROOT / "workloads" / w / "model.sha256").read_text()).items():
                self.assertEqual(seen.setdefault(n, d), d, n)


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

    def test_bench_refuses_cpu(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "ort_tasks.py"), "vad", "--bench"],
                           env=dict(os.environ, PW_TARGET="cpu"), capture_output=True, text=True)
        self.assertEqual(p.returncode, 77)

    def test_bad_usage(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "ort_tasks.py"), "nope"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__":
    unittest.main()
