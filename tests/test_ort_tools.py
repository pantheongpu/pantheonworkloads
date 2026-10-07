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

# functional workload -> the ort_tasks.py task its run.sh calls (the vision ones live in tools/ort_vision.py)
VISION = {
    "yunet-face-detection": "yunet", "sface-face-embedding": "sface", "pphumanseg-person-segmentation": "pphumanseg",
    "nanodet-object-detection": "nanodet", "yolox-object-detection": "yolox", "onnx-zoo-ssd-mobilenetv1": "ssdmobilenet",
    "crnn-text-recognition": "crnn", "onnx-zoo-shufflenet-v2": "shufflenet", "onnx-zoo-efficientnet-lite4": "efficientnet",
}
PERMISSIVE = {"MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0"}
ORT_WORKLOADS = ["silero-vad-onnx", "silero-vad-onnx-bench", "ppocr-rapidocr", "ppocr-rapidocr-bench",
                 "onnx-zoo-mnist", "onnx-zoo-mobilenetv2", "onnx-zoo-mobilenetv2-bench", "spacy-en-core-web-sm", "spacy-en-core-web-sm-bench"]
ORT_WORKLOADS += [w + suffix for w in VISION for suffix in ("", "-bench")]


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
                self.assertIn(name, assets.URLS, f"{w}: {name} has no URL")
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_urls_are_pinned_to_commits_or_release_tags(self):
        for name, url in assets.URLS.items():
            if "/releases/download/" in url:   # a release asset: the tag is the pin (and the sha256 pins the bytes)
                self.assertRegex(url, r"^https://github\.com/[^/]+/[^/]+/releases/download/[^/]*\d[^/]*/", name)
            else:
                self.assertRegex(url, r"^https://(raw|media)\.githubusercontent\.com/", name)
                self.assertRegex(url, r"[0-9a-f]{40}", name)   # a commit, never a branch

    def test_same_file_has_the_same_sum_everywhere(self):
        seen = {}
        for w in ORT_WORKLOADS:
            for n, d in assets.parse_sums((ROOT / "workloads" / w / "model.sha256").read_text()).items():
                self.assertEqual(seen.setdefault(n, d), d, n)


class VisionWorkloads(unittest.TestCase):
    def test_every_vision_task_is_registered_and_called_by_its_workloads(self):
        table = tasks.all_tasks()
        for w, task in VISION.items():
            self.assertIn(task, table, w)
            for suffix, flag in (("", ""), ("-bench", " --bench")):
                last = (ROOT / "workloads" / (w + suffix) / "run.sh").read_text().strip().splitlines()[-1]
                self.assertTrue(last.endswith(f'ort_tasks.py" {task}{flag}'), w + suffix)

    def test_manifests_record_a_permissive_licence_and_a_pinned_revision(self):
        import yaml
        for w in VISION:
            for suffix in ("", "-bench"):
                m = yaml.safe_load((ROOT / "workloads" / (w + suffix) / "manifest.yaml").read_text())
                self.assertIn(m["model"]["licence"], PERMISSIVE, w)
                self.assertRegex(m["model"]["revision"], r"[0-9a-f]{40}.*sha256 [0-9a-f]{64}", w)
                self.assertEqual(m["targets"], ["gpu"] if suffix else ["cpu", "gpu"], w)   # no sim: target is claimed
            self.assertTrue((ROOT / "workloads" / w / "reference.json").exists(), w)

    def test_the_manifest_digest_is_the_pinned_one(self):
        import yaml
        for w in VISION:
            m = yaml.safe_load((ROOT / "workloads" / w / "manifest.yaml").read_text())
            sums = assets.parse_sums((ROOT / "workloads" / w / "model.sha256").read_text())
            (model_file,) = [n for n in sums if n.endswith(".onnx")]
            self.assertIn(sums[model_file], m["model"]["revision"], w)


try:
    import numpy as np
except ImportError:   # the pure-function tests below need numpy
    np = None


@unittest.skipIf(np is None, "numpy not installed")
class VisionPure(unittest.TestCase):
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ort_vision", ROOT / "tools" / "ort_vision.py")
        self.v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.v)

    def test_iou(self):
        self.assertEqual(self.v.iou_xyxy([0, 0, 2, 2], [0, 0, 2, 2]), 1.0)
        self.assertEqual(self.v.iou_xyxy([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)
        self.assertAlmostEqual(self.v.iou_xyxy([0, 0, 2, 2], [1, 0, 3, 2]), 1 / 3)
        self.assertEqual(self.v.iou_xyxy([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_nms_keeps_the_best_of_an_overlapping_pair(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        self.assertEqual(self.v.nms(boxes, [0.6, 0.9, 0.5], 0.5), [1, 2])
        self.assertEqual(self.v.nms(boxes, [0.6, 0.9, 0.5], 0.95), [1, 0, 2])   # threshold above their IoU keeps both

    def test_nms_is_per_class_when_classes_are_given(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        self.assertEqual(self.v.nms(boxes, [0.9, 0.8], 0.5), [0])
        self.assertEqual(self.v.nms(boxes, [0.9, 0.8], 0.5, classes=[1, 2]), [0, 1])

    def test_nms_ties_go_to_the_lower_index(self):
        self.assertEqual(self.v.nms([[0, 0, 1, 1], [0, 0, 1, 1]], [0.5, 0.5], 0.5), [0])

    def test_similarity_transform_recovers_a_known_one(self):
        src = np.array([[0, 0], [10, 0], [10, 5], [0, 5], [4, 2]], dtype=float)
        theta, scale, shift = 0.3, 1.7, np.array([5.0, -3.0])
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dst = scale * src @ rot.T + shift
        m = self.v.similarity_transform(src, dst)
        np.testing.assert_allclose(src @ m[:, :2].T + m[:, 2], dst, atol=1e-9)

    def test_similarity_transform_never_reflects(self):
        src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        m = self.v.similarity_transform(src, src * [-1, 1])   # a mirror image cannot be matched by a rotation
        self.assertGreaterEqual(np.linalg.det(m[:, :2]), 0)

    def test_ctc_greedy(self):
        def onehot(ids, n=37):
            p = np.full((len(ids), n), 0.01)
            p[np.arange(len(ids)), ids] = 0.9
            return p
        # blank, h(18), h(18), blank, h(18), i(19), i(19): the repeat is collapsed, the one after a blank is kept
        text, conf = self.v.ctc_greedy(onehot([0, 18, 18, 0, 18, 19, 19]))
        self.assertEqual(text, "hhi")
        self.assertAlmostEqual(conf, 0.9)
        self.assertEqual(self.v.ctc_greedy(onehot([0, 0, 0])), ("", 0.0))
        self.assertEqual(self.v.ctc_greedy(onehot([1, 2, 10, 36]))[0], "019z")   # index i is CRNN_CHARSET[i - 1]

    def test_softmax_sums_to_one_and_is_stable(self):
        p = self.v.softmax(np.array([[1000.0, 1001.0, 999.0]]))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertEqual(int(p.argmax()), 1)

    def test_tf91_names(self):
        self.assertEqual([self.v.tf91_name(i) for i in (1, 17, 47, 67, 90)], ["person", "cat", "cup", "dining_table", "toothbrush"])
        self.assertEqual(self.v.tf91_name(12), "class12")   # an id the label map leaves out
        self.assertEqual(len(self.v.COCO80), 80)

    def test_letterbox_topleft(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV not installed")
        img = np.full((50, 100, 3), 7, dtype="uint8")
        canvas, ratio = self.v.letterbox_topleft(img, 200, 114.0)
        self.assertEqual(ratio, 2.0)
        self.assertEqual(canvas.shape, (200, 200, 3))
        self.assertTrue((canvas[:100] == 7).all() and (canvas[100:] == 114).all())

    def test_is_probability(self):
        self.assertTrue(tasks.is_probability([0.25, 0.75, 0.0]))
        self.assertFalse(tasks.is_probability([1.0, 2.0, -3.0]))
        self.assertFalse(tasks.is_probability([0.2, 0.2]))

    def test_the_astronaut_landmarks_align_onto_the_template(self):
        m = self.v.similarity_transform(self.v.ASTRONAUT_FACE_LANDMARKS, self.v.ARCFACE_112)
        moved = np.array(self.v.ASTRONAUT_FACE_LANDMARKS) @ m[:, :2].T + m[:, 2]
        self.assertLess(np.abs(moved - np.array(self.v.ARCFACE_112)).max(), 3.0)   # a real face: within 3 px of the template


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
