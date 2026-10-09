"""Tests for tools/versions.py and for the tools that report `versions` in their result line (copied by bin/pw
into bench records as environment.runtime_versions). No network, no onnxruntime, no model.

    python3 -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import unittest
from importlib import metadata
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ort_tasks as ort  # noqa: E402
import versions  # noqa: E402

FAKE = {"onnxruntime-gpu": "1.30.0", "numpy": "2.5.3", "spacy": "3.8.16"}


def fake_version(name):
    if name in FAKE:
        return FAKE[name]
    raise metadata.PackageNotFoundError(name)


class Collect(unittest.TestCase):
    def test_installed_package(self):
        self.assertEqual(versions.installed("pip"), metadata.version("pip"))

    def test_missing_package_is_none(self):
        self.assertIsNone(versions.installed("no-such-distribution-pw"))

    def test_only_installed_packages_are_reported(self):
        with mock.patch.object(metadata, "version", fake_version):
            self.assertEqual(versions.collect(versions.ORT), FAKE)
            self.assertEqual(versions.collect(("numpy", "cupy-cuda12x")), {"numpy": "2.5.3"})

    def test_extra_entries_are_added_and_empty_ones_dropped(self):
        with mock.patch.object(metadata, "version", fake_version):
            self.assertEqual(versions.collect((), {"whisper.cpp": "v1.9.5", "other": None, "blank": ""}), {"whisper.cpp": "v1.9.5"})

    def test_nothing_installed_is_an_empty_dict(self):
        self.assertEqual(versions.collect(("no-such-distribution-pw",)), {})

    def test_lists_name_the_distributions_the_workloads_use(self):
        for name in ("onnxruntime", "onnxruntime-gpu", "numpy"):
            self.assertIn(name, versions.ORT)
        for name in ("spacy", "numpy"):
            self.assertIn(name, versions.SPACY)


class ResultLines(unittest.TestCase):
    def run_main(self, module, argv):
        out = io.StringIO()
        with mock.patch.object(metadata, "version", fake_version), mock.patch.dict(os.environ, {"PW_TARGET": "gpu"}), \
                contextlib.redirect_stdout(out):
            rc = module.main(argv)
        self.assertEqual(rc, 0)
        return json.loads(out.getvalue().splitlines()[-1])

    def test_ort_tasks_result_carries_versions(self):
        with mock.patch.dict(ort.TASKS, {"fake": lambda bench: ({"x": 1}, "d", {"m": 2.0})}):
            rec = self.run_main(ort, ["ort_tasks.py", "fake"])
        self.assertEqual(rec["versions"], FAKE)
        self.assertEqual(set(rec), {"output", "detail", "metrics", "versions"})

    def test_text_tasks_result_carries_versions(self):
        import text_tasks
        with mock.patch.dict(text_tasks.TASKS, {"fake": lambda bench: ({"x": 1}, "d", {})}):
            rec = self.run_main(text_tasks, ["text_tasks.py", "fake"])
        self.assertEqual(rec["versions"], FAKE)

    def test_speech_tasks_use_the_shared_main(self):
        import speech_tasks
        with mock.patch.dict(ort.TASKS, {"fake": lambda bench: ({"x": 1}, "d", {})}):
            rec = self.run_main(speech_tasks, ["speech_tasks.py", "fake"])
        self.assertEqual(rec["versions"], FAKE)

    def test_whisper_transcript_reports_the_tag_only_when_given(self):
        text = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
        script = str(ROOT / "tools" / "whisper_transcript.py")
        env = {k: v for k, v in os.environ.items() if k != "PW_WHISPER_CPP_TAG"}
        p = subprocess.run([sys.executable, "-I", script], env=dict(env, PW_TEXT=text, PW_WHISPER_CPP_TAG="v1.9.5"), capture_output=True, text=True)
        self.assertEqual(json.loads(p.stdout)["versions"], {"whisper.cpp": "v1.9.5"})
        p = subprocess.run([sys.executable, "-I", script], env=dict(env, PW_TEXT=text), capture_output=True, text=True)
        self.assertEqual(json.loads(p.stdout)["versions"], {})

    def test_whisper_runners_pass_the_build_script_tag(self):
        tags = [l for l in (ROOT / "tools" / "build-whisper-cpp.sh").read_text().splitlines() if l.startswith("TAG=")]
        self.assertEqual(len(tags), 1)
        for w in ("whisper-cpp-tiny-en", "whisper-cpp-large-v3"):
            self.assertIn("PW_WHISPER_CPP_TAG=", (ROOT / "workloads" / w / "run.sh").read_text())

    def test_spacy_task_loads_the_helper(self):
        import spacy_task
        self.assertEqual(spacy_task.VERSIONS.SPACY, versions.SPACY)


if __name__ == "__main__":
    unittest.main()
