"""Tests for the whisper.cpp helpers in tools/. No network, no model, no build.

    python3 -m unittest discover -s tests -v
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import whisper_bench_parse as bench  # noqa: E402
import whisper_transcript as tr  # noqa: E402

BENCH_LOG = """
whisper_print_timings:     load time =   100.00 ms
whisper_print_timings:     fallbacks =   0 p /   0 h
whisper_print_timings:      mel time =     0.00 ms
whisper_print_timings:   sample time =     0.00 ms /     1 runs (     0.00 ms per run)
whisper_print_timings:   encode time =  1234.50 ms /     3 runs (   411.50 ms per run)
whisper_print_timings:   decode time =   200.00 ms /   256 runs (     0.78 ms per run)
whisper_print_timings:   batchd time =     0.00 ms /     1 runs (     0.00 ms per run)
whisper_print_timings:   prompt time =     0.00 ms /     1 runs (     0.00 ms per run)
"""


class Transcript(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(tr.norm(" And so, my fellow Americans:\n ask not..."), "and so my fellow americans ask not")

    def test_wer(self):
        self.assertEqual(tr.wer("a b c d", "a b c d"), 0.0)
        self.assertEqual(tr.wer("a b c d", "a x c d"), 0.25)
        self.assertEqual(tr.wer("a b c d", "a b c"), 0.25)
        self.assertEqual(tr.wer("a b", "a b c d"), 1.0)

    def test_truth_is_clean(self):
        self.assertEqual(tr.norm(tr.TRUTH), tr.TRUTH)

    def run_main(self, text, limit=None):
        env = dict(os.environ, PW_TEXT=text)
        if limit:
            env["PW_WHISPER_WER_MAX"] = limit
        return subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "whisper_transcript.py")],
                              env=env, capture_output=True, text=True)

    def test_main_pass_and_fail(self):
        ok = self.run_main(" And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(json.loads(ok.stdout.splitlines()[-1])["output"], tr.TRUTH)
        bad = self.run_main("completely different words here")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("word error rate", bad.stderr)


class BenchParse(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(bench.parse(BENCH_LOG), {"encode_ms_per_run": 411.5, "decode_ms_per_run": 0.78})

    def test_missing_encode_is_an_error(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "whisper_bench_parse.py")],
                           input="nothing useful", capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)


class Scripts(unittest.TestCase):
    """The scripts must say SKIP (exit 77), never fail, when they cannot do their job."""

    def sh(self, *cmd, **env):
        with tempfile.TemporaryDirectory() as cache:
            e = dict(os.environ, PW_CACHE=cache, **env)
            return subprocess.run([str(ROOT / cmd[0]), *cmd[1:]], env=e, capture_output=True, text=True)

    def test_unknown_backend(self):
        p = self.sh("tools/build-whisper-cpp.sh", "tpu")
        self.assertEqual(p.returncode, 77)
        self.assertIn("unknown backend", p.stdout)

    def test_bin_dir_without_binaries(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.sh("tools/build-whisper-cpp.sh", "cpu", PW_WHISPER_BIN_DIR=d).returncode, 77)

    def test_model_with_wrong_checksum(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"not a model")
            f.flush()
            p = self.sh("tools/whisper-assets.sh", "model", PW_WHISPER_MODEL=f.name)
        self.assertEqual(p.returncode, 77)
        self.assertIn("sha256", p.stdout)

    def test_workload_skips_without_model(self):
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as bindir:
            for n in ("whisper-cli", "whisper-bench"):
                pathlib.Path(bindir, n).write_text("#!/bin/sh\nexit 1\n")
                pathlib.Path(bindir, n).chmod(0o755)
            env = dict(os.environ, PW_TARGET="cpu", PW_OUT=out, PW_WHISPER_BIN_DIR=bindir,
                       PW_WHISPER_MODEL="/nonexistent")
            p = subprocess.run(["bash", str(ROOT / "workloads/whisper-cpp-tiny-en/run.sh")], env=env,
                               capture_output=True, text=True)
        self.assertEqual(p.returncode, 77, p.stderr)

    def test_bench_refuses_cpu(self):
        env = dict(os.environ, PW_TARGET="cpu", PW_OUT="/tmp")
        p = subprocess.run(["bash", str(ROOT / "workloads/whisper-cpp-bench-tiny-en/run.sh")], env=env,
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 77)


if __name__ == "__main__":
    unittest.main()
