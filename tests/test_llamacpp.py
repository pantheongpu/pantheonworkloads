"""Tests for the llama.cpp workloads' helpers. No network, no GPU, no llama.cpp build.

    python3 -m unittest discover -s tests -v
"""
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
LC = ROOT / "tools" / "llamacpp"
sys.path.insert(0, str(LC))
sys.path.insert(0, str(ROOT / "tools"))
import bench_parse  # noqa: E402
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

WORKLOADS = ["llamacpp-smollm2-135m", "llamacpp-bench-smollm2-135m", "llamacpp-bench-mistral-7b-v03"]

# The shape llama-bench -o json prints (field names from tools/llama-bench/llama-bench.cpp).
SAMPLE = [
    {"model_type": "x", "n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ns": 1, "avg_ts": 4000.123, "stddev_ts": 1.0},
    {"model_type": "x", "n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ns": 1, "avg_ts": 250.5, "stddev_ts": 0.5},
]


class BenchParse(unittest.TestCase):
    def test_pp_and_tg(self):
        self.assertEqual(bench_parse.metrics(SAMPLE),
                         {"pp512_tokens_per_s": 4000.12, "tg128_tokens_per_s": 250.5})

    def test_log_lines_before_json_are_ignored(self):
        text = "load_backend: loaded CPU\n" + json.dumps(SAMPLE)
        self.assertEqual(set(bench_parse.parse(text)), {"pp512_tokens_per_s", "tg128_tokens_per_s"})

    def test_depth_and_combined_tests_get_distinct_keys(self):
        rows = [{"n_prompt": 0, "n_gen": 32, "n_depth": 1024, "avg_ts": 1.0},
                {"n_prompt": 64, "n_gen": 32, "avg_ts": 2.0}]
        self.assertEqual(bench_parse.metrics(rows), {"tg32_d1024_tokens_per_s": 1.0, "pp64_tg32_tokens_per_s": 2.0})

    def test_empty_or_missing_json_is_an_error(self):
        with self.assertRaises(ValueError):
            bench_parse.metrics([])
        with self.assertRaises(ValueError):
            bench_parse.parse("no json here")


class Manifests(unittest.TestCase):
    def test_valid_and_licence_never_silently_asserted(self):
        for name in WORKLOADS:
            errors, warnings = validate.check(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(errors, [], name)
            m = validate.load(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(m["runtime"], "llama.cpp")
            # A licence is either read from the model's card (not the case while huggingface.co is
            # unreachable) or flagged: either way validate must say so when it is unverified.
            if str(m["model"]["licence"]).startswith("UNVERIFIED"):
                self.assertTrue(any("UNVERIFIED" in w for w in warnings), name)

    def test_benchmarks_are_real_targets_only(self):
        for name in WORKLOADS[1:]:
            m = validate.load(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(m["kind"], "benchmark")
            self.assertFalse(any(t.startswith("sim:") for t in m["targets"]), name)

    def test_unverified_licence_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "demo"
            d.mkdir()
            (d / "run.sh").write_text("#!/bin/sh\n")
            (d / "manifest.yaml").write_text(
                'name: demo\ndescription: d\nkind: benchmark\nruntime: llama.cpp\ntargets: [gpu]\n'
                'model: {id: m, revision: abc, licence: "UNVERIFIED (x)", source_url: "https://e.org"}\n')
            errors, warnings = validate.check(d / "manifest.yaml")
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("UNVERIFIED", warnings[0])

    def test_pinned_runtime_version_is_recorded(self):
        m = validate.load(ROOT / "workloads" / WORKLOADS[1] / "manifest.yaml")
        with mock.patch.object(pw, "detect_device", return_value=("X", None)):
            env = pw.collect_env(m, "gpu", None)
        self.assertIn("b11447", env["runtime_version"])


class Scripts(unittest.TestCase):
    def test_shell_syntax(self):
        files = [LC / "build.sh", LC / "common.sh", LC / "bench.sh"] + [ROOT / "workloads" / n / "run.sh" for n in WORKLOADS]
        for f in files:
            p = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, f"{f}: {p.stderr}")

    def test_pin_is_a_full_commit(self):
        text = (LC / "pin.env").read_text()
        commit = [l.split("=", 1)[1] for l in text.splitlines() if l.startswith("LLAMACPP_PINNED_COMMIT=")][0]
        self.assertRegex(commit, r"^[0-9a-f]{40}$")

    def run_workload(self, name, target, **env):
        """Run a workload's run.sh in an empty environment: it must skip (77), never fail."""
        with tempfile.TemporaryDirectory() as tmp:
            e = {"PATH": os.environ["PATH"], "HOME": tmp, "PW_TARGET": target, "PW_OUT": tmp,
                 "PW_WORKLOAD_DIR": str(ROOT / "workloads" / name), "PW_CACHE": tmp + "/cache",
                 "PW_LLAMACPP_NO_BUILD": "1"}
            e.update(env)
            return subprocess.run(["bash", str(ROOT / "workloads" / name / "run.sh")], cwd=ROOT, env=e,
                                  capture_output=True, text=True, timeout=60)

    def test_skips_without_a_binary(self):
        for name in WORKLOADS:
            p = self.run_workload(name, "cpu")
            self.assertEqual(p.returncode, 77, (name, p.stdout, p.stderr))
            self.assertIn("SKIP", p.stdout)

    def test_gpu_target_without_a_gpu_skips(self):
        p = self.run_workload(WORKLOADS[1], "gpu", PATH="/usr/bin:/bin")
        self.assertEqual(p.returncode, 77, (p.stdout, p.stderr))

    def test_unsupported_target_skips(self):
        self.assertEqual(self.run_workload(WORKLOADS[0], "tpu").returncode, 77)

    def test_fake_binaries_drive_the_whole_contract(self):
        """A stub llama-completion and llama-bench stand in for the real ones."""
        with tempfile.TemporaryDirectory() as tmp:
            bindir = pathlib.Path(tmp) / "prefix" / "bin"
            bindir.mkdir(parents=True)
            (bindir / "llama-completion").write_text("#!/bin/sh\nprintf ' Paris'\n")
            (bindir / "llama-bench").write_text("#!/bin/sh\nprintf 'log line\\n%s\\n' '" + json.dumps(SAMPLE) + "'\n")
            for f in bindir.iterdir():
                f.chmod(0o755)
            model = pathlib.Path(tmp) / "m.gguf"
            model.write_bytes(b"not a model")
            env = {"PW_LLAMACPP_BIN_DIR": str(bindir), "PW_MODEL_FILE": str(model)}
            p = self.run_workload(WORKLOADS[0], "cpu", **env)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout.splitlines()[-1])["output"], " Paris")
            p = self.run_workload(WORKLOADS[1], "cpu", **env)
            self.assertEqual(p.returncode, 0, p.stderr)
            rec = json.loads(p.stdout.splitlines()[-1])
            self.assertEqual(rec["metrics"]["tg128_tokens_per_s"], 250.5)
            # a wrong checksum is a failure, not a skip
            p = self.run_workload(WORKLOADS[1], "cpu", PW_MODEL_SHA256="0" * 64, **env)
            self.assertEqual(p.returncode, 1)
            # benchmarks refuse simulated targets
            p = self.run_workload(WORKLOADS[1], "sim:nvidia/h100", PANTHEONSIM_DIR=tmp, **env)
            self.assertEqual(p.returncode, 77)


if __name__ == "__main__":
    unittest.main()
