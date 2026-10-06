"""Tests for the vLLM and GPT-training workloads that need no GPU, no vLLM and no network.

The vLLM workloads cannot run here, so what is tested is their glue: they exit 77 with a reason
when something is missing, and, with a *fake* torch and vllm on PYTHONPATH (written below, which
say nothing about real vLLM), the benchmark's parsing and the functional workload's output
plumbing. The training workloads run for real when a Python with torch is available (PW_PYTHON,
or python3); otherwise those tests are skipped.

    python3 -m unittest discover -s tests -v
"""
import importlib.machinery
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
_loader = importlib.machinery.SourceFileLoader("pw", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

FAKE = {
    "torch/__init__.py": "__version__ = 'fake'\nclass cuda:\n    is_available = staticmethod(lambda: True)\n",
    "vllm/__init__.py": (
        "__version__ = 'fake'\n"
        "class _Out:\n    text = ' the capital'\n    token_ids = [1, 2]\n"
        "class _Res:\n    outputs = [_Out()]\n"
        "class LLM:\n    def __init__(self, **kw): self.kw = kw\n    def generate(self, prompts, params): return [_Res()]\n"
        "class SamplingParams:\n    def __init__(self, **kw): pass\n"
    ),
    "vllm/entrypoints/__init__.py": "",
    "vllm/entrypoints/cli/__init__.py": "",
    "vllm/entrypoints/cli/main.py": (
        "import json, sys\n"
        "a = sys.argv[1:]\n"
        "assert a[:2] == ['bench', 'throughput'], a\n"
        "assert '--random-input-len' in a and '--random-output-len' in a, a\n"
        "json.dump({'elapsed_time': 2.0, 'num_requests': 256, 'total_num_tokens': 65536,\n"
        "           'requests_per_second': 128.0, 'tokens_per_second': 32768.0}, open(a[a.index('--output-json') + 1], 'w'))\n"
        "print('Throughput: 128.00 requests/s, 32768.00 total tokens/s, 16384.00 output tokens/s')\n"
    ),
}


def run(name, target, **env):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
        for v in ("PANTHEONSIM_DIR", "VGPU_BUILD_DIR"):
            os.environ.pop(v, None) if v not in env else None
        return pw.execute(name, target, pathlib.Path(tmp))


def fake_env(tmp, with_vllm=True):
    root = pathlib.Path(tmp) / "fake"
    for rel, text in FAKE.items():
        if not with_vllm and rel.startswith("vllm"):
            continue
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return {"PYTHONPATH": str(root), "PW_PYTHON": sys.executable}


class VllmGlue(unittest.TestCase):
    def test_cpu_and_sim_nvidia_are_not_targets(self):
        for name in ("vllm-greedy-smollm2-135m", "vllm-bench-throughput"):
            for target in ("cpu", "sim:nvidia/h100"):
                self.assertEqual(run(name, target)[0], "SKIP", (name, target))

    def test_benchmark_is_not_a_simulator_workload(self):
        self.assertEqual(run("vllm-bench-throughput", "sim:amd/mi300x")[0], "SKIP")

    def test_no_torch_is_skip(self):
        result, _, detail, _ = run("vllm-bench-throughput", "gpu", PW_PYTHON="/nonexistent/python")
        self.assertEqual(result, "SKIP")
        self.assertIn("not found", detail)

    def test_no_vllm_is_skip_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("vllm-greedy-smollm2-135m", "vllm-bench-throughput"):
                result, _, detail, _ = run(name, "gpu", **fake_env(tmp, with_vllm=False))
                self.assertEqual(result, "SKIP", name)
                self.assertIn("vllm", detail)

    def test_sim_amd_needs_pantheonsim(self):
        result, _, detail, _ = run("vllm-greedy-smollm2-135m", "sim:amd/mi300x")
        self.assertEqual(result, "SKIP")
        self.assertIn("PANTHEONSIM_DIR", detail)

    def test_rx6900xt_is_skipped_with_the_reason(self):
        result, _, detail, _ = run("vllm-greedy-smollm2-135m", "sim:amd/rx6900xt")
        self.assertEqual(result, "SKIP")
        self.assertIn("gfx1030", detail)

    def test_benchmark_parses_the_tools_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, detail, rec = run("vllm-bench-throughput", "gpu", **fake_env(tmp))
        self.assertEqual(result, "PASS", detail)
        self.assertEqual(rec["metrics"], {"requests_per_s": 128.0, "total_tokens_per_s": 32768.0,
                                          "output_tokens_per_s": 16384.0})

    def test_functional_output_is_the_generated_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, detail, rec = run("vllm-greedy-smollm2-135m", "gpu", **fake_env(tmp))
        self.assertEqual(rec["output"], " the capital")
        self.assertEqual(rec["metrics"], {})
        self.assertEqual(result, "SKIP")        # no reference.json is recorded
        self.assertIn("no reference", detail)


HAVE_TORCH = importlib.util.find_spec("torch") is not None or bool(os.environ.get("PW_PYTHON"))


class GptTrain(unittest.TestCase):
    def test_no_python_is_skip(self):
        for name in ("gpt-train-fp32", "gpt-train-bf16", "gpt-train-bench"):
            result, _, detail, _ = run(name, "cpu", PW_PYTHON="/nonexistent/python")
            self.assertEqual(result, "SKIP", name)

    def test_sim_targets_need_pantheonsim(self):
        for target in ("sim:nvidia/h100", "sim:amd/mi300x"):
            result, _, detail, _ = run("gpt-train-fp32", target)
            self.assertEqual(result, "SKIP", target)
            self.assertIn("PANTHEONSIM_DIR", detail)

    def test_vgpu_build_dir_names_the_build_directly_and_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in ({}, {"PANTHEONSIM_DIR": "/nonexistent/checkout"}):
                result, _, detail, _ = run("gpt-train-fp32", "sim:nvidia/h100", VGPU_BUILD_DIR=tmp, **extra)
                self.assertEqual(result, "SKIP")
                self.assertIn(f"no CUDA 13 runtime shim in {tmp}/shim", detail)

    def test_benchmark_is_not_a_simulator_workload(self):
        self.assertEqual(run("gpt-train-bench", "sim:nvidia/h100")[0], "SKIP")

    def test_references_are_recorded_and_match_their_tolerance_policy(self):
        for name, tol in (("gpt-train-fp32", 1e-3), ("gpt-train-bf16", 2e-2)):
            m = pw.validate.load(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(m["tolerance"]["abs"], tol)
            self.assertEqual(len(pw.json.loads((ROOT / "workloads" / name / "reference.json").read_text())["output"]), 20)

    @unittest.skipUnless(HAVE_TORCH, "needs a Python with torch (PW_PYTHON)")
    def test_fp32_and_bf16_pass_on_cpu_against_the_recorded_references(self):
        for name in ("gpt-train-fp32", "gpt-train-bf16"):
            result, _, detail, _ = run(name, "cpu")
            if result == "SKIP" and "import torch" in detail:
                self.skipTest(detail)
            self.assertEqual(result, "PASS", detail)

    def test_tolerance_catches_a_wrong_loss(self):
        m = pw.validate.load(ROOT / "workloads" / "gpt-train-fp32" / "manifest.yaml")
        ref = pw.json.loads((ROOT / "workloads" / "gpt-train-fp32" / "reference.json").read_text())["output"]
        self.assertTrue(pw.compare(m, [v + 5e-4 for v in ref], ref))
        self.assertFalse(pw.compare(m, [v + 5e-3 for v in ref], ref))


if __name__ == "__main__":
    unittest.main()
