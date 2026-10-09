"""Tests for the architecture-coverage workloads (arch-*) and tools/torch-cpu-env.sh that need no
network, no GPU and no transformers. The end-to-end test runs the workloads against their recorded
references when PW_PYTHON points at a Python with torch and transformers (tools/torch-cpu-env.sh makes one).

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
_loader = importlib.machinery.SourceFileLoader("pw", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

sys.path.insert(0, str(ROOT / "workloads" / "_shared"))
import arch_cases  # noqa: E402  (imports only the standard library at module level)

ENV_SH = ROOT / "tools" / "torch-cpu-env.sh"
GROUP_WORKLOAD = {"llama-family": "arch-llama-family", "moe": "arch-moe", "ssm": "arch-ssm",
                  "vision": "arch-vision", "encdec": "arch-encdec"}


def run(name, target, **env):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
        if "PANTHEONSIM_DIR" not in env:
            os.environ.pop("PANTHEONSIM_DIR", None)
        return pw.execute(name, target, pathlib.Path(tmp))


def have_stack():
    py = os.environ.get("PW_PYTHON")
    if not py:
        return False
    return subprocess.run([py, "-c", "import torch, transformers"], capture_output=True).returncode == 0


class Cases(unittest.TestCase):
    def test_every_group_has_a_workload_and_cases(self):
        self.assertEqual(sorted(arch_cases.GROUPS), sorted(GROUP_WORKLOAD))
        for group, workload in GROUP_WORKLOAD.items():
            cases = arch_cases.cases(group)
            self.assertTrue(cases, group)
            self.assertIn(f"--group {group}", (ROOT / "workloads" / workload / "run.sh").read_text())
            for name, spec in cases.items():
                self.assertEqual(len(spec), 5, name)

    def test_references_cover_exactly_the_cases(self):
        for group, workload in GROUP_WORKLOAD.items():
            ref = json.loads((ROOT / "workloads" / workload / "reference.json").read_text())
            self.assertEqual(ref["recorded_on"], "cpu")
            self.assertEqual(sorted(ref["output"]), sorted(arch_cases.cases(group)), group)
            for name, entry in ref["output"].items():
                self.assertIsInstance(entry["ids"], str, name)               # ids are exact strings
                self.assertTrue(all(isinstance(v, float) for v in entry["stats"]), name)

    def test_only_filter(self):
        with mock.patch.dict(os.environ, {"PW_ARCH_ONLY": "mistral-swa,nonexistent"}):
            self.assertEqual(list(arch_cases.cases("llama-family")), ["mistral-swa"])

    def test_unknown_group_is_an_error(self):
        with self.assertRaises(SystemExit):
            arch_cases.cases("nope")

    def test_ids_str_and_rounding(self):
        class T:
            def flatten(self): return self
            def tolist(self): return [3, 1, 4]
        self.assertEqual(arch_cases.ids_str(T()), "3 1 4")
        self.assertEqual(arch_cases.r6(1.23456789), 1.234568)

    def test_the_sliding_window_case_is_shorter_than_its_prompt(self):
        # the Mistral case only tests the window mask if the window is shorter than the 11-token prompt
        self.assertLess(arch_cases.cases("llama-family")["mistral-swa"][3]["sliding_window"], 11)

    def test_gqa_cases_really_group_heads(self):
        c = arch_cases.cases("llama-family")
        for name in ("llama-gqa4", "mistral-swa", "qwen2-gqa3"):
            kw = c[name][3]
            self.assertGreater(kw["num_attention_heads"], kw["num_key_value_heads"], name)

    def test_tolerances_in_manifests_match_the_engine(self):
        for workload in GROUP_WORKLOAD.values():
            m = pw.validate.load(ROOT / "workloads" / workload / "manifest.yaml")
            self.assertEqual(m["tolerance"], {"abs": 1.0e-3, "rel": 1.0e-3})
            self.assertEqual(m["model"]["licence"].split()[0], "Apache-2.0")
            self.assertIn("NOT THE QUALITY", m["notes"])


class Tolerance(unittest.TestCase):
    def test_recorded_reference_policy(self):
        m = pw.validate.load(ROOT / "workloads" / "arch-moe" / "manifest.yaml")
        ref = json.loads((ROOT / "workloads" / "arch-moe" / "reference.json").read_text())["output"]
        good = json.loads(json.dumps(ref))
        good["mixtral-top2of8"]["stats"][1] += 1e-4
        self.assertTrue(pw.compare(m, good, ref))
        bad_float = json.loads(json.dumps(ref))
        bad_float["mixtral-top2of8"]["stats"][1] += 0.1
        self.assertFalse(pw.compare(m, bad_float, ref))
        bad_id = json.loads(json.dumps(ref))
        bad_id["mixtral-top3of5"]["ids"] += " 1"
        self.assertFalse(pw.compare(m, bad_id, ref))


class Skips(unittest.TestCase):
    def test_no_python_is_skip(self):
        for name in list(GROUP_WORKLOAD.values()) + ["arch-bench"]:
            result, _, detail, _ = run(name, "cpu", PW_PYTHON="/nonexistent/python")
            self.assertEqual(result, "SKIP", name)
            self.assertIn("not found", detail)

    def test_sim_targets_need_pantheonsim(self):
        result, _, detail, _ = run("arch-moe", "sim:amd/mi300x")
        self.assertEqual((result, "PANTHEONSIM_DIR" in detail), ("SKIP", True))

    def test_benchmark_is_not_a_simulator_workload(self):
        self.assertEqual(run("arch-bench", "sim:nvidia/h100")[0], "SKIP")
        self.assertEqual(run("arch-bench", "sim:amd/mi300x")[0], "SKIP")

    def test_torch_without_transformers_is_skip_with_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "torch").mkdir()
            (pathlib.Path(tmp) / "torch" / "__init__.py").write_text("__version__ = 'fake'\n")
            # a transformers that is not installed: block it even if the interpreter has one
            (pathlib.Path(tmp) / "transformers.py").write_text("raise ImportError('blocked for the test')\n")
            result, _, detail, _ = run("arch-vision", "cpu", PW_PYTHON=sys.executable, PYTHONPATH=tmp)
        self.assertEqual(result, "SKIP")
        self.assertIn("transformers is not installed", detail)


class EnvHelper(unittest.TestCase):
    def helper(self, prefix, *args, **env):
        e = dict(os.environ, PW_TORCH_CPU_PREFIX=str(prefix), **env)
        return subprocess.run(["bash", str(ENV_SH), *args], capture_output=True, text=True, env=e)

    def test_check_fails_when_there_is_no_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.helper(pathlib.Path(tmp) / "env", "--check")
            self.assertEqual((p.returncode, p.stdout), (1, ""))

    def test_existing_environment_is_reused_and_only_the_path_is_printed(self):
        with tempfile.TemporaryDirectory() as tmp:
            py = pathlib.Path(tmp) / "env" / "bin" / "python"
            py.parent.mkdir(parents=True)
            py.write_text("#!/bin/sh\nexit 0\n")        # `import torch, transformers` "succeeds"
            py.chmod(0o755)
            p = self.helper(pathlib.Path(tmp) / "env")
            self.assertEqual((p.returncode, p.stdout.strip()), (0, str(py)))

    def test_unbuildable_environment_exits_77(self):
        with tempfile.TemporaryDirectory() as tmp:
            mm = pathlib.Path(tmp) / "micromamba"
            mm.write_text("#!/bin/sh\necho solver failed >&2\nexit 1\n")
            mm.chmod(0o755)
            p = self.helper(pathlib.Path(tmp) / "env", PW_TORCH_ROUTE="conda", PW_MICROMAMBA=str(mm), PW_MAMBA_ROOT=str(pathlib.Path(tmp) / "root"))
            self.assertEqual(p.returncode, 77)
            self.assertEqual(p.stdout, "")
            self.assertIn("SKIP", p.stderr)

    def test_unreachable_pip_index_exits_77_and_names_the_conda_escape_hatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.helper(pathlib.Path(tmp) / "env", PW_CPU_TORCH_INDEX="http://127.0.0.1:9/whl/cpu", PIP_RETRIES="0")
            self.assertEqual(p.returncode, 77)
            self.assertEqual(p.stdout, "")
            self.assertIn("PW_TORCH_ROUTE=conda", p.stderr)

    def test_conda_route_pins_conda_forge_only(self):
        text = ENV_SH.read_text()
        self.assertIn("--override-channels -c conda-forge", text)
        self.assertIn("https://conda.anaconda.org/conda-forge/", text)


@unittest.skipUnless(have_stack(), "needs PW_PYTHON with torch and transformers (tools/torch-cpu-env.sh)")
class EndToEnd(unittest.TestCase):
    def test_groups_pass_on_cpu_against_the_recorded_references(self):
        for workload in GROUP_WORKLOAD.values():
            result, _, detail, _ = run(workload, "cpu", OMP_NUM_THREADS="2")
            self.assertEqual(result, "PASS", (workload, detail))

    def test_bench_runs_on_cpu_and_reports_metrics(self):
        result, _, _, rec = run("arch-bench", "cpu", OMP_NUM_THREADS="2", PW_ARCH_BENCH_REPS="1")
        self.assertEqual(result, "PASS")
        self.assertEqual(sorted(rec["metrics"]), ["llama_small_decode_tokens_per_s", "llama_small_prefill_tokens_per_s"])

    def test_a_perturbed_architecture_changes_the_ids(self):
        # the workload must notice a changed head layout: run one case with another seed offset
        with mock.patch.dict(os.environ, {"PW_ARCH_ONLY": "vit-p8", "PW_ARCH_SEED_OFFSET": "7"}):
            p = subprocess.run([os.environ["PW_PYTHON"], str(ROOT / "workloads/_shared/arch_cases.py"),
                                "--mode", "functional", "--group", "vision"],
                               capture_output=True, text=True, env=dict(os.environ, PW_TARGET="cpu"))
        self.assertEqual(p.returncode, 0, p.stderr[-500:])
        out = json.loads(p.stdout.strip().splitlines()[-1])["output"]["vit-p8"]
        ref = json.loads((ROOT / "workloads/arch-vision/reference.json").read_text())["output"]["vit-p8"]
        self.assertNotEqual(out["ids"], ref["ids"])


if __name__ == "__main__":
    unittest.main()
