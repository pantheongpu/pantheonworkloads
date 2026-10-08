"""The nightly simulated-GPU workflow (.github/workflows/sim-nightly.yml) parses and names real things.

Framework-free: needs only PyYAML, like the rest of the CI unit-test step.
"""
import pathlib
import re
import sys
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
WORKFLOW = ROOT / ".github" / "workflows" / "sim-nightly.yml"
NVIDIA_PROFILES_DOC = ROOT / "docs" / "sim-validation.md"


class SimNightly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wf = yaml.safe_load(WORKFLOW.read_text())
        # PyYAML reads the key `on` as the boolean True.
        cls.triggers = cls.wf.get("on", cls.wf.get(True))

    def test_triggers(self):
        self.assertIn("schedule", self.triggers)
        self.assertIn("workflow_dispatch", self.triggers)

    def test_jobs_are_bounded_and_cancel_in_progress(self):
        self.assertTrue(self.wf["concurrency"]["cancel-in-progress"])
        for name, job in self.wf["jobs"].items():
            self.assertIn("timeout-minutes", job, name)
            self.assertEqual(job["runs-on"], "ubuntu-24.04", f"{name}: hosted runners only")
        self.assertIs(self.wf["jobs"]["run"]["strategy"]["fail-fast"], False)

    def test_workloads_exist(self):
        names = self.wf["env"]["WORKLOADS"].split()
        self.assertTrue(names)
        for n in names:
            self.assertTrue((ROOT / "workloads" / n / "manifest.yaml").exists(), f"no such workload: {n}")
        # the point of the workflow
        for prefix in ("arch-", "lib-"):
            self.assertTrue(any(n.startswith(prefix) for n in names))

    def test_functional_workloads_only(self):
        """A benchmark has no CPU reference to check against and is not a simulator target."""
        for n in self.wf["env"]["WORKLOADS"].split():
            manifest = yaml.safe_load((ROOT / "workloads" / n / "manifest.yaml").read_text())
            self.assertEqual(manifest.get("kind"), "functional", n)
            self.assertTrue(any(re.fullmatch(p.replace("*", ".*"), "sim:nvidia/h100") for p in manifest.get("targets", [])),
                            f"{n} does not list sim:nvidia/*")

    def test_gpu_profiles(self):
        gpus = self.wf["jobs"]["run"]["strategy"]["matrix"]["gpu"]
        self.assertEqual(len(gpus), len(set(gpus)))
        for g in gpus:
            self.assertRegex(g, r"^[a-z0-9]+$")
            self.assertIn(f"sim:nvidia/{g}", NVIDIA_PROFILES_DOC.read_text(), f"{g} is not a profile the docs know")

    def test_simulator_is_pinned(self):
        self.assertRegex(self.wf["env"]["SIM_REF_PIN"], r"^[0-9a-f]{40}$")
        self.assertIn("torch==", self.wf["env"]["TORCH_CONSTRAINTS"])


if __name__ == "__main__":
    unittest.main()
