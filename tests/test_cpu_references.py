"""The CPU-references workflow (.github/workflows/cpu-references.yml) parses and names real workloads."""
import pathlib
import re
import sys
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

WORKFLOW = ROOT / ".github" / "workflows" / "cpu-references.yml"


class CpuReferencesWorkflow(unittest.TestCase):
    def setUp(self):
        self.text = WORKFLOW.read_text()
        self.doc = yaml.safe_load(self.text)

    def test_parses_and_has_the_triggers(self):
        triggers = self.doc.get(True) or self.doc.get("on")
        self.assertIn("workflow_dispatch", triggers)
        self.assertIn("schedule", triggers)
        self.assertIn("tools/cpu-torch-pip.sh", triggers["pull_request"]["paths"])

    def test_runs_only_existing_functional_cpu_workloads(self):
        run = re.search(r"bin/pw run (.*?) --target cpu", self.text, re.S).group(1).replace("\\\n", " ").split()
        self.assertGreaterEqual(len(run), 10)
        for name in run:
            m = validate.load(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(m["kind"], "functional", name)
            self.assertIn("cpu", m["targets"], name)
            self.assertTrue((ROOT / "workloads" / name / "reference.json").exists(), name)

    def test_job_is_hosted_and_bounded(self):
        job = self.doc["jobs"]["cpu"]
        self.assertEqual(job["runs-on"], "ubuntu-24.04")
        self.assertLessEqual(job["timeout-minutes"], 120)


if __name__ == "__main__":
    unittest.main()
