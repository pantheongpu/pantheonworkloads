"""Tests for tools/coverage_table.py and `bin/pw coverage`: the README table is generated, never hand-edited.

    python3 -m unittest test_coverage -v
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import coverage_table as coverage  # noqa: E402


def make_repo(tmp):
    """A tiny repository: a functional workload with a reference and a sim PASS, a bench twin, an unpinned one."""
    tmp = pathlib.Path(tmp)
    (tmp / "docs").mkdir()
    (tmp / "workloads" / "lib-x").mkdir(parents=True)
    (tmp / "workloads" / "lib-x" / "manifest.yaml").write_text(
        "name: lib-x\ndescription: d\nkind: functional\nruntime: pytorch\ntargets: [cpu, gpu, 'sim:nvidia/*']\n"
        "compare: exact\nmodel: null\n")
    (tmp / "workloads" / "lib-x" / "reference.json").write_text('{"recorded_on": "cpu", "output": 1}')
    (tmp / "workloads" / "foo-onnx").mkdir()
    (tmp / "workloads" / "foo-onnx" / "manifest.yaml").write_text(
        "name: foo-onnx\ndescription: d\nkind: functional\nruntime: onnxruntime\ntargets: [cpu]\ncompare: exact\n"
        "model:\n  id: a/b\n  revision: null\n  licence: 'UNVERIFIED (card unread)'\n  source_url: https://x\n")
    (tmp / "workloads" / "foo-onnx-bench").mkdir()
    (tmp / "workloads" / "foo-onnx-bench" / "manifest.yaml").write_text(
        "name: foo-onnx-bench\ndescription: d\nkind: benchmark\nruntime: onnxruntime\ntargets: [gpu]\n"
        "model:\n  id: a/b\n  revision: 'sha256 abc'\n  licence: MIT\n  source_url: https://x\n")
    (tmp / "docs" / "sim-validation.md").write_text(
        "| Workload | Target | Result | Detail | Date |\n| --- | --- | --- | --- | --- |\n"
        "| lib-x, foo-onnx | sim:nvidia/{h100, a100} | PASS | ok | d |\n"
        "| foo-onnx | sim:amd/mi300x | SKIP | no | d |\n"
        "| lib-* | sim:amd/mi300x | PASS | ok | d |\n")
    return tmp


class TestRows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = make_repo(self.tmp)

    def by_name(self):
        return {r["name"]: r for r in coverage.rows(self.root)}

    def test_cells_come_from_the_data(self):
        r = self.by_name()
        self.assertEqual(r["lib-x"]["family"], "lib")
        self.assertEqual(r["lib-x"]["reference"], "yes (cpu)")
        self.assertEqual(r["lib-x"]["sim"], "amd x1, nvidia x2")  # brace list expanded, wildcard row matched
        self.assertEqual(r["lib-x"]["pinned"], "n/a")
        self.assertEqual(r["foo-onnx"]["family"], "pretrained")
        self.assertEqual(r["foo-onnx"]["reference"], "no")
        self.assertEqual(r["foo-onnx"]["licence"], "UNVERIFIED")
        self.assertEqual(r["foo-onnx"]["pinned"], "no")
        self.assertEqual(r["foo-onnx"]["sim"], "nvidia x2")  # the SKIP row does not count
        self.assertEqual(r["foo-onnx-bench"]["reference"], "n/a")
        self.assertEqual(r["foo-onnx-bench"]["pinned"], "yes")
        self.assertEqual(r["foo-onnx-bench"]["sim"], "no")

    def test_gpu_validated_needs_a_bench_record_or_a_gpu_pass(self):
        self.assertEqual(self.by_name()["foo-onnx-bench"]["gpu"], "no")
        (self.root / "bench" / "foo-onnx-bench").mkdir(parents=True)
        (self.root / "bench" / "foo-onnx-bench" / "a.json").write_text("{}")
        with open(self.root / "docs" / "sim-validation.md", "a") as f:
            f.write("| lib-x | gpu | PASS | ok | d |\n")
        r = self.by_name()
        self.assertEqual(r["foo-onnx-bench"]["gpu"], "bench")
        self.assertEqual(r["lib-x"]["gpu"], "functional")

    def test_check_detects_drift_and_write_fixes_it(self):
        readme = self.root / "README.md"
        readme.write_text(f"intro\n\n{coverage.BSTART}\n{coverage.BEND}\n\n{coverage.START}\n{coverage.END}\n\ntail\n")
        self.assertTrue(coverage.check(self.root))
        coverage.write(self.root)
        self.assertEqual(coverage.check(self.root), [])
        text = readme.read_text()
        self.assertTrue(text.startswith("intro\n") and text.endswith("\ntail\n"))
        readme.write_text(text.replace("yes (cpu)", "yes (gpu)"))
        self.assertTrue(coverage.check(self.root))

    def write_bench(self, name, device, date, metrics, versions=None):
        d = self.root / "bench" / name
        d.mkdir(parents=True, exist_ok=True)
        env = {"device": device, "runtime_version": "ort 1.0 (PyPI; long text)", "driver": "1", "repo_commit": "abcdef123456"}
        if versions:
            env["runtime_versions"] = versions
        rec = {"schema": "pw-bench/0", "workload": name, "date": date, "environment": env, "repeats": 5,
               "metrics": {k: {"median": v, "values": [v - 1, v, v + 1]} for k, v in metrics.items()}}
        (d / (date + ".json")).write_text(__import__("json").dumps(rec))

    def test_results_use_the_newest_record_per_workload_and_device_and_show_versions(self):
        self.write_bench("foo-onnx-bench", "A10G", "2026-10-01T00:00:00Z", {"images_per_s": 100.0})
        self.write_bench("foo-onnx-bench", "A10G", "2026-10-09T00:00:00Z", {"images_per_s": 2500.0, "latency_ms": 3.2},
                         {"torch": "2.14.1+cu130", "numpy": "2.5"})
        self.write_bench("foo-onnx-bench", "H100", "2026-10-05T00:00:00Z", {"images_per_s": 9000.0})
        compact = coverage.render_bench(self.root)
        self.assertIn("2,500", compact)
        self.assertNotIn("| images_per_s 100", compact)
        self.assertIn("torch 2.14.1+cu130", compact)
        self.assertIn("ort 1.0 | 2026-10-05", compact)              # no versions object: the manifest-style string, cut at ' ('
        full = coverage.render_results(self.root)
        self.assertIn("## foo-onnx-bench on A10G", full)
        self.assertIn("1 older record", full)
        self.assertIn("| latency_ms | 3.2 | 2.2 | 4.2 |", full)

    def test_stale_results_page_is_reported_and_written(self):
        self.write_bench("foo-onnx-bench", "A10G", "2026-10-09T00:00:00Z", {"images_per_s": 1.0})
        (self.root / "README.md").write_text(f"{coverage.BSTART}\n{coverage.BEND}\n{coverage.START}\n{coverage.END}\n")
        self.assertTrue(any("docs/results.md" in p for p in coverage.check(self.root)))
        coverage.write(self.root)
        self.assertEqual(coverage.check(self.root), [])
        self.write_bench("foo-onnx-bench", "A10G", "2026-10-10T00:00:00Z", {"images_per_s": 2.0})
        self.assertTrue(coverage.check(self.root))

    def test_missing_markers_are_reported(self):
        (self.root / "README.md").write_text("no markers here\n")
        self.assertIn("markers are missing", coverage.check(self.root)[0])


class TestRepository(unittest.TestCase):
    def test_every_bench_record_is_covered_by_the_results_page(self):
        results = (ROOT / "docs" / "results.md").read_text()
        for f in (ROOT / "bench").glob("*/*.json"):
            self.assertIn(f"## {f.parent.name} on ", results, f.name)

    def test_every_workload_is_in_exactly_one_row(self):
        names = sorted(p.parent.name for p in (ROOT / "workloads").glob("*/manifest.yaml"))
        self.assertEqual(sorted(r["name"] for r in coverage.rows(ROOT)), names)

    def test_readme_block_matches_generated_output(self):
        self.assertEqual(coverage.check(ROOT), [], "run `bin/pw coverage --write` and commit README.md")

    def test_cli_check_passes(self):
        p = subprocess.run([sys.executable, str(ROOT / "bin" / "pw"), "coverage", "--check"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
