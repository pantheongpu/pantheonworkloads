"""Tests for tools/sim_vs_gpu.py: docs/sim-vs-gpu.md is generated from docs/data/sim-vs-gpu/ and nothing else.

    python3 -m unittest test_sim_vs_gpu -v
"""
import csv
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import sim_vs_gpu as svg  # noqa: E402

DATA = ROOT / "docs" / "data" / "sim-vs-gpu"
FILES = ["real_results.tsv", "sim_results.tsv", "no_counterpart.tsv", "parity_metrics.tsv", "pairs.tsv", "datasets.tsv", "profiles.tsv"]


def rows(name):
    with open(DATA / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def put(root, name, header, body):
    path = pathlib.Path(root) / "docs" / "data" / "sim-vs-gpu"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / name, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(body)


REAL_H = "workload gpu result date stack detail reference wall_seconds commit note provenance".split()
SIM_H = "workload profile scope result seconds seconds_note detail date source host provenance".split()


def make_root(tmp, real_seconds="", a10_seconds=""):
    """Two paired workloads, one real-only on the A10G, one real-only on the L40S."""
    put(tmp, "real_results.tsv", REAL_H, [
        ["w-a", "nvidia-a10g", "PASS", "2026-01-01", "torch", "worst 1e-3", "ref", real_seconds, "c1", "", "doc A"],
        ["w-b", "nvidia-a10g", "PASS", "2026-01-01", "torch", "worst 2e-3", "ref", "", "c1", "", "doc A"],
        ["w-b", "nvidia-a10g", "FAIL", "2025-12-01", "old torch", "bad", "ref", "", "c0", "superseded by the re-run", "doc A"],
        ["w-c", "nvidia-a10g", "PASS", "2026-01-01", "x", "ok", "ref", "", "", "", "doc B"],
        ["w-d", "nvidia-l40s", "PASS", "2026-01-02", "x", "ok", "rec", "", "", "", "doc C"]])
    put(tmp, "sim_results.tsv", SIM_H, [
        ["w-a", "a10", "full", "PASS", a10_seconds, "", "d", "2026-01-05", "nightly", "host", "run 1"],
        ["w-a", "a100", "full", "PASS", "12", "", "d", "2026-01-05", "nightly", "host", "run 1"],
        ["w-b", "a100", "fp8 sub-check", "PASS", "", "", "d", "2026-01-03", "doc", "host", "doc S"],
        ["w-b", "a100", "full", "FAIL", "7", "", "d", "2026-01-04", "doc", "host", "doc S"]])
    put(tmp, "no_counterpart.tsv", "workload code reason provenance".split(), [["w-c", "no-sim-target", "manifest lists none", "manifest"]])
    put(tmp, "parity_metrics.tsv", "workload metric real_value real_provenance profile sim_value sim_provenance".split(), [
        ["w-a", "err", "0.001", "doc A", "a100", "0.001", "run 1"], ["w-b", "err", "0.002", "doc A", "a100", "0.004", "doc S"]])
    put(tmp, "pairs.tsv", "workload real_commit sim_commit same_reference_and_tolerance basis".split(), [["w-a", "c1", "s1", "yes", "diff"]])
    put(tmp, "datasets.tsv", "id side date commit environment provenance".split(), [
        ["nightly", "sim", "2026-01-05", "s1", "env", "run 1"], ["real-rerun", "real", "2026-01-01", "c1", "env", "doc A"],
        ["sim-validation", "sim", "2026-01-03", "s0", "env", "doc S"]])
    put(tmp, "profiles.tsv", "profile real_counterpart architecture provenance".split(), [["a10", "A10G", "sm_86", "p"]])


class CommittedPage(unittest.TestCase):
    def test_data_directory_exists(self):
        for name in FILES:
            self.assertTrue((DATA / name).is_file(), name)

    def test_page_is_current(self):
        self.assertEqual(svg.check(ROOT), [], "run `bin/pw coverage --write`")

    def test_every_row_has_provenance(self):
        for name, col in [("real_results.tsv", "provenance"), ("sim_results.tsv", "provenance"), ("no_counterpart.tsv", "provenance"),
                          ("parity_metrics.tsv", "real_provenance"), ("parity_metrics.tsv", "sim_provenance"), ("datasets.tsv", "provenance"),
                          ("profiles.tsv", "provenance")]:
            for r in rows(name):
                self.assertTrue(r[col].strip(), f"{name}: {r} has no {col}")

    def test_pairs_name_workloads_present_on_both_sides(self):
        real = {r["workload"] for r in rows("real_results.tsv")}
        sim = {r["workload"] for r in rows("sim_results.tsv")}
        for r in rows("pairs.tsv"):
            self.assertIn(r["workload"], real & sim)

    def test_no_real_wall_clock_is_invented(self):
        # no source records one; the day one does, this test is the place to look at its provenance
        self.assertEqual([r["workload"] for r in rows("real_results.tsv") if r["wall_seconds"]], [])

    def test_no_counterpart_is_disjoint_from_the_pairs(self):
        sim = {r["workload"] for r in rows("sim_results.tsv")}
        for r in rows("no_counterpart.tsv"):
            self.assertNotIn(r["workload"], sim)

    def test_page_states_the_rule(self):
        text = (ROOT / "docs" / "sim-vs-gpu.md").read_text(encoding="utf-8")
        self.assertIn("no timing model", text)
        self.assertIn("cost of simulating (CPU time), not GPU performance", text)


class Generator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_reads_only_the_data_files(self):
        make_root(self.tmp)
        self.assertFalse((pathlib.Path(self.tmp) / "README.md").exists())
        text = svg.render(self.tmp)
        self.assertIn("## Table 1", text)

    def test_pairing_and_no_counterpart(self):
        make_root(self.tmp)
        text = svg.render(self.tmp)
        table1 = text.split("## Table 1")[1].split("## Table 2")[0]
        self.assertIn("| w-a |", table1)
        self.assertIn("| w-b |", table1)
        self.assertNotIn("| w-c |", table1)
        tail = text.split("## No counterpart")[1]
        self.assertIn("| w-c | nvidia-a10g", tail)
        self.assertIn("| w-d | nvidia-l40s", tail)
        self.assertIn("no-sim-profile", tail)

    def test_superseded_real_record_is_a_note_not_the_verdict(self):
        make_root(self.tmp)
        row = [l for l in svg.render(self.tmp).splitlines() if l.startswith("| w-b |")][0]
        self.assertIn("| PASS; earlier FAIL on old torch (2025-12-01) |", row)

    def test_full_run_beats_subcheck_and_unrun_card_is_not_a_pass(self):
        make_root(self.tmp)
        d = svg.Data(self.tmp)
        self.assertEqual(svg.cell(d.best("w-b", "a100")), "FAIL")
        self.assertEqual(svg.cell(d.best("w-b", "h100")), "not run")

    def test_ratio_is_empty_without_both_seconds(self):
        make_root(self.tmp, real_seconds="", a10_seconds="40")
        text = svg.render(self.tmp)
        self.assertIn("cannot be computed", text)
        row = [l for l in text.split("## Table 2")[1].splitlines() if l.startswith("| w-a |")][0]
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        self.assertEqual(cells[1], "")        # no real wall-clock
        self.assertEqual(cells[2], "40")      # the simulated seconds are shown
        self.assertEqual(cells[8], "")        # so there is no ratio

    def test_ratio_when_both_are_recorded(self):
        make_root(self.tmp, real_seconds="10", a10_seconds="40")
        text = svg.render(self.tmp)
        self.assertIn("sim/real ratio (1 pairs): median 4, range 4 to 4", text)

    def test_deviations_are_compared_with_the_same_threshold(self):
        make_root(self.tmp)
        t = svg.render(self.tmp).split("### Table 1b")[1].split("## Table 2")[0]
        self.assertIn("| w-a | err | 0.001 | a100 0.001 | 1 | same |", t)
        self.assertIn("| w-b | err | 0.002 | a100 0.004 | 2 | differs |", t)

    def test_check_detects_drift_and_write_fixes_it(self):
        make_root(self.tmp)
        self.assertTrue(svg.check(self.tmp))                 # no page yet
        svg.write(self.tmp)
        self.assertEqual(svg.check(self.tmp), [])
        page = pathlib.Path(self.tmp) / svg.PAGE
        page.write_text(page.read_text(encoding="utf-8") + "edited by hand\n", encoding="utf-8")
        self.assertTrue(svg.check(self.tmp))
        make_root(self.tmp, real_seconds="10", a10_seconds="40")   # the data changed, the page did not
        svg.write(self.tmp)
        put(self.tmp, "datasets.tsv", "id side date commit environment provenance".split(), [
            ["nightly", "sim", "2026-02-01", "s2", "env", "run 2"], ["real-rerun", "real", "2026-01-01", "c1", "env", "doc A"],
            ["sim-validation", "sim", "2026-01-03", "s0", "env", "doc S"]])
        self.assertTrue(svg.check(self.tmp))

    def test_coverage_check_includes_the_page(self):
        import coverage_table
        self.assertEqual(coverage_table.check(ROOT), [])


if __name__ == "__main__":
    unittest.main()
