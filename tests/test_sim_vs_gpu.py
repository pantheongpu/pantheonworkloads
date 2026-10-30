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

MAIN_RUN, MAIN_COMMIT = "41049161447", "a4f74b7"      # the newest scheduled nightly on main used for every profile but l4
PR_RUN = "41059794866"                               # the pull_request run that supplies sim l4 (the profile is new in the matrix)
A10G_COMMIT = "659c10a"
DATA = ROOT / "docs" / "data" / "sim-vs-gpu"
FILES = ["real_results.tsv", "real_runs.tsv", "sim_results.tsv", "no_counterpart.tsv", "parity_metrics.tsv", "pairs.tsv", "datasets.tsv", "profiles.tsv"]


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


RUNS_H = "workload gpu run cold result seconds run_dir provenance".split()
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
    if real_seconds:
        put(tmp, "real_runs.tsv", RUNS_H, [["w-a", "nvidia-a10g", str(n), "yes" if n == 1 else "no", "PASS", real_seconds, f"run{n}/w-a", "rerun"] for n in (1, 2, 3)])
    put(tmp, "sim_results.tsv", SIM_H, [
        ["w-a", "a10g", "full", "PASS", a10_seconds, "", "d", "2026-01-05", "nightly", "host", "run 1"],
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


def extend(tmp, name, dict_rows):
    """Append rows (dicts) to a data file of a synthetic root, adding any new column."""
    path = pathlib.Path(tmp) / "docs" / "data" / "sim-vs-gpu" / name
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        head, old = list(reader.fieldnames), list(reader)
    for r in dict_rows:
        head += [k for k in r if k not in head]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head, delimiter="\t", lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(old + dict_rows)


def add_t4(tmp):
    """A second real card: w-a ran (median 5 s), w-b is a SKIP on both sides; the simulated T4 took 15 s."""
    extend(tmp, "real_results.tsv", [
        dict(workload="w-a", gpu="nvidia-t4", result="PASS", date="2026-02-01", stack="torch", detail="worst 3e-3", reference="ref", wall_seconds="5",
             commit="c2", note="", provenance="rig t4"),
        dict(workload="w-b", gpu="nvidia-t4", result="SKIP", date="2026-02-01", stack="torch", detail="SKIP: no bf16", reference="ref", wall_seconds="",
             commit="c2", note="exits 77", provenance="rig t4")])
    extend(tmp, "real_runs.tsv", [
        dict(workload="w-a", gpu="nvidia-t4", run=str(n), cold="yes" if n == 1 else "no", result="PASS", seconds=sec, run_dir=f"run{n}/w-a", provenance="rig t4")
        for n, sec in ((1, "9"), (2, "5"), (3, "5"))] + [
        dict(workload="w-b", gpu="nvidia-t4", run=str(n), cold="yes" if n == 1 else "no", result="SKIP", seconds="4", run_dir=f"run{n}/w-b", provenance="rig t4")
        for n in (1, 2, 3)])
    extend(tmp, "sim_results.tsv", [
        dict(workload="w-a", profile="t4", scope="full", result="PASS", seconds="15", detail="d", date="2026-02-02", source="nightly", host="h", provenance="run 2"),
        dict(workload="w-b", profile="t4", scope="full", result="SKIP", seconds="3", detail="SKIP", date="2026-02-02", source="nightly", host="h", provenance="run 2")])
    extend(tmp, "pairs.tsv", [dict(workload="w-a", card="t4", real_commit="c2", sim_commit="c2", same_reference_and_tolerance="yes", basis="same commit")])
    extend(tmp, "parity_metrics.tsv", [
        dict(workload="w-a", metric="err", real_value="0.003", real_provenance="rig t4", profile="t4", sim_value="0.0033", sim_provenance="run 2", real_gpu="nvidia-t4")])


class CommittedPage(unittest.TestCase):
    def test_data_directory_exists(self):
        for name in FILES:
            self.assertTrue((DATA / name).is_file(), name)

    def test_page_is_current(self):
        self.assertEqual(svg.check(ROOT), [], "run `bin/pw coverage --write`")

    def test_every_row_has_provenance(self):
        for name, col in [("real_results.tsv", "provenance"), ("sim_results.tsv", "provenance"), ("no_counterpart.tsv", "provenance"),
                          ("real_runs.tsv", "provenance"), ("parity_metrics.tsv", "real_provenance"), ("parity_metrics.tsv", "sim_provenance"), ("datasets.tsv", "provenance"),
                          ("profiles.tsv", "provenance")]:
            for r in rows(name):
                self.assertTrue(r[col].strip(), f"{name}: {r} has no {col}")

    def test_pairs_name_workloads_present_on_both_sides(self):
        real = {r["workload"] for r in rows("real_results.tsv")}
        sim = {r["workload"] for r in rows("sim_results.tsv")}
        for r in rows("pairs.tsv"):
            self.assertIn(r["workload"], real & sim)

    def test_every_real_wall_clock_comes_from_three_recorded_runs(self):
        # a wall-clock is never typed in: it is the median of the runs in real_runs.tsv, each with its provenance
        runs = {}
        for r in rows("real_runs.tsv"):
            self.assertTrue(r["provenance"].strip() and r["run_dir"].strip() and r["gpu"].strip(), r)
            runs.setdefault((r["gpu"], r["workload"]), []).append(r)
        recs = {(r["gpu"], r["workload"]): r for r in rows("real_results.tsv") if not r["note"].startswith("superseded")}
        timed = [r for r in recs.values() if r["wall_seconds"]]
        self.assertTrue(timed)
        expect = {"nvidia-a10g": (A10G_COMMIT, "rerun 2026-10-22"), "nvidia-t4": (MAIN_COMMIT, "rig tag pw-t4real"),
                  "nvidia-l4": (MAIN_COMMIT, "rig tag pw-l4real"), "nvidia-l40s": (MAIN_COMMIT, "rig tag pw-l40sreal")}
        for r in timed:
            k = (r["gpu"], r["workload"])
            self.assertEqual(len(runs.get(k, [])), 3, k)
            secs = sorted(float(x["seconds"]) for x in runs[k])
            self.assertEqual(float(r["wall_seconds"]), secs[1], k)
            self.assertEqual([x["cold"] for x in sorted(runs[k], key=lambda x: int(x["run"]))], ["yes", "no", "no"], k)
            commit, marker = expect[r["gpu"]]
            self.assertEqual(r["commit"], commit, k)
            self.assertIn(marker, r["provenance"], k)
            self.assertIn(commit, r["provenance"], k)
            self.assertIn("clean tree", r["provenance"], k)
            self.assertFalse(r["note"].startswith("superseded"), k)
            self.assertTrue(all(x["result"] == r["result"] for x in runs[k]), k)
        # a series without a wall-clock is a SKIP (the seconds are the time to the skip), recorded run by run
        for k in set(runs) - {(r["gpu"], r["workload"]) for r in timed}:
            self.assertEqual(recs[k]["result"], "SKIP", k)
            self.assertTrue(all(x["result"] == "SKIP" for x in runs[k]), k)

    def test_real_t4_and_l4_rows_have_provenance_and_the_recorded_verdicts(self):
        recs = {(r["gpu"], r["workload"]): r for r in rows("real_results.tsv") if r["date"] == "2026-10-30"}
        for gpu, tag, inst in (("nvidia-t4", "pw-t4real", "g4dn.xlarge"), ("nvidia-l4", "pw-l4real", "g6.xlarge"),
                               ("nvidia-l40s", "pw-l40sreal", "g6e.xlarge")):
            mine = {w: r for (g, w), r in recs.items() if g == gpu}
            self.assertEqual(len(mine), 14, gpu)
            for w, r in mine.items():
                self.assertIn(tag, r["provenance"])
                self.assertIn(inst, r["provenance"])
                self.assertIn(f"bin/pw run {w} --target gpu", r["provenance"])
                self.assertIn("torch 2.14.1+cu130", r["stack"])
                self.assertIn("transformers 5.18.0", r["stack"])
                self.assertIn("driver 595.112.07", r["stack"])
                self.assertEqual(r["commit"], MAIN_COMMIT)
            runs = [x for x in rows("real_runs.tsv") if x["gpu"] == gpu]
            self.assertEqual(len(runs), 42, gpu)
            for x in runs:
                self.assertIn(tag, x["provenance"])
                self.assertIn(MAIN_COMMIT, x["provenance"])
        t4 = {w: r["result"] for (g, w), r in recs.items() if g == "nvidia-t4"}
        self.assertEqual(t4["gpt-train-bf16"], "SKIP")
        self.assertEqual(t4["lib-attention-precision"], "FAIL")       # a finding, kept: nothing was loosened
        self.assertEqual(sorted(w for w, v in t4.items() if v != "PASS"), ["gpt-train-bf16", "lib-attention-precision"])
        self.assertIn("9 flash ops", recs[("nvidia-t4", "lib-attention-precision")]["note"])
        self.assertTrue(all(r["result"] == "PASS" for (g, w), r in recs.items() if g in ("nvidia-l4", "nvidia-l40s")))

    def test_the_nightly_rows_are_the_scheduled_main_run_and_l4_the_pull_request_run(self):
        nightly = [r for r in rows("sim_results.tsv") if r["source"] == "nightly"]
        self.assertTrue(nightly)
        for r in nightly:
            if r["profile"] == "l4":
                # a pull-request run, said plainly in the row, until a scheduled main run has the profile
                self.assertIn(f"run {PR_RUN}", r["provenance"])
                self.assertIn("pull_request run", r["provenance"])
                self.assertIn("NOT a run on main", r["provenance"])
                self.assertIn("PR #40", r["provenance"])
            else:
                self.assertIn(f"run {MAIN_RUN}", r["provenance"])
                self.assertIn("scheduled run on main", r["provenance"])
                self.assertIn(f"commit {MAIN_COMMIT}", r["provenance"])
            self.assertEqual(r["date"], "2026-10-30")
        # all 14 workloads of the real runs are paired with sim a10g, t4 and l4, and the SKIP on the T4 is recorded as one
        for prof, gpu in (("a10g", "nvidia-a10g"), ("t4", "nvidia-t4"), ("l4", "nvidia-l4"), ("l40s", "nvidia-l40s")):
            have = {r["workload"] for r in nightly if r["profile"] == prof}
            self.assertEqual(have, {r["workload"] for r in rows("real_runs.tsv") if r["gpu"] == gpu}, prof)
        t4 = {r["workload"]: r["result"] for r in nightly if r["profile"] == "t4"}
        self.assertEqual(t4["gpt-train-bf16"], "SKIP")
        self.assertEqual(t4["pytorch-microsuite"], "PASS")

    def test_pairs_name_their_card_and_the_commits(self):
        pairs = rows("pairs.tsv")
        by_card = {}
        for r in pairs:
            by_card.setdefault(r["card"], []).append(r)
            self.assertEqual(r["same_reference_and_tolerance"], "yes")
        self.assertEqual({c: len(v) for c, v in by_card.items()}, {"a10g": 14, "t4": 14, "l4": 14, "l40s": 14})
        self.assertTrue(all(r["real_commit"] == A10G_COMMIT and "identical" in r["basis"] for r in by_card["a10g"]))
        self.assertTrue(all(r["real_commit"] == MAIN_COMMIT and r["sim_commit"] == MAIN_COMMIT for r in by_card["t4"]))
        self.assertTrue(all(r["real_commit"] == MAIN_COMMIT and "pull_request" in r["basis"] for r in by_card["l4"]))
        self.assertTrue(all(r["real_commit"] == MAIN_COMMIT and r["sim_commit"] == MAIN_COMMIT for r in by_card["l40s"]))

    def test_page_has_the_ratio(self):
        text = (ROOT / "docs" / "sim-vs-gpu.md").read_text(encoding="utf-8")
        self.assertIn("sim/real ratio, simulated seconds on `sim:nvidia/a10g` over the real median (14 pairs)", text)
        self.assertIn("sim/real ratio, simulated seconds on `sim:nvidia/t4` over the real median (13 pairs)", text)     # the T4's SKIP has none
        self.assertIn("sim/real ratio, simulated seconds on `sim:nvidia/l4` over the real median (14 pairs)", text)
        self.assertNotIn("cannot be computed", text)
        self.assertIn("### Cost of simulation by card", text)

    def test_no_counterpart_is_disjoint_from_the_pairs(self):
        sim = {r["workload"] for r in rows("sim_results.tsv")}
        for r in rows("no_counterpart.tsv"):
            self.assertNotIn(r["workload"], sim)

    def test_l40s_has_a_profile_and_the_page_does_not_deny_it(self):
        text = (ROOT / "docs" / "sim-vs-gpu.md").read_text(encoding="utf-8")
        self.assertNotIn("no simulated profile", text)
        self.assertNotIn("no-sim-profile", text)
        self.assertIn("sim:nvidia/l40s", text)
        self.assertEqual(svg.PAIR, "a10g")
        # the L40S column is empty unless a real L40S run of these workloads is recorded; either way the page says which
        l40s = {r["workload"] for r in rows("real_results.tsv") if r["gpu"] == "nvidia-l40s"} & {r["workload"] for r in rows("sim_results.tsv")}
        if l40s:
            self.assertIn("### Parity, real L40S against `sim:nvidia/l40s`", text)
            self.assertNotIn("no workload in common (no real L40S", text)
        else:
            self.assertIn("**no workload in common** (no real L40S result", text)

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
        self.assertNotIn("no-sim-profile", tail)

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
        header = [c.strip() for c in [l for l in text.split("## Table 2")[1].splitlines() if l.startswith("| Workload |")][0].strip().strip("|").split("|")]
        self.assertEqual(cells[1], "")        # no real wall-clock
        self.assertEqual(cells[4], "40")      # the simulated seconds are shown
        self.assertEqual(cells[header.index("sim/real (median)")], "")       # so there is no ratio

    def test_ratio_when_both_are_recorded(self):
        make_root(self.tmp, real_seconds="10", a10_seconds="40")
        text = svg.render(self.tmp)
        self.assertIn("over the real median (1 pairs): median 4, range 4 (w-a) to 4 (w-a)", text)

    def test_a_second_card_has_its_own_tables_and_a_skip_has_no_ratio(self):
        make_root(self.tmp, real_seconds="10", a10_seconds="40")
        add_t4(self.tmp)
        text = svg.render(self.tmp)
        # the A10G tables do not take the T4's rows
        self.assertIn("over the real median (1 pairs): median 4, range 4 (w-a) to 4 (w-a)", text)
        t2b = text.split("### Table 2b")[1].split("### Cost of simulation by card")[0]
        self.assertIn("#### Real T4 (one g4dn.xlarge (4 vCPU)) against `sim:nvidia/t4`", t2b)
        row_a = [l for l in t2b.splitlines() if l.startswith("| w-a |")][0]
        self.assertIn("| 5 | 9 | 9, 5, 5 | PASS | 15 | PASS | 3 | 1.67 |", row_a)
        row_b = [l for l in t2b.splitlines() if l.startswith("| w-b |")][0]
        self.assertIn("| SKIP | SKIP |  | SKIP | SKIP | SKIP |  |  |", row_b)
        self.assertIn("over the real median (1 pairs): median 3, range 3 (w-a) to 3 (w-a)", t2b)
        self.assertIn("No ratio for: w-b", t2b)
        by_card = text.split("### Cost of simulation by card")[1]
        self.assertIn("| T4 | t4 | one g4dn.xlarge (4 vCPU) | 1 | 3 | 3 (w-a) to 3 (w-a) | 1.67 | 1.67 to 1.67 |", by_card)
        self.assertIn("| L4 | l4 |  | 0 |  | no real run of these workloads |  |  |", by_card)
        parity = text.split("### Parity, real T4 against `sim:nvidia/t4`")[1].split("### Table 1b")[0]
        self.assertIn("| w-b | SKIP (exits 77) |", parity)
        self.assertIn("| yes | yes (c2 vs c2) |", parity)
        self.assertIn("Real T4 against `sim:nvidia/t4`: **2** workloads", text)
        t1b = text.split("#### Real T4 against `sim:nvidia/t4`")[1].split("## Table 2")[0]
        self.assertIn("| w-a | err | 0.003 | t4 0.0033 | 1.1 | same |", t1b)

    def test_a_card_without_real_results_is_an_empty_pair_not_a_zero(self):
        make_root(self.tmp)
        text = svg.render(self.tmp)
        self.assertIn("Real T4 against `sim:nvidia/t4`: **no workload in common**", text)
        self.assertIn("No workload has both a real L4 result and a simulated one", text)
        self.assertIn("No real L40S result exists for these workloads: no seconds, no ratio.", text)

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
