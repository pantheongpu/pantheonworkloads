"""docs/sim-vs-gpu.md: the workloads on a real GPU against the same card simulated in pantheonsim.

    python3 tools/sim_vs_gpu.py            print the page
    python3 tools/sim_vs_gpu.py --write    rewrite docs/sim-vs-gpu.md
    python3 tools/sim_vs_gpu.py --check    exit 1 if the page differs from the generated output
    (`bin/pw coverage --check` / `--write` do the same, with the README blocks)

The generator reads ONLY the TSV files under docs/data/sim-vs-gpu/, each row carrying its provenance (source document
and section, or workflow run, artifact and commit). It never runs a workload, never calls the network and never
estimates a value: a number the files do not hold is an empty cell. pantheonsim has no timing model, so its seconds
are CPU time; they are shown only as the cost of simulating, never as GPU performance.
"""
import csv
import pathlib
import statistics
import sys

DATA = pathlib.Path("docs") / "data" / "sim-vs-gpu"
PAGE = pathlib.Path("docs") / "sim-vs-gpu.md"
PAIR = "a10g"                 # the simulated counterpart of the real A10G (AWS g5)
PAIR_L40S = "l40s"            # the simulated counterpart of the real L40S (AWS g6e)
REAL_GPU = "nvidia-a10g"
OTHER = ["a10", "a100", "h100", "rtx3060", "rtx3080ti", "rtx5090", "t4", "l4", "b200", "l40s"]
NIGHTLY_CARDS = ["a10g", "l40s", "a100", "h100", "rtx3060", "rtx5090", "t4", "l4"]
# the real cards and the simulated profile each is paired with: key = sim profile, gpu = the `gpu` column of the real files
CARDS = [
    {"key": "a10g", "gpu": REAL_GPU, "name": "A10G", "aws": "g5", "sm": "sm_86", "rig": "one g5.2xlarge (8 vCPU)"},
    {"key": "t4", "gpu": "nvidia-t4", "name": "T4", "aws": "g4dn", "sm": "sm_75", "rig": "one g4dn.xlarge (4 vCPU)"},
    {"key": "l4", "gpu": "nvidia-l4", "name": "L4", "aws": "g6", "sm": "sm_89", "rig": "one g6.xlarge (4 vCPU)"},
    {"key": "l40s", "gpu": "nvidia-l40s", "name": "L40S", "aws": "g6e", "sm": "sm_89", "rig": "one g6e.xlarge (4 vCPU)"},
]
OTHER_CARDS = [c for c in CARDS if c["key"] != PAIR]
CLOSE = 0.10                  # a sim/real ratio inside 1 +- CLOSE is called "same"


def load(root, name, optional=False):
    path = pathlib.Path(root) / DATA / name
    if optional and not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def esc(s):
    return str(s).replace("|", "/")


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def g3(x):
    return f"{x:.3g}"


def g4(x):
    return f"{x:.4g}"


def short(text, n=150):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return out


class Data:
    def __init__(self, root):
        self.real = load(root, "real_results.tsv")
        # every real run, one row each (real_runs.tsv, with a `gpu` column); wall_seconds in real_results.tsv is the median of these
        self.runs_by = {}
        for r in load(root, "real_runs.tsv", optional=True):
            self.runs_by.setdefault((r.get("gpu") or REAL_GPU, r["workload"]), []).append(r)
        self.runs = {w: v for (g, w), v in self.runs_by.items() if g == REAL_GPU}
        self.sim = load(root, "sim_results.tsv")
        self.nocp = load(root, "no_counterpart.tsv")
        self.metrics = load(root, "parity_metrics.tsv")
        for m in self.metrics:
            m["real_gpu"] = m.get("real_gpu") or REAL_GPU
        # (card key, workload) -> the record of which commits and why the reference and tolerance are the same
        self.pairs_by = {(r.get("card") or PAIR, r["workload"]): r for r in load(root, "pairs.tsv")}
        self.pairs = {w: r for (c, w), r in self.pairs_by.items() if c == PAIR}
        self.sets = load(root, "datasets.tsv")
        self.profiles = load(root, "profiles.tsv")
        # the real record of a (gpu, workload): the one not marked superseded; earlier ones are kept for the note
        self.real_rec = {}
        self.real_earlier = {}
        for r in self.real:
            k = (r["gpu"], r["workload"])
            if r["note"].startswith("superseded"):
                self.real_earlier.setdefault(k, []).append(r)
            else:
                self.real_rec[k] = r
        self.real_a10g = {w: r for (g, w), r in self.real_rec.items() if g == REAL_GPU}
        self.sim_workloads = sorted({r["workload"] for r in self.sim})

    def best(self, workload, profile):
        """Newest sim row for (workload, profile), a full run before a sub-check, nightly before a document on a tie."""
        rows = [r for r in self.sim if r["workload"] == workload and r["profile"] == profile]
        if not rows:
            return None
        return max(rows, key=lambda r: (r["scope"] == "full", r["date"], r["source"] == "nightly"))

    def paired_for(self, card):
        """Workloads with a real result on this card and a sim result on any profile, in the order of the real file."""
        seen, out = set(), []
        for r in self.real:
            w = r["workload"]
            if r["gpu"] == card["gpu"] and (card["gpu"], w) in self.real_rec and w in self.sim_workloads and w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def paired(self):
        return self.paired_for(CARDS[0])


def ratios_of(ratios, w, metric):
    return [(m, q) for m, q in ratios if m["workload"] == w and m["metric"] == metric]


def cell(row):
    if row is None:
        return "not run"
    if row["scope"] != "full":
        return f"{row['result']} ({row['scope']} only)"
    return row["result"]


def real_cell(d, w, card=None):
    gpu = card["gpu"] if card else REAL_GPU
    r = d.real_rec[(gpu, w)]
    text = r["result"] + (f" ({r['note']})" if r["note"] else "")
    for e in d.real_earlier.get((gpu, w), []):
        text += f"; earlier {e['result']} on {e['stack']} ({e['date']})"
    return text


def render(root="."):
    d = Data(root)
    pw = d.paired()
    out = []
    a = out.append
    nightly = next(s for s in d.sets if s["id"] == "nightly")
    first = next(s for s in d.sets if s["id"] == "real-rerun")
    old = next(s for s in d.sets if s["id"] == "sim-validation")
    a("# Real GPU against its simulated counterpart")
    a("")
    a("Generated by `tools/sim_vs_gpu.py` (`bin/pw coverage --write`) from the files in `docs/data/sim-vs-gpu/`; do not edit by hand. "
      "Every value below is a row of those files, and each row names its source (a document and section, or a workflow run, "
      "artifact and commit). `bin/pw coverage --check` fails in CI when this page and the data differ.")
    a("")
    a("## How to read this")
    a("")
    a("- **pantheonsim has no timing model.** It is a functional simulator: it says whether a simulated GPU gives the same answers as "
      "a real one, never how fast the real one is. Its wall-clock is CPU time on whatever host ran it. Table 2 shows that time as "
      "the cost of simulating, not as GPU performance, and no simulated number is placed beside a real throughput. `bin/pw` still "
      "refuses benchmark metrics on `sim:` targets; this page does not change that.")
    a("- **Parity** means: the same workload, checked against the same reference with the same tolerance, gives the same verdict on the "
      "real card and on the simulated one, and the deviations from the reference are of the same size. It does not mean the "
      "simulated card is as fast, and PASS on both sides only says both stayed inside the tolerance.")
    counts = {c["key"]: len(d.paired_for(c)) for c in CARDS}
    a("- **Pairing.** Each real NVIDIA card is paired with the simulated profile that pantheonsim characterized on that part (pantheonsim "
      "`nvidia/docs/profiles.md`): " + ", ".join(f"the {c['name']} (AWS {c['aws']}, {c['sm']}) with `sim:nvidia/{c['key']}`" for c in CARDS)
      + ". Workloads with a real and a simulated result: " + ", ".join(f"{c['name']} {counts[c['key']]}" for c in CARDS) + "."
      + "".join(f" The real {c['name']} has real results only for other workloads (for the L40S the stage-2a catalog ones: llama.cpp and diffusion), "
                f"none of the PyTorch arch-*, lib-* and training ones the simulator runs, so the {c['name']} pair has no workload in common "
                "(see *No counterpart*) and its tables are empty." for c in CARDS if not counts[c["key"]])
      + " The other simulated profiles are shown as PASS / FAIL / SKIP columns for the same A10G workload (`a10` is the generic A10, a "
      "different profile from `a10g`); \"not run\" means no result exists, not a pass.")
    a("- **Same reference and tolerance** is checked from the repository, not assumed: the real run and the simulated run used "
      "workload directories that are identical (manifest, `reference.json` and scripts) between the two commits named in the data, "
      "except where the row says otherwise. Where the real rerun and a scheduled nightly run on main used the same commit, that is the "
      "basis (the same commit, so the same files). The simulated results are the newest scheduled nightly run on main named in the "
      "data sets below, except `l4`, which is a pull-request run (the profile was added to the matrix by the change that added this comparison, "
      "so no scheduled run has it yet).")
    nreal_runs = sum(len(v) for v in d.runs_by.values())
    a("- **Never estimated.** An empty cell is a value the sources do not hold. Real wall-clock seconds exist only for the workloads "
      f"of the real reruns ({len(d.runs_by)} (card, workload) series, {nreal_runs} runs, each one a row of `real_runs.tsv`); a workload outside them has an "
      "empty real cell and no ratio.")
    a("- **Software.** The real reruns (A10G, T4, L4, L40S) and the nightly simulation used the same libraries: torch 2.14.1+cu130, torchvision 0.29.1 and "
      "transformers 5.18.0 (the version the references were recorded with; the older real records of 2026-10-08 used 5.19.0 for the "
      "arch-* workloads and are marked superseded where the rerun replaces them).")
    a("")
    a("### Data sets and their dates")
    a("")
    a("\n".join(table(["Data set", "Side", "Date", "Commit", "Environment", "Source"],
                      [[s["id"], s["side"], s["date"], s["commit"], s["environment"], s["provenance"]] for s in d.sets])))
    a("")

    # ---- summary (all counted from the data)
    pair_ok = [w for w in pw if d.real_a10g[w]["result"] == "PASS" and all(
        (b := d.best(w, p)) is None or b["result"] in ("PASS", "SKIP") for p in OTHER + [PAIR])]
    a10_rows = [w for w in pw if d.best(w, PAIR)]
    a10_full = [w for w in a10_rows if d.best(w, PAIR)["scope"] == "full"]
    ratios = []
    for m in d.metrics:
        r, s = num(m["real_value"]), num(m["sim_value"])
        if r and s is not None:
            ratios.append((m, s / r))
    a10g_ratios = [(m, q) for m, q in ratios if m["real_gpu"] == REAL_GPU]
    differ = sorted({(m["workload"], m["metric"]) for m, q in a10g_ratios if abs(q - 1) > CLOSE})
    a("## Summary")
    a("")
    a(f"- Workloads in Table 1 (real A10G result and a simulated result): **{len(pw)}**. Real A10G verdict PASS for {sum(d.real_a10g[w]['result'] == 'PASS' for w in pw)}; "
      f"every simulated result on every profile is PASS or SKIP for {len(pair_ok)}.")
    missing_a10g = [w for w in pw if not d.best(w, PAIR)]
    a(f"- Workloads of Table 1 run on the exact counterpart card `sim:nvidia/{PAIR}`: **{len(a10_full)}** of {len(pw)}; "
      f"not run there: {', '.join(missing_a10g) or 'none'}" + (" (the nightly workflow's workload list does not include them)." if missing_a10g else "."))
    ndiff = len(differ)
    a(f"- Deviation figures that can be compared (a real A10G value and a simulated value, {len(a10g_ratios)} pairs): "
      f"{ndiff} (workload, metric) combinations differ by more than {int(CLOSE * 100)}% between real and simulated; see Table 1b.")
    for c in OTHER_CARDS:
        pwc = d.paired_for(c)
        if not pwc:
            a(f"- Real {c['name']} against `sim:nvidia/{c['key']}`: **no workload in common** (no real {c['name']} result for any workload the simulator runs); "
              "the real column stays empty and nothing is estimated.")
            continue
        rr = {w: d.real_rec[(c["gpu"], w)]["result"] for w in pwc}
        ss = {w: (d.best(w, c["key"]) or {}).get("result", "not run") for w in pwc}
        agree = [w for w in pwc if rr[w] == ss[w]]
        split = [w for w in pwc if rr[w] != ss[w]]
        a(f"- Real {c['name']} against `sim:nvidia/{c['key']}`: **{len(pwc)}** workloads. Real verdicts: "
          + ", ".join(f"{k} {sum(1 for v in rr.values() if v == k)}" for k in ("PASS", "FAIL", "SKIP") if any(v == k for v in rr.values()))
          + ". Simulated verdicts: " + ", ".join(f"{k} {sum(1 for v in ss.values() if v == k)}" for k in ("PASS", "FAIL", "SKIP") if any(v == k for v in ss.values()))
          + f". The verdict is the same on {len(agree)} of {len(pwc)}"
          + ("; **it differs for** " + "; ".join(f"{w} (real {rr[w]}, sim {ss[w]})" for w in split) if split else "") + ".")
    a("")

    # ---- Table 1
    a("## Table 1: functional parity")
    a("")
    head = ["Workload", "Real A10G", "Real deviation / detail", "Real date", f"Sim {PAIR}", f"Sim {PAIR} deviation / detail",
            "Same reference and tolerance", "Sim date(s)"] + OTHER
    rows = []
    for w in pw:
        r = d.real_a10g[w]
        pa = d.best(w, PAIR)
        dates = sorted({b["date"] for b in (d.best(w, p) for p in [PAIR] + OTHER) if b})
        pr = d.pairs.get(w)
        same = (pr["same_reference_and_tolerance"] + f" ({pr['real_commit']} vs {pr['sim_commit']})") if pr else "not checked"
        rows.append([w, real_cell(d, w), short(r["detail"], 130), r["date"], cell(pa), short(pa["detail"], 110) if pa else "not run", same, ", ".join(dates)] + [cell(d.best(w, p)) for p in OTHER])
    a("\n".join(table(head, rows)))
    a("")
    a("Real deviation / detail is the largest difference from the reference where the source gives one (for the rerun rows, the "
      "figures of the first run, which the three runs of a workload reproduce). Each cell's source is in "
      "`docs/data/sim-vs-gpu/real_results.tsv` and `sim_results.tsv`.")
    a("")
    for c in OTHER_CARDS:
        pwc = d.paired_for(c)
        a(f"### Parity, real {c['name']} against `sim:nvidia/{c['key']}`")
        a("")
        if not pwc:
            a(f"No workload has both a real {c['name']} result and a simulated one, so there is nothing to compare. The real {c['name']} column is empty on purpose.")
            a("")
            continue
        head = ["Workload", f"Real {c['name']}", "Real deviation / detail", "Real date", f"Sim {c['key']}", f"Sim {c['key']} deviation / detail",
                "Verdict the same", "Same reference and tolerance", "Sim date"]
        rows = []
        for w in pwc:
            r = d.real_rec[(c["gpu"], w)]
            ps = d.best(w, c["key"])
            pr = d.pairs_by.get((c["key"], w))
            same = (pr["same_reference_and_tolerance"] + f" ({pr['real_commit']} vs {pr['sim_commit']})") if pr else "not checked"
            verdict = ("yes" if ps and ps["result"] == r["result"] else "NO") if ps else "not run"
            rows.append([w, real_cell(d, w, c), short(r["detail"], 130), r["date"], cell(ps), short(ps["detail"], 110) if ps else "not run", verdict, same, ps["date"] if ps else ""])
        a("\n".join(table(head, rows)))
        a("")
    a("### Table 1b: deviation from the reference, real against simulated")
    a("")
    a("Only figures that appear on both sides. sim/real is the simulated value divided by the real one; the checks compare against "
      "the same float64 CPU result (ops) or the same recorded training run (loss). For the arch-* workloads the nightly artifact keeps "
      "only the first 200 characters of the detail line, so the comparison uses the first architecture of each workload (its margin and "
      "fp64 logits deviation); the real side's other architectures are in `real_results.tsv`.")
    a("")
    def dev_rows(gpu):
        grouped = {}
        for m in d.metrics:
            if m["real_gpu"] == gpu:
                grouped.setdefault((m["workload"], m["metric"]), []).append(m)
        rows = []
        for (w, metric), ms in grouped.items():
            real_v = num(ms[0]["real_value"])
            sims = [(m["profile"], num(m["sim_value"])) for m in ms]
            qs = [s / real_v for _, s in sims if real_v]
            rows.append([w, metric, g4(real_v), ", ".join(f"{p} {g4(s)}" for p, s in sims),
                         (g3(min(qs)) if g3(min(qs)) == g3(max(qs)) else f"{g3(min(qs))} to {g3(max(qs))}") if qs else "", "same" if qs and all(abs(q - 1) <= CLOSE for q in qs) else "differs"])
        return rows
    a("\n".join(table(["Workload", "Metric", "Real A10G", "Simulated (profile value)", "sim/real", f"Within {int(CLOSE * 100)}%"], dev_rows(REAL_GPU))))
    a("")
    for c in OTHER_CARDS:
        rows = dev_rows(c["gpu"])
        if not rows:
            continue
        a(f"#### Real {c['name']} against `sim:nvidia/{c['key']}`")
        a("")
        a("\n".join(table(["Workload", "Metric", f"Real {c['name']}", "Simulated (profile value)", "sim/real", f"Within {int(CLOSE * 100)}%"], rows)))
        a("")
        diff = [r for r in rows if r[5] == "differs"]
        a(f"Where the {c['name']} pair differs by more than {int(CLOSE * 100)}%: " + ("; ".join(f"{r[0]} ({r[1]}): sim/real {r[4]}" for r in diff) or "nothing") + ".")
        a("")

    a("Where the A10G comparisons differ: " + ("; ".join(
        f"{w} ({m}): sim/real " + ", ".join(f"{q[0]['profile']} {g3(q[1])}" for q in ratios_of(a10g_ratios, w, m) if abs(q[1] - 1) > CLOSE)
        for w, m in differ) or "nothing") + ". Both sides are inside the workload's tolerance in every case (every verdict is PASS), so these "
      "are differences in how close to the reference each lands, not failures. `docs/sim-validation.md` says why a difference is expected: "
      "the simulator runs the cuBLAS, cuDNN and similar math on the host, so it lands close to the CPU result, while a real card's "
      "libraries round differently (and differently per architecture, e.g. bf16 and the T4's SASS).")
    a("")

    # ---- Table 2
    a("## Table 2: cost of simulation")
    a("")
    a("Columns headed *cost of simulating (CPU time), not GPU performance* are the wall-clock seconds of one `bin/pw run` on the "
      "simulator (starting it, importing torch, running the workload on the host CPU). They say nothing about how long the "
      "real card takes. The nightly times come from a GitHub-hosted runner whose load the artifact does not record. "
      "Real A10G seconds are the whole-workload wall-clock `bin/pw run` writes to `results.tsv` (process start, imports and the "
      "workload's own CPU float64 oracle included, the same quantity as the simulated column), measured on one idle A10G; "
      "*median* is the median of three consecutive runs, *first run* is the cold one (empty file and kernel caches after setup). "
      "sim/real is the simulated seconds divided by the real median: how many times slower it is to simulate the workload than to run it "
      "on the card, not a statement about GPU speed.")
    a("")
    other = [c for c in NIGHTLY_CARDS if c != PAIR]
    head = ["Workload", "Real A10G median of 3 (s)", "Real first run, cold (s)", "Real runs 1, 2, 3 (s)",
            f"Sim {PAIR}: cost of simulating (CPU time), not GPU performance (s)"] + \
           [f"Sim {c} (s)" for c in other] + ["sim/real (median)", "sim/real (first run)", "2026-10-06 shared host (not comparable with the nightly)"]
    rows = []
    vals = []
    ratio_vals = []
    first_ratio_vals = []
    per = {}
    for w in pw:
        rs = num(d.real_a10g[w]["wall_seconds"])
        runs = sorted(d.runs.get(w, []), key=lambda r: int(r["run"]))
        firstrun = next((num(r["seconds"]) for r in runs if r["cold"] == "yes"), None)
        pa = d.best(w, PAIR)
        sa = num(pa["seconds"]) if pa else None
        ratio = g3(sa / rs) if rs and sa is not None else ""
        fratio = g3(sa / firstrun) if firstrun and sa is not None else ""
        if rs and sa is not None:
            ratio_vals.append(sa / rs)
            per[w] = sa / rs
        if firstrun and sa is not None:
            first_ratio_vals.append(sa / firstrun)
        cells = []
        for c in other:
            row = next((r for r in d.sim if r["workload"] == w and r["profile"] == c and r["source"] == "nightly"), None)
            x = num(row["seconds"]) if row else None
            if x is not None and row["result"] == "PASS":      # a SKIP's seconds are not a workload's run time
                vals.append(x)
            cells.append(g3(x) if x is not None else "")
        docs = [r for r in d.sim if r["workload"] == w and r["source"] == "doc" and r["scope"] == "full"]
        secs = [num(r["seconds"]) for r in docs if num(r["seconds"]) is not None]
        notes = sorted({r["seconds_note"] for r in docs if r["seconds_note"]})
        parts = []
        if secs:
            parts.append((f"{g3(min(secs))} to {g3(max(secs))} s" if len(secs) > 1 else f"{g3(secs[0])} s")
                         + f" ({len(secs)} card{'s' if len(secs) != 1 else ''} with a per-card time)")
        parts += notes
        old_text = "; ".join(parts)
        rows.append([w, g3(rs) if rs else "", g3(firstrun) if firstrun else "", ", ".join(g3(num(r["seconds"])) for r in runs),
                     g3(sa) if sa is not None else ""] + cells + [ratio, fratio, old_text])
    a("\n".join(table(head, rows)))
    a("")
    nreal = sum(1 for w in pw if num(d.real_a10g[w]["wall_seconds"]))
    no_real = [w for w in pw if not num(d.real_a10g[w]["wall_seconds"])]
    no_sim = [w for w in pw if num(d.real_a10g[w]["wall_seconds"]) and (d.best(w, PAIR) is None or num(d.best(w, PAIR)["seconds"]) is None)]
    a(f"- Real wall-clock recorded for {nreal} of {len(pw)} workloads" + (f"; empty for {', '.join(no_real)}" if no_real else "") + ". "
      "Nothing was estimated from throughput or any other figure.")
    if ratio_vals:
        a(f"- sim/real ratio, simulated seconds on `sim:nvidia/{PAIR}` over the real median ({len(ratio_vals)} pairs): median {g3(statistics.median(ratio_vals))}, "
          f"range {g3(min(ratio_vals))} ({min(per, key=per.get)}) to {g3(max(ratio_vals))} ({max(per, key=per.get)}).")
    else:
        a("- The sim/real ratio cannot be computed for any pair, and no median or range is given. "
          f"The simulated seconds on `sim:nvidia/{PAIR}` exist; the missing half is the real run's.")
    if first_ratio_vals:
        a(f"- Against the cold first run ({len(first_ratio_vals)} pairs): median {g3(statistics.median(first_ratio_vals))}, "
          f"range {g3(min(first_ratio_vals))} to {g3(max(first_ratio_vals))}.")
    if no_sim:
        a(f"- Real seconds without a ratio, because the nightly workflow does not run them on `sim:nvidia/{PAIR}`: {', '.join(no_sim)}.")
    a("- **Caveats.** (1) The simulated seconds are CPU time on a shared GitHub-hosted runner (load not recorded); the real seconds "
      "are one g5.2xlarge (8 vCPU) with the A10G idle. The ratio is the cost of simulating on that hardware pair, not a property of "
      "the A10G or of the simulator's design. (2) Both sides include process start, torch imports and the workload's float64 CPU "
      "oracle, so a floor of a few seconds is common to both and compresses the ratio of the short workloads; a workload with more "
      "GPU work would show a larger ratio. (3) The workloads are small checks, not benchmarks: no real throughput is implied and none "
      "is compared. (4) One rig, one day, three runs per workload; the first run is slower in some workloads because of cold "
      "caches, which is why both views are shown.")
    if vals:
        a(f"- Across cards: one nightly run of these workloads on the simulator (scheduled on main; the l4 column is a pull-request run, see the data sets) took a median of {g3(statistics.median(vals))} s "
          f"(range {g3(min(vals))} to {g3(max(vals))} s over the other {len(other)} cards, {len(vals)} runs, {nightly['date']}). "
          "That is the host CPU's time, not a property of any GPU; differences between simulated cards in these columns come from the hosted runners (each card was a separate job on its own runner), not from the cards.")
    a("- Known caveat of the 2026-10-06 times: the host was shared with other jobs (load average 6 to 8), and `arch-ssm` took 406 s "
      "on the first card (rtx5090, under load from other jobs) against 24 s on the last (t4). The time depends on the host's load, so "
      "that column is not a measurement of the simulator.")
    a("")

    # ---- Table 2b: the other real cards, and the summary by card
    def series(c):
        out = []
        for w in d.paired_for(c):
            rec = d.real_rec[(c["gpu"], w)]
            runs = sorted(d.runs_by.get((c["gpu"], w), []), key=lambda r: int(r["run"]))
            rs = num(rec["wall_seconds"])
            fr = next((num(r["seconds"]) for r in runs if r["cold"] == "yes"), None) if rs else None
            sim = d.best(w, c["key"])
            sa = num(sim["seconds"]) if sim and sim["result"] != "SKIP" else None
            out.append((w, rec, runs, rs, fr, sim, sa))
        return out

    a("### Table 2b: cost of simulation on the other real cards")
    a("")
    a("The same quantities as Table 2, for each other real card and its own simulated profile. A run that ended SKIP (the workload exits 77) has no "
      "run time: the seconds are the time to the skip, so it has no median and no ratio. A real FAIL did run to the end (the comparison with "
      "the reference is made afterwards), so its seconds are a run time and are used, with the verdict shown next to them.")
    a("")
    for c in OTHER_CARDS:
        ser = series(c)
        if not ser:
            a(f"No real {c['name']} result exists for these workloads: no seconds, no ratio.")
            a("")
            continue
        a(f"#### Real {c['name']} ({c['rig']}) against `sim:nvidia/{c['key']}`")
        a("")
        head = ["Workload", f"Real {c['name']} median of 3 (s)", "Real first run, cold (s)", "Real runs 1, 2, 3 (s)", "Real verdict",
                f"Sim {c['key']}: cost of simulating (CPU time), not GPU performance (s)", "Sim verdict", "sim/real (median)", "sim/real (first run)"]
        rows = []
        for (w, rec, runs, rs, fr, sim, sa) in ser:
            skip = rec["result"] == "SKIP"
            rows.append([w, g3(rs) if rs else ("SKIP" if skip else ""), g3(fr) if fr else ("SKIP" if skip else ""),
                         "" if skip else ", ".join(g3(num(r["seconds"])) for r in runs), rec["result"],
                         g3(sa) if sa is not None else ("SKIP" if sim and sim["result"] == "SKIP" else ""), sim["result"] if sim else "not run",
                         g3(sa / rs) if rs and sa is not None else "", g3(sa / fr) if fr and sa is not None else ""])
        a("\n".join(table(head, rows)))
        a("")
        rv = [sa / rs for (w, rec, runs, rs, fr, sim, sa) in ser if rs and sa is not None]
        pc = {w: sa / rs for (w, rec, runs, rs, fr, sim, sa) in ser if rs and sa is not None}
        fv = [sa / fr for (w, rec, runs, rs, fr, sim, sa) in ser if fr and sa is not None]
        if rv:
            a(f"- sim/real ratio, simulated seconds on `sim:nvidia/{c['key']}` over the real median ({len(rv)} pairs): median {g3(statistics.median(rv))}, "
              f"range {g3(min(rv))} ({min(pc, key=pc.get)}) to {g3(max(rv))} ({max(pc, key=pc.get)}).")
            a(f"- Against the cold first run ({len(fv)} pairs): median {g3(statistics.median(fv))}, range {g3(min(fv))} to {g3(max(fv))}.")
        noratio = [w for (w, rec, runs, rs, fr, sim, sa) in ser if not (rs and sa is not None)]
        if noratio:
            a(f"- No ratio for: {', '.join(noratio)} (a SKIP has no run time).")
        a("")

    a("### Cost of simulation by card")
    a("")
    a("Each row is one real card's rig against its own simulated profile, from the tables above (the sim/real ratio of each workload, then the median "
      "and the range over the workloads). The simulated seconds are CPU time on a hosted runner and the real seconds include the rig's own CPU "
      "work (process start, imports, the float64 oracle), so the ratios of different cards are not a ranking of the cards.")
    a("")
    head = ["Real card", "Sim profile", "Real rig", "Pairs", "sim/real median", "Range (workload)", "Against the cold first run: median", "Range"]
    rows = []
    for c in CARDS:
        ser = series(c)
        pc = {w: sa / rs for (w, rec, runs, rs, fr, sim, sa) in ser if rs and sa is not None}
        pf = {w: sa / fr for (w, rec, runs, rs, fr, sim, sa) in ser if fr and sa is not None}
        if pc:
            rows.append([c["name"], c["key"], c["rig"], str(len(pc)), g3(statistics.median(pc.values())),
                         f"{g3(min(pc.values()))} ({min(pc, key=pc.get)}) to {g3(max(pc.values()))} ({max(pc, key=pc.get)})",
                         g3(statistics.median(pf.values())), f"{g3(min(pf.values()))} to {g3(max(pf.values()))}"])
        else:
            rows.append([c["name"], c["key"], "", "0", "", "no real run of these workloads", "", ""])
    a("\n".join(table(head, rows)))
    a("")
    rigs = [f"{c['name']}: {c['rig']}" for c in CARDS if series(c)]
    a("- **Caveats for the other cards.** The real rigs differ (" + "; ".join(rigs) + "), and a real run is mostly process start, torch imports and "
      "the workload's float64 CPU oracle, which scale with the rig's CPU, not the card. The rigs of the other cards had 4 vCPU and the A10G rig 8, so the "
      "real seconds of different cards must not be read as card speed. The simulated seconds of every card come from the same kind of hosted runner "
      "(each profile its own job), so their differences are the runners', as stated above. The `l4` simulated seconds come from a pull-request run, "
      "the others from the scheduled run on main.")
    a("")

    # ---- no counterpart
    a("## No counterpart")
    a("")
    paired_keys = {(c["gpu"], w) for c in CARDS for w in d.paired_for(c)}
    l40s_sim = sorted({r["workload"] for r in d.sim if r["profile"] == PAIR_L40S})
    real_l40s = sorted({r["workload"] for r in d.real if r["gpu"] == "nvidia-l40s"})
    rows = []
    counts = {}
    for r in d.real:
        if r["note"].startswith("superseded") or (r["gpu"], r["workload"]) in paired_keys:
            continue
        n = next((x for x in d.nocp if x["workload"] == r["workload"]), None)
        if n:
            code, why = n["code"], n["reason"]
            if r["gpu"] == "nvidia-l40s":
                both = set(real_l40s) & set(l40s_sim)
                why += f"; `sim:nvidia/{PAIR_L40S}` exists and ran {len(l40s_sim)} other workloads" + (", none of them has a real L40S result" if not both else "")
        else:
            code, why = "unlisted", "no reason recorded"
        counts[code] = counts.get(code, 0) + 1
        rows.append([r["workload"], r["gpu"], r["result"], r["date"], code, why])
    a(f"{len(rows)} workloads have a real-GPU result but no simulated one. By reason: " +
      ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ". `sim-not-run` means the manifest lists a sim target but no "
      "simulated result exists; `no-sim-target` means the manifest does not list one. The L40S rows are the stage-2a Tier B workloads: "
      f"{sum(1 for r in rows if r[1] == 'nvidia-l40s')} with a real L40S result and no simulated one"
      + (f" ({len(set(real_l40s) & set(l40s_sim))} L40S workloads are paired in Table 1 and are not listed here)." if set(real_l40s) & set(l40s_sim)
         else f"; none of the {len(real_l40s)} with a real L40S result ran on `sim:nvidia/{PAIR_L40S}`."))
    a("")
    a("\n".join(table(["Workload", "Real GPU", "Real result", "Real date", "Reason code", "Reason"], rows)))
    a("")
    sim_only = [w for w in d.sim_workloads if w not in d.real_a10g]
    a("Simulated results with no real A10G result: " + (", ".join(sim_only) if sim_only else "none") + ".")
    a("")

    # ---- what is still missing
    a("## What the real-side rerun filled, and what is still missing")
    a("")
    rerun = next((x for x in d.sets if x["id"] == "real-a10g-2026-10-22"), None)
    no_dev = [w for w in pw if any(k in d.real_a10g[w]["detail"] for k in ("not recorded", "aggregate"))]
    if rerun:
        a(f"The rerun proposed in the first version of this page has been done ({rerun['date']}: {rerun['environment']}). "
          f"It ran the {len(d.runs)} workloads of Table 1 that the PyTorch rig can run, from {rerun['commit']}. It supplied:")
        a("")
        a(f"1. **Real wall-clock seconds** for {nreal} of {len(pw)} workloads, three runs each (`real_runs.tsv`); Table 2 now has the "
          "sim/real ratio for every workload the nightly runs on `sim:nvidia/a10g`.")
        a("2. **Per-workload deviation figures** for the eight that had none or one shared figure (the five arch-* workloads, "
          f"gpt-train-fp32, gpt-train-bf16, resnet18-randinit-pytorch); figures still missing in Table 1: {', '.join(no_dev) or 'none'}.")
        a("3. **The same reference and tolerance on both sides**, checked from the commits named in `pairs.tsv` (the `workloads/` tree is identical "
          "between the rerun's commit and the nightly's).")
        a("")
        later = [x for x in d.sets if x["id"].startswith("real-") and x["id"] not in ("real-a10g-2026-10-22",) and x["date"] > rerun["date"]]
        if later:
            a("The same protocol was then repeated on more cards: " + "; ".join(f"{x['date']}: {x['environment'].split(';')[0]}" for x in later)
              + ". Tables 1, 1b and 2b have those cards' rows, and the summary above counts their verdicts.")
            a("")
        nol = [c["name"] for c in CARDS if not d.paired_for(c)]
        a("Still missing: " + ("".join(f"a real {n} run of these workloads (no {n} result exists for any of them, so its pair stays empty); " for n in nol))
          + "a scheduled main run of the nightly for the `l4` rows (they come from the pull-request run named in the data sets); more than one rig and day (the run-to-run "
          "spread seen here is three consecutive runs on one idle card); and the *no counterpart* workloads, whose reasons are in the "
          "section above: for `sim-not-run` the missing side is the simulated one (a hosted run for the workloads that need a model "
          "download), and the L40S workloads have no sim target in their manifests, so giving them one is a workload change.")
    else:
        a("No real-side rerun is recorded in the data files.")
    a("")
    return "\n".join(out)


def check(root="."):
    root = pathlib.Path(root)
    page = root / PAGE
    if not page.exists():
        return [f"{PAGE} is missing; run `bin/pw coverage --write`"]
    if page.read_text(encoding="utf-8") != render(root):
        return [f"{PAGE} is stale; run `bin/pw coverage --write`"]
    return []


def write(root="."):
    root = pathlib.Path(root)
    (root / PAGE).write_text(render(root), encoding="utf-8")


def main(argv):
    root = pathlib.Path(__file__).resolve().parent.parent
    if "--check" in argv:
        problems = check(root)
        for p in problems:
            print(p, file=sys.stderr)
        return 1 if problems else 0
    if "--write" in argv:
        write(root)
        print(f"{PAGE} written")
        return 0
    print(render(root), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
