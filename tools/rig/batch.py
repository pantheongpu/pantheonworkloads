#!/usr/bin/env python3
"""Run workloads on a rig (docs/rigs.md): sequentially, with per-workload timeouts, results copied back, nothing pushed.

    run_batch.sh TAG [workload ...] [--tier 2b] [--timeout S] [--pack] [--bench --repeat N] [--mirror M] [--env K=V ...]
                     [--remote-dir D] [--all-gpus] [--dry-run]

The rig must have the repository checked out at --remote-dir (default ~/pantheonworkloads) with the runtimes installed.
For every workload: skipped here (no ssh) when its manifest `requires` cannot be met by the rig's GPUs or it has no `gpu`
target; otherwise run as `bin/pw run <w> --target gpu` under `timeout`, on exactly the GPUs it needs
(CUDA_VISIBLE_DEVICES; --all-gpus leaves all visible), each GPU guarded by a flock on the rig so that nothing else can
share it. One job at a time unless --pack, which runs jobs concurrently on disjoint GPU sets. The out directory of each
job is copied back to ~/.cache/pw-rigs/<tag>/results/<batch>/ as soon as it finishes. A summary goes to batch.tsv there.
Exit 0 when nothing FAILed, timed out or found a GPU busy; 1 otherwise. It never runs git, never records into the repo.
"""
import argparse
import concurrent.futures
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rig as riglib  # noqa: E402
import wl  # noqa: E402

OUTER_GRACE_S = float(os.environ.get("PW_BATCH_GRACE_S", "120"))      # the outer `timeout` is the workload's own limit plus this
KILL_AFTER_S = float(os.environ.get("PW_BATCH_KILL_S", "30"))        # then SIGKILL this long after the SIGTERM
DEADLINE_MARGIN_S = 300  # a job is not started unless it can finish this long before the rig's deadline
EXIT_BUSY = 75


def load_pw():
    loader = importlib.machinery.SourceFileLoader("pw_batch_bin", str(wl.ROOT / "bin" / "pw"))
    spec = importlib.util.spec_from_loader("pw_batch_bin", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def parse_gpus(text):
    """nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version csv (no header, MiB) -> bin/pw's GPU dicts."""
    gpus = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            mem = round(float(parts[1]) / 1024, 2)
        except ValueError:
            mem = None
        gpus.append({"vendor": "nvidia", "name": parts[0], "memory_gib": mem,
                     "compute_capability": parts[2] if len(parts) > 2 and parts[2] else None,
                     "driver": parts[3] if len(parts) > 3 else None})
    return gpus


def remote_gpus(rig):
    p = riglib.ssh_run(rig, "nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits", timeout=120)
    if p.returncode != 0:
        raise riglib.RigError(f"nvidia-smi on the rig failed (ssh exit {p.returncode}): {(p.stderr or p.stdout).strip()[-200:]}")
    return parse_gpus(p.stdout)


def plan(names, gpus, pw, all_gpus=False):
    """([job], [(name, reason)]): jobs carry the eligible GPU indexes and how many they need; skipped ones say why."""
    jobs, skipped = [], []
    for n in names:
        m = wl.manifest(n)
        if "gpu" not in (m.get("targets") or []):
            skipped.append((n, "no gpu target"))
            continue
        why = pw.unmet_requires(m, gpus)
        if why:
            skipped.append((n, why))
            continue
        req = wl.requires(n) or {}
        eligible = [i for i, g in enumerate(gpus) if pw.meets(g, req)]
        need = len(gpus) if all_gpus else req.get("gpus", 1)
        jobs.append({"name": n, "need": need, "eligible": eligible, "timeout": float(m.get("timeout_s", 600))})
    return jobs, skipped


def dq(path):
    """Quote a remote path for bash, keeping a leading ~/ working (it becomes $HOME/)."""
    path = str(path)
    if path.startswith("~/"):
        return '"$HOME/' + path[2:].replace('"', '\\"') + '"'
    return shlex.quote(path)


def job_script(job, gpu_set, a, remote_out):
    """The bash that runs one job on the rig: lock every GPU in the set (fail fast when one is held), run bin/pw under timeout."""
    t = a.timeout or job["timeout"]
    out = dq("~/" + remote_out)   # remote_out is relative to the ssh login directory (where scp looks too)
    lines = ["set -u", f"cd {dq(a.remote_dir)}", f"mkdir -p {out}"]
    for k, g in enumerate(gpu_set):
        fd = 200 + k
        lines.append(f"exec {fd}>{a.lock_dir}/pw-gpu-{g}.lock; flock -n {fd} || {{ echo 'GPU {g} is held by another job' >&2; echo {EXIT_BUSY} > {out}/{job['name']}.exit; exit {EXIT_BUSY}; }}")
    env = {"PW_TIMEOUT_S": f"{t:g}"}
    if not a.all_gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))
    if a.mirror:
        env["PW_MODEL_MIRROR"] = a.mirror
        env["PW_MODEL_MIRROR_STRICT"] = "1" if a.strict_mirror else "0"
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + f' PW_BENCH_DIR="{out[1:-1]}/bench"'

    cmd = f"bin/pw run {shlex.quote(job['name'])} --target gpu --out {out}" + (f" --bench --repeat {a.repeat}" if a.bench else "")
    lines.append(f"env {exports} timeout -k {KILL_AFTER_S:g} {t + OUTER_GRACE_S:g} {cmd} > {out}/{job['name']}.stdout 2>&1")
    lines.append(f"echo $? > {out}/{job['name']}.exit")
    return "\n".join(lines)


def scp_from(rig, remote, local):
    d = riglib.rig_dir(rig["tag"])
    cmd = [os.environ.get("PW_SCP") or "scp", "-r", "-q", "-i", str(d / "key.pem"), "-o", "IdentitiesOnly=yes",
           "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={d / 'known_hosts'}",
           f"{rig['user']}@{rig['ip']}:{remote}/.", str(local)]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_job(rig, job, gpu_set, a, remote_out, local_out):
    """Run one job over ssh, copy its output back; returns a result row dict."""
    t0 = time.time()
    p = riglib.ssh_run(rig, f"bash -c {shlex.quote(job_script(job, gpu_set, a, remote_out))}", timeout=(a.timeout or job["timeout"]) + 2 * OUTER_GRACE_S + 600)
    c = scp_from(rig, remote_out, local_out)
    row = {"workload": job["name"], "gpus": ",".join(map(str, gpu_set)), "seconds": round(time.time() - t0, 1)}
    res = local_out / f"{job['name']}.json"
    code = None
    try:
        code = int((local_out / f"{job['name']}.exit").read_text().strip())
    except (OSError, ValueError):
        pass
    if res.is_file():
        r = json.loads(res.read_text())
        row.update(result=r.get("result", "?"), detail=str(r.get("detail", "")))
    elif code == EXIT_BUSY or p.returncode == EXIT_BUSY:
        row.update(result="BUSY", detail=(p.stderr or "a GPU is held by another job").strip()[-150:])
    elif code in (124, 137):
        row.update(result="TIMEOUT", detail=f"killed by the outer timeout (exit {code}); bin/pw wrote no result")
    else:
        row.update(result="NO-RESULT", detail=f"ssh exit {p.returncode}, job exit {code}, scp exit {c.returncode}: {(p.stderr or c.stderr).strip()[-120:]}")
    return row


class Pool:
    """GPU bookkeeping for --pack: a job gets GPUs only from the free set, so concurrent jobs can never share one."""

    def __init__(self, n):
        self.free = set(range(n))
        self.lock = threading.Lock()

    def take(self, job):
        with self.lock:
            cand = [g for g in job["eligible"] if g in self.free]
            if len(cand) < job["need"]:
                return None
            chosen = sorted(cand)[:job["need"]]
            self.free -= set(chosen)
            return chosen

    def give(self, chosen):
        with self.lock:
            self.free |= set(chosen)


def schedule(jobs, ngpus, run, pack, can_start=lambda job: True, workers=8):
    """Run jobs with run(job, gpu_set) -> row. Without pack: strictly one at a time, in order. With pack: concurrently,
    each on a set of GPUs disjoint from every running job's. Returns (rows, not_run) in job order. can_start(job) False
    stops launching (deadline)."""
    rows, not_run = {}, []
    pool = Pool(ngpus)
    if not pack:
        for k, job in enumerate(jobs):
            if not can_start(job):
                not_run = [j["name"] for j in jobs[k:]]
                break
            gset = sorted(job["eligible"])[:job["need"]]
            rows[job["name"]] = run(job, gset)
        return [rows[j["name"]] for j in jobs if j["name"] in rows], not_run
    pending = list(jobs)
    running = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        stop = False
        while pending or running:
            started = False
            if not stop:
                for job in list(pending):
                    if not can_start(job):
                        stop = True
                        break
                    gset = pool.take(job)
                    if gset is None:
                        continue
                    pending.remove(job)
                    fut = ex.submit(run, job, gset)
                    running[fut] = (job, gset)
                    started = True
            if stop and not running:
                not_run = [j["name"] for j in pending]
                break
            if not running:
                not_run = [j["name"] for j in pending]   # nothing fits the GPUs at all
                break
            if not started or stop:
                done, _ = concurrent.futures.wait(list(running), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    job, gset = running.pop(fut)
                    pool.give(gset)
                    rows[job["name"]] = fut.result()
    return [rows[j["name"]] for j in jobs if j["name"] in rows], not_run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tag")
    ap.add_argument("workload", nargs="*")
    ap.add_argument("--tier", choices=["A", "B", "2b", "big"])
    ap.add_argument("--list", help="a file of workload names, one per line")
    ap.add_argument("--timeout", type=float, help="seconds per workload (default: its manifest timeout_s)")
    ap.add_argument("--pack", action="store_true", help="run jobs concurrently on disjoint GPU sets")
    ap.add_argument("--all-gpus", action="store_true", help="leave every GPU visible to each job (not recommended)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--mirror", default=os.environ.get("PW_MODEL_MIRROR"))
    ap.add_argument("--strict-mirror", action="store_true", help="a model missing from the mirror fails instead of using the Hub")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--remote-dir", default="~/pantheonworkloads")
    ap.add_argument("--lock-dir", default="/tmp", help="where the rig keeps its per-GPU lock files")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    names = list(a.workload)
    if a.list:
        names += [x.strip() for x in pathlib.Path(a.list).read_text().splitlines() if x.strip() and not x.startswith("#")]
    if a.tier:
        names += wl.in_tier(a.tier)
    names = list(dict.fromkeys(names))
    if not names:
        ap.error("name workloads, --list or --tier")
    try:
        tag = riglib.norm_tag(a.tag)
        for n in names:
            if not (wl.WORKLOADS / n / "manifest.yaml").is_file():
                raise riglib.RigError(f"no such workload: {n}")
        rig = riglib.load_rig(tag)
        if rig is None or rig.get("state") != "running":
            raise riglib.RigError(f"{tag} is not a running rig launched from this machine")
        pw = load_pw()
        gpus = remote_gpus(rig)
        jobs, skipped = plan(names, gpus, pw, a.all_gpus)
    except riglib.RigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{tag}: {pw.describe_gpus(gpus)}; {len(jobs)} to run, {len(skipped)} skipped")
    for n, why in skipped:
        print(f"SKIP    {n}  ({why})")
    if a.dry_run:
        for j in jobs:
            print(f"WOULD   {j['name']}  {j['need']} GPU(s), timeout {a.timeout or j['timeout']:g} s")
        return 0
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    batch, n = stamp, 0
    while True:                      # two batches in one second must not share a directory
        local_out = riglib.rig_dir(tag) / "results" / batch
        try:
            local_out.mkdir(parents=True)
            break
        except FileExistsError:
            n += 1
            batch = f"{stamp}-{n}"
    remote_out = f"pw-batch/{batch}"
    # the remote dir is relative to the ssh login directory, like ~/pantheonworkloads
    print(f"results: {local_out}")

    def can_start(job):
        left = rig["deadline"] - riglib.now()
        ok = left > (a.timeout or job["timeout"]) + OUTER_GRACE_S + DEADLINE_MARGIN_S
        if not ok:
            print(f"STOP    {job['name']}: {left / 60:.0f} min to the rig's deadline is too little for its timeout; extend.sh or relaunch", file=sys.stderr)
        return ok

    def run(job, gset):
        row = run_job(rig, job, gset, a, remote_out, local_out)
        print(f"{row['result']:9} {row['workload']}  gpus {row['gpus']}  {row['seconds']:.0f}s  {row.get('detail', '')[:120]}", flush=True)
        return row

    rows, not_run = schedule(jobs, len(gpus), run, a.pack, can_start)
    with open(local_out / "batch.tsv", "w") as f:
        f.write("workload\tresult\tgpus\tseconds\tdetail\n")
        for r in rows:
            f.write("\t".join([r["workload"], r["result"], r["gpus"], str(r["seconds"]), r.get("detail", "").replace("\t", " ").replace("\n", " ")]) + "\n")
        for n, why in skipped:
            f.write(f"{n}\tSKIP-REQUIRES\t\t0\t{why}\n")
        for n in not_run:
            f.write(f"{n}\tNOT-RUN\t\t0\tstopped before the rig's deadline\n")
    bad = [r for r in rows if r["result"] in ("FAIL", "TIMEOUT", "BUSY", "NO-RESULT")]
    print(f"{len(rows) - len(bad)} ok/skipped by the runner, {len(bad)} bad, {len(skipped)} skipped on requires, {len(not_run)} not run; summary {local_out / 'batch.tsv'}")
    return 1 if bad or not_run else 0


if __name__ == "__main__":
    sys.exit(main())
