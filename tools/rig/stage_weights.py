#!/usr/bin/env python3
"""Stage the weights of catalog workloads into a mirror (S3 prefix or local directory) from a cheap CPU box.

    stage_weights.py [workload ...] [--tier A|B|2b|big] --mirror s3://BUCKET/pw-weights|/dir [options]

For each workload it reads workloads/<w>/model.sha256 (the pins and, in its header comment, `<repo> @ <commit>`) and the
file list in tools/catalog/hub.json, and stages under <mirror>/<repo>/<commit>/: the pinned weight files plus the small
config/tokenizer files the loaders also fetch. Each file is downloaded from the Hub (curl, resumable), its sha256 checked
against the pin (or the Hub's LFS oid, or computed for small non-LFS files), uploaded, and listed in manifest.json, which is
rewritten after every file so an interrupted run resumes and a model is only complete when the manifest says so. Files
already in the manifest with the right size and sha256 are skipped. Workloads sharing a repository and commit (a
functional workload and its bench twin) are staged once.

    --dry-run   print the plan (files, GiB, what is already staged) and touch nothing
    --check     read the mirror's manifests only; exit 1 when any model is missing or incomplete (no download)
    --status    print the progress of the running (or last) staging run from <state-dir>/status.json and exit
    --scratch   where files are downloaded (default ~/.cache/pw-stage); needs room for --batch-gb

Polite mode, for a shared box (docs/rigs.md "Staging from a shared box"): curl and aws run under `nice -n 19` and
`ionice -c3` (--no-nice to disable), --threads files at once, --rate-limit MB/s total for downloads and uploads,
--aws-max-concurrent uploads at once; everything pauses (partial files kept) while a GitHub Actions `Runner.Worker`
runs, the 1-minute load average is above --max-load, or free space is below --min-free-gb; at most --batch-gb is staged
but not yet uploaded; a local copy is deleted once S3 holds an object of the same size. tools/rig/stage_supervisor.sh
restarts it, single-instance, from cron.

Consumers set PW_MODEL_MIRROR to the same --mirror value (tools/rig/mirror.py, docs/rigs.md). HF_TOKEN is sent when set.
Environment for tests: PW_AWS (aws command), PW_CURL (curl command), PW_PGREP (pgrep command).
"""
import argparse
import fnmatch
import fcntl
import json
import logging
import logging.handlers
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mirror as mirrorlib  # noqa: E402
import wl  # noqa: E402

SMALL = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "*.yaml", "*.py", "tokenizer*", "*.vocab", "*.bpe"]
SMALL_MAX = 64 << 20          # a non-pinned file bigger than this is not a "small config"; it is left out and reported
HUB = "https://huggingface.co"


def hub_files(repo, rev):
    path = wl.ROOT / "tools" / "catalog" / "hub.json"
    repos = json.loads(path.read_text())["repos"]
    info = repos.get(repo)
    if not info or info.get("sha") != rev:
        return None
    return {f[0]: {"size": f[1], "lfs": f[2]} for f in info["files"]}


def plan_model(repo, rev, pins):
    """[{path, size, sha256 (maybe None), pinned}], or raises ValueError when the pins and the catalog disagree."""
    files = hub_files(repo, rev)
    if files is None:
        raise ValueError(f"{repo}@{rev[:8]} is not in tools/catalog/hub.json at that commit (cannot list its files)")
    out = []
    for path, sha in sorted(pins.items()):
        meta = files.get(path)
        if meta is None:
            raise ValueError(f"{repo}: pinned file {path} is not in the catalog's file list")
        if meta["lfs"] and meta["lfs"] != sha:
            raise ValueError(f"{repo}: {path} is pinned to {sha} but the Hub's LFS oid is {meta['lfs']}")
        out.append({"path": path, "size": meta["size"], "sha256": sha, "pinned": True})
    for path, meta in sorted(files.items()):
        if path in pins or not any(fnmatch.fnmatch(path, g) for g in SMALL):
            continue
        if meta["size"] > SMALL_MAX:
            print(f"note: {repo}: {path} ({meta['size']} bytes) matches a small-file pattern but is large; not staged", file=sys.stderr)
            continue
        out.append({"path": path, "size": meta["size"], "sha256": meta["lfs"], "pinned": False})
    return out


def groups(workloads):
    """{(repo, rev): {"workloads": [...], "pins": {path: sha}}}; a workload with no repo pin goes to the second result."""
    g, unstageable = {}, []
    for w in workloads:
        repo, rev, pins = wl.read_pin(w)
        if not repo or not pins:
            unstageable.append(w)
            continue
        e = g.setdefault((repo, rev), {"workloads": [], "pins": {}})
        e["workloads"].append(w)
        for p, s in pins.items():
            if e["pins"].setdefault(p, s) != s:
                raise ValueError(f"{w} pins {p} differently from {e['workloads'][0]}")
    return g, unstageable


def staged_manifest(mirror, repo, rev):
    try:
        return mirrorlib.read_manifest(mirror, repo, rev)
    except mirrorlib.Miss:
        return None


def already(manifest, f):
    m = (manifest or {}).get("files", {}).get(f["path"])
    return bool(m) and m["size"] == f["size"] and (f["sha256"] is None or m["sha256"] == f["sha256"])


class Stopped(Exception):
    """SIGTERM/SIGINT: leave partial files in place and unwind."""


class Ctx:
    """Everything the transfer functions need: politeness settings, the gate, logging, shared progress."""

    def __init__(self, a, scratch, state_dir):
        self.mirror = a.mirror
        self.hf_base = a.hf_base
        self.scratch = pathlib.Path(scratch)
        self.state_dir = pathlib.Path(state_dir)
        self.poll = a.poll_s
        self.rate = a.rate_limit               # MiB/s over all downloads (and over all uploads); 0 = unlimited
        self.threads = max(1, a.threads)
        self.aws_max = max(1, a.aws_max_concurrent)
        self.max_load = a.max_load
        self.min_free = a.min_free_gb * 2**30
        self.nice = a.nice
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.upload_sem = threading.Semaphore(self.aws_max)
        self.paused = None                     # (reason, since_monotonic) while the gate is closed
        self.paused_total = 0.0
        self.aws_env = None
        self.active = {}                       # path -> {"phase", "size", "part"}
        self.logger = None

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self.lock:
            print(line, flush=True)
            if self.logger:
                self.logger.info(msg)

    def prefix(self):
        if not self.nice:
            return []
        p = ["nice", "-n", "19"]
        if shutil.which("ionice"):
            p += ["ionice", "-c3"]
        return p

    # gate: pause while CI runs, the box is loaded, or the disk is nearly full
    def workers_running(self):
        p = subprocess.run([os.environ.get("PW_PGREP") or "pgrep", "-x", "Runner.Worker"], capture_output=True, text=True)
        return len(p.stdout.split()) if p.returncode == 0 else 0

    def reason(self):
        n = self.workers_running()
        if n:
            return f"{n} Runner.Worker process(es) running (CI goes first)"
        load = os.getloadavg()[0]
        if load > self.max_load:
            return f"1-minute load average {load:.1f} is above --max-load {self.max_load:g}"
        free = shutil.disk_usage(self.scratch).free
        if free < self.min_free:
            return f"free space {free / 2**30:.0f} GiB is below --min-free-gb {self.min_free / 2**30:g}"
        return None

    def note(self, reason):
        with self.lock:
            if reason and self.paused is None:
                self.log(f"PAUSE: {reason}")
                self.paused = (reason, time.monotonic())
            elif reason and self.paused[0] != reason:
                self.log(f"PAUSE (reason now): {reason}")
                self.paused = (reason, self.paused[1])
            elif not reason and self.paused is not None:
                dur = time.monotonic() - self.paused[1]
                self.paused_total += dur
                self.log(f"RESUME after {dur:.0f}s (was: {self.paused[0]})")
                self.paused = None

    def wait_gate(self):
        """Block while any gate is closed, rechecking every poll seconds."""
        while True:
            if self.stop.is_set():
                raise Stopped()
            r = self.reason()
            self.note(r)
            if not r:
                return
            self.stop.wait(self.poll)

    def share(self, n):
        """Per-process bandwidth cap in KiB/s so n processes together stay within --rate-limit; 0 = none."""
        return max(1, int(self.rate * 1024 / n)) if self.rate else 0

    def aws_cmd(self, args):
        return self.prefix() + [os.environ.get("PW_AWS") or "aws", *args]


def write_aws_config(ctx):
    """An AWS config for the upload processes: bandwidth share and request concurrency. Returns the env for aws."""
    path = ctx.state_dir / "aws-config"
    kib = ctx.share(ctx.aws_max)
    s3 = f"  max_concurrent_requests = {min(4, max(1, ctx.aws_max))}\n  preferred_transfer_client = classic\n"
    if kib:
        s3 += f"  max_bandwidth = {kib}KB/s\n"
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    body = ""
    for head in ("default", f"profile {os.environ['AWS_PROFILE']}" if os.environ.get("AWS_PROFILE") else None):
        if head:
            body += f"[{head}]\n" + (f"region = {region}\n" if region else "") + "s3 =\n" + s3
    path.write_text(body)
    env = dict(os.environ)
    env["AWS_CONFIG_FILE"] = str(path)
    return env


def run_polled(ctx, cmd, env=None, on_closed="freeze"):
    """Run cmd, rechecking the gate every poll seconds. on_closed 'kill': terminate the process and return (None, "")
    (the caller waits and restarts it); 'freeze': SIGSTOP its process group until the gate opens, then SIGCONT.
    Returns (returncode, stderr)."""
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
    while True:
        try:
            _, err = p.communicate(timeout=ctx.poll)
            return p.returncode, err
        except subprocess.TimeoutExpired:
            if ctx.stop.is_set():
                os.killpg(p.pid, signal.SIGKILL)
                p.communicate()
                raise Stopped()
            r = ctx.reason()
            if not r:
                continue
            ctx.note(r)
            if on_closed == "kill":
                os.killpg(p.pid, signal.SIGTERM)
                p.communicate()
                return None, ""
            os.killpg(p.pid, signal.SIGSTOP)
            try:
                ctx.wait_gate()
            finally:
                os.killpg(p.pid, signal.SIGCONT)


def download(ctx, repo, rev, path, dest, expect_size):
    url = f"{ctx.hf_base.rstrip('/')}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
    kib = ctx.share(ctx.threads)
    while True:
        if os.path.exists(dest):
            have = os.path.getsize(dest)
            if have == expect_size:
                break       # an earlier run finished the download but not the upload; the sha256 check follows
            if have > expect_size:
                os.unlink(dest)
        cmd = ctx.prefix() + [os.environ.get("PW_CURL") or "curl", "-fL", "--retry", "3", "-C", "-", "-sS", "-o", str(dest)]
        if kib:
            cmd += ["--limit-rate", f"{kib}k"]
        if os.environ.get("HF_TOKEN"):
            cmd += ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
        rc, err = run_polled(ctx, cmd + [url], on_closed="kill")
        if rc is None:                       # stopped for the gate; the partial file is kept and curl resumes it
            ctx.wait_gate()
            continue
        if rc != 0:
            raise OSError(f"download of {url} failed (curl exit {rc}): {err.strip()[-200:]}; the partial file is kept for a resume")
        break
    if os.path.getsize(dest) != expect_size:
        raise OSError(f"{path}: downloaded {os.path.getsize(dest)} bytes, the catalog says {expect_size}")


def remote_size(ctx, repo, rev, name):
    """Size of the object in the mirror, or None when it is absent."""
    loc = mirrorlib.location(ctx.mirror, repo, rev) + "/" + name
    if not mirrorlib.is_s3(ctx.mirror):
        return os.path.getsize(loc) if os.path.isfile(loc) else None
    bucket, _, key = loc[len("s3://"):].partition("/")
    p = subprocess.run(ctx.aws_cmd(["s3api", "head-object", "--bucket", bucket, "--key", key, "--query", "ContentLength", "--output", "text"]),
                       capture_output=True, text=True, env=ctx.aws_env)
    if p.returncode != 0:
        return None
    try:
        return int(p.stdout.strip())
    except ValueError:
        return None


def put(ctx, repo, rev, name, src):
    dst = mirrorlib.location(ctx.mirror, repo, rev) + "/" + name
    if mirrorlib.is_s3(ctx.mirror):
        cmd = ctx.aws_cmd(["s3", "cp", "--only-show-errors", str(src), dst])
        with ctx.upload_sem:
            ctx.wait_gate()
            rc, err = run_polled(ctx, cmd, env=ctx.aws_env, on_closed="freeze")
        if rc != 0:
            raise OSError(f"aws s3 cp {name} failed: {err.strip()[-200:]}")
    else:
        pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)


def write_manifest(ctx, repo, rev, files, complete, d):
    m = {"schema": mirrorlib.SCHEMA, "repo": repo, "revision": rev, "complete": complete,
         "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "total_bytes": sum(f["size"] for f in files.values()), "files": files}
    tmp = pathlib.Path(d) / "manifest.json"
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    put(ctx, repo, rev, "manifest.json", tmp)


class Budget:
    """Bytes staged on local disk but not yet uploaded; a single file larger than the limit may run alone."""

    def __init__(self, limit):
        self.limit, self.used, self.cv = limit, 0, threading.Condition()

    def acquire(self, n, stop):
        with self.cv:
            while self.used and self.used + n > self.limit:
                if stop.is_set():
                    raise Stopped()
                self.cv.wait(1)
            self.used += n

    def release(self, n):
        with self.cv:
            self.used -= n
            self.cv.notify_all()


class Progress:
    def __init__(self, ctx, total, preexisting, n_models, n_staged):
        self.ctx, self.total, self.pre = ctx, total, preexisting
        self.n_models, self.n_staged = n_models, n_staged
        self.done = 0            # bytes verified in the mirror by this run
        self.t0 = time.monotonic()
        self.samples = [(self.t0, 0)]
        self.errors = 0

    def rates(self):
        now = time.monotonic()
        el = now - self.t0
        sustained = self.done / el if el > 0 and self.done else None
        old = next((s for s in self.samples if now - s[0] <= 600), self.samples[0])
        recent = (self.done - old[1]) / (now - old[0]) if self.done and now - old[0] > 0 else None
        return sustained, recent

    def snapshot(self, running=True):
        sustained, recent = self.rates()
        left = self.total - self.pre - self.done
        ctx = self.ctx
        with ctx.lock:
            current = [{"path": k, "phase": v["phase"], "size": v["size"],
                        "have": os.path.getsize(v["part"]) if os.path.exists(v["part"]) else 0} for k, v in list(ctx.active.items())]
            paused = {"reason": ctx.paused[0], "since": time.time() - (time.monotonic() - ctx.paused[1])} if ctx.paused else None
        return {"pid": os.getpid(), "running": running, "updated": time.time(), "started": time.time() - (time.monotonic() - self.t0),
                "mirror": ctx.mirror, "total_bytes": self.total, "done_bytes": self.pre + self.done, "run_bytes": self.done,
                "models_total": self.n_models, "models_staged": self.n_staged, "current": current, "paused": paused,
                "paused_seconds": ctx.paused_total, "rate_sustained": sustained, "rate_recent": recent,
                "eta_seconds": (left / sustained) if sustained and left > 0 else (0 if left <= 0 else None), "errors": self.errors}

    def write(self, running=True):
        self.samples.append((time.monotonic(), self.done))
        self.samples = self.samples[-400:]
        p = self.ctx.state_dir / "status.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.snapshot(running), indent=1))
        os.replace(tmp, p)


def fmt_dur(s):
    if s is None:
        return "unknown (nothing staged yet to measure a rate from)"
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    return (f"{d}d " if d else "") + f"{h}h {s // 60}m"


def print_status(state_dir):
    p = pathlib.Path(state_dir) / "status.json"
    if not p.is_file():
        print(f"no status at {p}: no staging run has started from this state dir")
        return 1
    s = json.loads(p.read_text())
    alive = False
    try:
        os.kill(s["pid"], 0)
        alive = s.get("running", False)
    except (OSError, KeyError):
        pass
    age = time.time() - s["updated"]
    gib = 2**30
    print(f"pw-stage {'RUNNING' if alive else 'NOT RUNNING'} (pid {s['pid']}, status updated {age:.0f}s ago) -> {s['mirror']}")
    if alive and s.get("paused"):
        print(f"state: PAUSED for {time.time() - s['paused']['since']:.0f}s: {s['paused']['reason']}")
    else:
        print("state: " + ("transferring" if alive else "stopped"))
    pct = 100 * s["done_bytes"] / s["total_bytes"] if s["total_bytes"] else 100
    print(f"models staged: {s['models_staged']}/{s['models_total']}")
    print(f"data: {s['done_bytes'] / gib:.1f} / {s['total_bytes'] / gib:.1f} GiB ({pct:.1f}%), {s['run_bytes'] / gib:.1f} GiB by this run")
    for c in s["current"]:
        print(f"current: {c['path']} [{c['phase']}] {c['have'] / gib:.2f}/{c['size'] / gib:.2f} GiB")

    def mb(r):
        return "n/a" if r is None else f"{r / 2**20:.1f} MiB/s"
    print(f"rate: sustained {mb(s['rate_sustained'])} (whole run, pauses included), last 10 min {mb(s['rate_recent'])}; paused {s['paused_seconds']:.0f}s so far")
    print(f"ETA: {fmt_dur(s['eta_seconds'])} at the sustained rate" + (f"; {s['errors']} file error(s) so far" if s["errors"] else ""))
    return 0


def stage_all(ctx, models, budget_bytes, order, total_bytes, n_models_all):
    """models: [{repo, rev, plan, manifest, workloads}] still to stage. Returns the number of errors."""
    from concurrent.futures import ThreadPoolExecutor
    for m in models:
        m["files"] = dict((m["manifest"] or {}).get("files", {}))
        m["missing"] = [f for f in m["plan"] if not already(m["manifest"], f)]
        m["lock"] = threading.Lock()
        m["errors"] = []
        m["staged"] = 0
        m["pending"] = len(m["missing"])
        m["dir"] = ctx.scratch / m["repo"].replace("/", "__") / m["rev"]
        m["dir"].mkdir(parents=True, exist_ok=True)
        m["size"] = sum(f["size"] for f in m["plan"])
    if order == "smallest-first":
        models.sort(key=lambda m: (m["size"], m["repo"]))
    todo = sum(f["size"] for m in models for f in m["missing"])
    prog = Progress(ctx, total_bytes, total_bytes - todo, n_models_all, n_models_all - len(models))
    budget = Budget(budget_bytes)
    ctx.aws_env = write_aws_config(ctx) if mirrorlib.is_s3(ctx.mirror) else None

    def finish(m):
        """Called with m['lock'] held (or single-threaded) when the last file of a model is done."""
        if m["errors"]:
            return
        write_manifest(ctx, m["repo"], m["rev"], m["files"], True, m["dir"])
        prog.n_staged += 1
        ctx.log(f"MODEL STAGED {m['repo']}@{m['rev'][:8]} ({m['size'] / 2**30:.1f} GiB; {prog.n_staged}/{prog.n_models})")
        print(f"  staged {m['staged']}, already there {len(m['plan']) - len(m['missing'])}   [{m['repo']}]", flush=True)

    def work(m, f):
        if ctx.stop.is_set():
            return
        part = m["dir"] / (f["path"].replace("/", "__") + ".part")
        held = False
        try:
            ctx.wait_gate()
            budget.acquire(f["size"], ctx.stop)
            held = True
            ctx.wait_gate()
            ctx.active[f["path"]] = {"phase": "downloading", "size": f["size"], "part": str(part)}
            ctx.log(f"START {m['repo']} {f['path']} ({f['size'] / 2**30:.2f} GiB)")
            t0 = time.monotonic()
            download(ctx, m["repo"], m["rev"], f["path"], part, f["size"])
            ctx.active[f["path"]]["phase"] = "hashing"
            got = mirrorlib.sha256_file(part)         # 4 MiB chunks; never the whole file in memory
            if f["sha256"] and got != f["sha256"]:
                part.unlink(missing_ok=True)   # a corrupt download is not resumed
                raise ValueError(f"{f['path']}: sha256 of the download is {got}, expected {f['sha256']}; not uploaded")
            ctx.active[f["path"]]["phase"] = "uploading"
            put(ctx, m["repo"], m["rev"], f["path"], part)
            rs = remote_size(ctx, m["repo"], m["rev"], f["path"])
            if rs != f["size"]:
                raise OSError(f"{f['path']}: the mirror holds {rs} bytes after the upload, expected {f['size']}; local copy kept, not recorded")
            with m["lock"]:
                m["files"][f["path"]] = {"size": f["size"], "sha256": got, "pinned": f["pinned"],
                                         "source": "pin" if f["pinned"] else ("lfs-oid" if f["sha256"] else "computed")}
                write_manifest(ctx, m["repo"], m["rev"], m["files"], False, m["dir"])
                m["staged"] += 1
            part.unlink(missing_ok=True)            # verified in the mirror: free the disk
            prog.done += f["size"]
            sus, _ = prog.rates()
            ctx.log(f"DONE {m['repo']} {f['path']} in {time.monotonic() - t0:.0f}s; run total {prog.done / 2**30:.2f} GiB, "
                    f"sustained {'n/a' if sus is None else f'{sus / 2**20:.1f} MiB/s'}")
            prog.write()
        except Stopped:
            pass
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            with m["lock"]:
                m["errors"].append(str(e))
            prog.errors += 1
            ctx.log(f"ERROR {m['repo']}: {e}")
            print(f"  ERROR {e}", file=sys.stderr, flush=True)
        finally:
            ctx.active.pop(f["path"], None)
            if held:
                budget.release(f["size"])
            with m["lock"]:
                m["pending"] -= 1
                if m["pending"] == 0 and not ctx.stop.is_set():
                    try:
                        finish(m)
                    except (OSError, ValueError) as e:
                        m["errors"].append(str(e))
                        prog.errors += 1
                        ctx.log(f"ERROR {m['repo']}: {e}")

    # a model whose files are all there but whose manifest still says incomplete only needs the final manifest
    for m in models:
        if m["pending"] == 0:
            try:
                finish(m)
            except (OSError, ValueError) as e:
                m["errors"].append(str(e))
                prog.errors += 1
    prog.write()
    ticker_stop = threading.Event()

    def ticker():
        while not ticker_stop.wait(10):
            prog.write()
    threading.Thread(target=ticker, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=ctx.threads) as ex:
            for m in models:
                for f in m["missing"]:
                    ex.submit(work, m, f)
    finally:
        ticker_stop.set()
        prog.write(running=False)
    sus, _ = prog.rates()
    ctx.log(f"SUMMARY: {prog.done / 2**30:.2f} GiB staged this run in {time.monotonic() - prog.t0:.0f}s "
            f"(paused {ctx.paused_total:.0f}s), sustained {'n/a' if sus is None else f'{sus / 2**20:.2f} MiB/s'}, "
            f"{prog.n_staged}/{prog.n_models} models staged")
    return sum(len(m["errors"]) for m in models)


def self_nice():
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 19)
        if shutil.which("ionice"):
            subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], capture_output=True)
    except OSError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workload", nargs="*")
    ap.add_argument("--tier", choices=["A", "B", "2b", "big"])
    ap.add_argument("--mirror", default=os.environ.get("PW_MODEL_MIRROR"), required=False)
    ap.add_argument("--hf-base", default=HUB)
    ap.add_argument("--scratch", default=os.path.expanduser("~/.cache/pw-stage"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--status", action="store_true", help="print the state of the staging run and exit")
    ap.add_argument("--state-dir", default=os.environ.get("PW_STAGE_HOME") or os.path.expanduser("~/.pw-stage"),
                    help="status.json, the run lock and the generated aws config (default ~/.pw-stage)")
    ap.add_argument("--log-file", help="also log to this file (rotated at 10 MB, 5 kept)")
    ap.add_argument("--nice", action=argparse.BooleanOptionalAction, default=True, help="run curl and aws under nice -n 19 and ionice -c3 (default on)")
    ap.add_argument("--threads", type=int, default=4, help="files transferred at once (default 4)")
    ap.add_argument("--rate-limit", type=float, default=20, metavar="MBPS", help="MiB/s over all downloads, and over all uploads (default 20; 0 = unlimited)")
    ap.add_argument("--aws-max-concurrent", type=int, default=2, help="simultaneous aws uploads (default 2)")
    ap.add_argument("--max-load", type=float, default=12, help="pause while the 1-minute load average is above this (default 12)")
    ap.add_argument("--min-free-gb", type=float, default=500, help="pause while the scratch filesystem has less free space (default 500)")
    ap.add_argument("--batch-gb", type=float, default=1500, help="most GiB staged locally and not yet uploaded (default 1500)")
    ap.add_argument("--order", choices=["smallest-first", "manifest"], default="smallest-first")
    ap.add_argument("--poll-s", type=float, default=30, help="seconds between gate checks while waiting or transferring (default 30)")
    a = ap.parse_args(argv)
    if a.status:
        return print_status(a.state_dir)
    workloads = list(a.workload) + (wl.in_tier(a.tier) if a.tier else [])
    if not workloads:
        ap.error("name workloads or pass --tier")
    if not a.mirror:
        ap.error("--mirror (or PW_MODEL_MIRROR) is required")
    for w in workloads:
        if not (wl.WORKLOADS / w / "manifest.yaml").is_file():
            print(f"no such workload: {w}", file=sys.stderr)
            return 2
    try:
        g, unstageable = groups(workloads)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if unstageable:
        print(f"not stageable (no `# repo @ commit` header and pins in model.sha256): {', '.join(unstageable)}", file=sys.stderr)
    bad = 0
    total = todo = 0
    pending = []
    for (repo, rev), e in sorted(g.items()):
        try:
            plan = plan_model(repo, rev, e["pins"])
        except ValueError as err:
            print(f"ERROR {err}", file=sys.stderr)
            bad += 1
            continue
        manifest = staged_manifest(a.mirror, repo, rev)
        missing = [f for f in plan if not already(manifest, f)]
        size = sum(f["size"] for f in plan)
        need = sum(f["size"] for f in missing)
        total += size
        todo += need
        done = bool(manifest) and manifest.get("complete") and not missing
        state = "staged" if done else f"{len(missing)}/{len(plan)} files to stage"
        print(f"{repo}@{rev[:8]}  {size / 2**30:8.1f} GiB  {state}  ({', '.join(e['workloads'])})")
        if a.check:
            bad += 0 if done else 1
        elif not (a.dry_run or done):
            pending.append({"repo": repo, "rev": rev, "plan": plan, "manifest": manifest, "workloads": e["workloads"]})
    if pending:
        os.makedirs(a.scratch, exist_ok=True)
        os.makedirs(a.state_dir, exist_ok=True)
        lockf = open(os.path.join(a.state_dir, "stage.lock"), "w")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another staging run holds {a.state_dir}/stage.lock; exiting", file=sys.stderr)
            return 4
        ctx = Ctx(a, a.scratch, a.state_dir)
        if a.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(a.log_file)), exist_ok=True)
            ctx.logger = logging.getLogger(f"pw-stage-{id(ctx)}")
            ctx.logger.setLevel(logging.INFO)
            ctx.logger.propagate = False
            h = logging.handlers.RotatingFileHandler(a.log_file, maxBytes=10 << 20, backupCount=5)
            h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            ctx.logger.addHandler(h)
        old = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                old[sig] = signal.signal(sig, lambda *_: ctx.stop.set())
        ctx.log(f"start: {len(pending)} model(s) to stage, threads {ctx.threads}, rate limit {ctx.rate:g} MiB/s, "
                f"aws uploads {ctx.aws_max}, nice {'on' if ctx.nice else 'off'}, max-load {ctx.max_load:g}, min-free {a.min_free_gb:g} GB, "
                f"batch {a.batch_gb:g} GB, order {a.order}")
        try:
            bad += stage_all(ctx, pending, int(a.batch_gb * 2**30), a.order, total, len(g))
        finally:
            for sig, h in old.items():
                signal.signal(sig, h)
            if ctx.logger:
                for h in list(ctx.logger.handlers):
                    h.close()
                    ctx.logger.removeHandler(h)
        if ctx.stop.is_set():
            print("stopped by signal; partial files kept", file=sys.stderr)
            return 143
    print(f"total {total / 2**30:.1f} GiB in {len(g)} model(s); {todo / 2**30:.1f} GiB not yet staged")
    if not a.check and not a.dry_run:
        print(f"consumers: export PW_MODEL_MIRROR={a.mirror}")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--status" not in sys.argv and "--dry-run" not in sys.argv and "--no-nice" not in sys.argv:
        self_nice()
    sys.exit(main())
