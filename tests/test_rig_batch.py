"""Tests for tools/rig/batch.py (run_batch.sh): skipping on `requires`, one job at a time, --pack on disjoint GPU sets,
per-workload timeout, busy-GPU refusal, copying results back, the rig deadline guard. A fake ssh/scp run the "remote"
commands locally against a fake repository whose bin/pw only logs how it was called. No AWS, no GPU.

    python3 -m unittest discover -s tests -p 'test_rig*.py' -v
"""
import contextlib
import fcntl
import io
import json
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKES = ROOT / "tests" / "rig_fakes"
sys.path.insert(0, str(ROOT / "tools" / "rig"))
import batch  # noqa: E402
import rig  # noqa: E402
import wl  # noqa: E402

NOW = 1_800_000_000.0
TIER_A, TIER_2B, TIER_BIG = wl.in_tier("A")[:4], wl.in_tier("2b"), wl.in_tier("big")   # read the manifests once

FAKE_PW = r'''#!/usr/bin/env bash
# Fake bin/pw: logs how it was called, writes <out>/<name>.json like the real runner.
name=$2; out=""; bench=0; repeat=""
shift 2
while [[ $# -gt 0 ]]; do case $1 in --out) out=$2; shift 2;; --bench) bench=1; shift;; --repeat) repeat=$2; shift 2;; *) shift;; esac; done
start=$(date +%s.%N)
if [[ "${FAKE_HANG:-}" == "$name" ]]; then exec sleep 100; fi
sleep "${FAKE_SLEEP:-0}"
res=PASS; [[ "${FAKE_FAIL:-}" == "$name" ]] && res=FAIL
end=$(date +%s.%N)
mkdir -p "$out"
printf '{"workload": "%s", "target": "gpu", "result": "%s", "seconds": 1, "detail": "fake"}\n' "$name" "$res" > "$out/$name.json"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${CUDA_VISIBLE_DEVICES-unset}" "${PW_MODEL_MIRROR-}" "${PW_MODEL_MIRROR_STRICT-}" \
  "${PW_TIMEOUT_S-}" "$bench:$repeat:${EXTRA_KNOB-}" "$start" "$end" >> "$FAKE_PW_LOG"
[[ $res == PASS ]]
'''


class BatchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="batchtest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.remote = self.tmp / "remote"
        (self.remote / "bin").mkdir(parents=True)
        pw = self.remote / "bin" / "pw"
        pw.write_text(FAKE_PW)
        pw.chmod(pw.stat().st_mode | stat.S_IXUSR)
        self.locks = self.tmp / "locks"
        self.locks.mkdir()
        env = {"PW_SSH": str(FAKES / "ssh"), "PW_SCP": str(FAKES / "scp"), "FAKE_SSH_LOG": str(self.tmp / "ssh.log"),
               "FAKE_BOX_DIR": str(self.tmp / "box"), "FAKE_AWS_STATE": str(self.tmp / "aws.json"),
               "FAKE_PW_LOG": str(self.tmp / "pw.log"), "PW_RIGS_HOME": str(self.tmp / "rigs"), "PW_RIGS_ALLOW_UNSAFE_HOME": "1",
               "PW_RIG_NOW": str(NOW), "HOME": str(self.home), "PATH": f"{FAKES}:{os.environ['PATH']}",
               "PW_BATCH_GRACE_S": "1", "PW_BATCH_KILL_S": "1"}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        for k in ("FAKE_GPUS", "FAKE_HANG", "FAKE_FAIL", "FAKE_SLEEP", "PW_MODEL_MIRROR", "PW_IGNORE_REQUIRES"):
            os.environ.pop(k, None)
        # the module-level constants read the environment at import; follow the test's
        mg = mock.patch.object(batch, "OUTER_GRACE_S", 1.0)
        mk = mock.patch.object(batch, "KILL_AFTER_S", 1.0)
        mg.start(); mk.start()
        self.addCleanup(mg.stop); self.addCleanup(mk.stop)
        self.make_rig("pw-b1", deadline=NOW + 6 * 3600)
        self.one = TIER_A[0]
        self.eight = next(n for n in TIER_2B if wl.requires(n)["gpus"] == 8)
        self.big = TIER_BIG[0]

    def make_rig(self, tag, deadline, state="running"):
        d = self.tmp / "rigs" / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / "key.pem").write_text("k")
        (d / "rig.json").write_text(json.dumps({"tag": tag, "state": state, "ip": "203.0.113.10", "user": "ubuntu", "deadline": deadline,
                                                "region": "us-east-1", "hourly": 41.0, "max_dollars": 500, "created": NOW, "launched": NOW}))

    def batch(self, *argv, tag="pw-b1"):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = batch.main([tag, *argv, "--remote-dir", str(self.remote), "--lock-dir", str(self.locks)])
        return rc, out.getvalue(), err.getvalue()

    def pw_log(self):
        f = self.tmp / "pw.log"
        rows = [l.split("\t") for l in f.read_text().splitlines()] if f.exists() else []
        return [{"name": r[0], "cvd": r[1], "mirror": r[2], "strict": r[3], "timeout": r[4], "knob": r[5], "start": float(r[6]), "end": float(r[7])} for r in rows]

    def results_dir(self):
        return sorted((self.tmp / "rigs" / "pw-b1" / "results").iterdir())[-1]

    def ssh_cmds(self):
        f = self.tmp / "ssh.log"
        return [json.loads(l)["cmd"] for l in f.read_text().splitlines()] if f.exists() else []


class TestPlanAndSkips(BatchCase):
    def test_workloads_the_rig_cannot_host_are_skipped_without_running(self):
        cpu_only = next(n for n in wl.names() if "gpu" not in (wl.manifest(n).get("targets") or []))
        rc, out, _ = self.batch(self.one, self.big, cpu_only)
        self.assertEqual(rc, 0, out)
        self.assertEqual([r["name"] for r in self.pw_log()], [self.one])
        self.assertRegex(out, rf"SKIP\s+{self.big}\s+\(needs")
        self.assertIn("this host has 8 x NVIDIA A100-SXM4-80GB 80 GiB", out)
        self.assertIn(f"SKIP    {cpu_only}  (no gpu target)", out)
        rows = (self.results_dir() / "batch.tsv").read_text()
        self.assertIn(f"{self.big}\tSKIP-REQUIRES", rows)
        self.assertNotIn(self.big, " ".join(self.ssh_cmds()).replace("nvidia-smi", ""))

    def test_a_small_rig_skips_the_eight_gpu_workload(self):
        os.environ["FAKE_GPUS"] = "NVIDIA A10G, 23028, 8.6, 595.1\nNVIDIA A10G, 23028, 8.6, 595.1"
        rc, out, _ = self.batch(self.eight, self.one)
        self.assertEqual([r["name"] for r in self.pw_log()], [self.one])
        self.assertIn("this host has 2 x NVIDIA A10G", out)

    def test_dry_run_only_inspects_the_gpus(self):
        rc, out, _ = self.batch(self.one, self.eight, "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn(f"WOULD   {self.one}", out)
        self.assertEqual(self.pw_log(), [])
        self.assertEqual(len(self.ssh_cmds()), 1)

    def test_tier_selects_from_the_manifests(self):
        rc, out, _ = self.batch("--tier", "big", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn(f"SKIP    {self.big}", out)

    def test_a_rig_that_is_not_running_is_refused(self):
        self.make_rig("pw-b2", NOW + 3600, state="terminated")
        rc, _, err = self.batch(self.one, tag="pw-b2")
        self.assertEqual(rc, 2)
        self.assertIn("not a running rig", err)
        self.assertEqual(self.pw_log(), [])


class TestSequential(BatchCase):
    def test_jobs_run_one_at_a_time_on_exactly_the_gpus_they_need(self):
        os.environ["FAKE_SLEEP"] = "0.3"
        rc, out, err = self.batch(self.one, self.eight, self.one + "-bench" if (wl.WORKLOADS / (self.one + "-bench")).exists() else self.one)
        self.assertEqual(rc, 0, err + out)
        log = self.pw_log()
        self.assertEqual(log[0]["cvd"], "0")
        self.assertEqual(log[1]["cvd"], "0,1,2,3,4,5,6,7")
        for a, b in zip(log, log[1:]):
            self.assertLessEqual(a["end"], b["start"], "two jobs overlapped without --pack")

    def test_results_are_copied_back_and_summarised(self):
        rc, out, _ = self.batch(self.one)
        d = self.results_dir()
        self.assertEqual(json.loads((d / f"{self.one}.json").read_text())["result"], "PASS")
        self.assertIn(f"{self.one}\tPASS\t0\t", (d / "batch.tsv").read_text())
        self.assertTrue((d / f"{self.one}.stdout").exists() and (d / f"{self.one}.exit").exists())

    def test_a_failure_does_not_stop_the_batch_but_fails_the_exit_status(self):
        os.environ["FAKE_FAIL"] = self.one
        rc, out, _ = self.batch(self.one, self.eight)
        self.assertEqual(rc, 1)
        self.assertEqual([r["name"] for r in self.pw_log()], [self.one, self.eight])
        self.assertIn("FAIL", (self.results_dir() / "batch.tsv").read_text())

    def test_options_reach_the_rig_and_nothing_is_pushed(self):
        rc, out, _ = self.batch(self.one, "--mirror", "s3://b/pw-weights", "--strict-mirror", "--bench", "--repeat", "2",
                                "--env", "EXTRA_KNOB=7", "--timeout", "77")
        self.assertEqual(rc, 0, out)
        (row,) = self.pw_log()
        self.assertEqual((row["mirror"], row["strict"], row["timeout"], row["knob"]), ("s3://b/pw-weights", "1", "77", "1:2:7"))
        for c in self.ssh_cmds():
            self.assertNotRegex(c, r"\bgit\b")
            self.assertNotIn("gh ", c)

    def test_timeout_kills_a_hung_workload_and_the_batch_continues(self):
        os.environ["FAKE_HANG"] = self.one
        t0 = time.time()
        rc, out, _ = self.batch(self.one, self.eight, "--timeout", "1")
        self.assertLess(time.time() - t0, 30)
        self.assertEqual(rc, 1)
        rows = {l.split("\t")[0]: l.split("\t")[1] for l in (self.results_dir() / "batch.tsv").read_text().splitlines()[1:]}
        self.assertEqual(rows[self.one], "TIMEOUT")
        self.assertEqual(rows[self.eight], "PASS")


class TestBusyAndDeadline(BatchCase):
    def hold(self, gpu):
        fh = open(self.locks / f"pw-gpu-{gpu}.lock", "w")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(fh.close)

    def test_a_gpu_held_by_another_job_is_never_shared(self):
        self.hold(0)
        rc, out, _ = self.batch(self.one)
        self.assertEqual(rc, 1)
        self.assertIn("BUSY", out)
        self.assertEqual(self.pw_log(), [])                     # bin/pw was never started

    def test_a_job_needing_all_gpus_is_refused_if_any_one_is_held(self):
        self.hold(5)
        rc, out, _ = self.batch(self.eight)
        self.assertEqual(rc, 1)
        self.assertIn("BUSY", out)
        self.assertEqual(self.pw_log(), [])

    def test_locks_are_released_when_a_job_ends(self):
        self.assertEqual(self.batch(self.one)[0], 0)
        self.assertEqual(self.batch(self.one)[0], 0)             # would be BUSY if the first had kept GPU 0

    def test_nothing_starts_that_cannot_finish_before_the_rig_dies(self):
        self.make_rig("pw-b1", deadline=NOW + 400)
        rc, out, err = self.batch(self.one, self.eight, "--timeout", "600")
        self.assertEqual(rc, 1)
        self.assertEqual(self.pw_log(), [])
        self.assertIn("too little", err)
        rows = (self.results_dir() / "batch.tsv").read_text()
        self.assertIn(f"{self.one}\tNOT-RUN", rows)
        self.assertIn(f"{self.eight}\tNOT-RUN", rows)


class TestPack(BatchCase):
    def test_pack_runs_concurrently_on_disjoint_gpu_sets(self):
        os.environ["FAKE_SLEEP"] = "1.0"
        ones = TIER_A[:3]
        rc, out, err = self.batch(*ones, "--pack")
        self.assertEqual(rc, 0, err + out)
        log = self.pw_log()
        self.assertEqual(len(log), 3)
        sets = [set(r["cvd"].split(",")) for r in log]
        overlapping = [(a, b) for i, a in enumerate(log) for b in log[i + 1:] if a["start"] < b["end"] and b["start"] < a["end"]]
        self.assertTrue(overlapping, "--pack should have overlapped these jobs")
        for a, b in overlapping:
            self.assertFalse(set(a["cvd"].split(",")) & set(b["cvd"].split(",")), (a, b))
        self.assertEqual(len({r["cvd"] for r in log}), 3)

    def test_without_pack_the_same_jobs_never_overlap(self):
        os.environ["FAKE_SLEEP"] = "0.3"
        ones = TIER_A[:3]
        self.assertEqual(self.batch(*ones)[0], 0)
        log = self.pw_log()
        for a, b in zip(log, log[1:]):
            self.assertLessEqual(a["end"], b["start"])

    def test_scheduler_never_hands_one_gpu_to_two_running_jobs(self):
        in_use, violations, peak = set(), [], [0]
        lock = threading.Lock()
        running = [0]

        def run(job, gset):
            with lock:
                if in_use & set(gset):
                    violations.append((job["name"], gset))
                in_use.update(gset)
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                in_use.difference_update(gset)
                running[0] -= 1
            return {"workload": job["name"], "result": "PASS", "gpus": ",".join(map(str, gset)), "seconds": 0}

        jobs = [{"name": f"j{i}", "need": n, "eligible": list(range(8)), "timeout": 1} for i, n in enumerate([4, 4, 2, 2, 1, 1, 8, 3, 3, 2])]
        rows, not_run = batch.schedule(jobs, 8, run, pack=True)
        self.assertEqual((violations, not_run, len(rows)), ([], [], 10))
        self.assertGreater(peak[0], 1)
        # an 8-GPU job gets the whole node to itself
        self.assertIn("j6", [r["workload"] for r in rows])

    def test_scheduler_without_pack_is_serial_and_in_order(self):
        order = []
        def run(job, gset):
            order.append(job["name"])
            return {"workload": job["name"], "result": "PASS", "gpus": "", "seconds": 0}
        jobs = [{"name": f"j{i}", "need": 1, "eligible": [0, 1], "timeout": 1} for i in range(4)]
        rows, _ = batch.schedule(jobs, 2, run, pack=False)
        self.assertEqual(order, ["j0", "j1", "j2", "j3"])

    def test_jobs_respect_eligible_gpus(self):
        pool = batch.Pool(4)
        job = {"name": "j", "need": 2, "eligible": [2, 3]}
        self.assertEqual(pool.take(job), [2, 3])
        self.assertIsNone(pool.take(job))
        pool.give([2, 3])
        self.assertEqual(pool.take(job), [2, 3])


class TestScript(BatchCase):
    def test_script_locks_every_gpu_of_the_set_then_runs_under_timeout(self):
        a = mock.Mock(timeout=None, remote_dir="~/pantheonworkloads", lock_dir="/tmp", all_gpus=False, mirror=None, env=[], bench=False,
                      repeat=3, strict_mirror=False)
        s = batch.job_script({"name": "w", "timeout": 100.0}, [2, 3], a, "pw-batch/x")
        self.assertIn('cd "$HOME/pantheonworkloads"', s)
        self.assertEqual(s.count("flock -n"), 2)
        self.assertIn("pw-gpu-2.lock", s)
        self.assertIn("CUDA_VISIBLE_DEVICES=2,3", s)
        self.assertIn("timeout -k", s)
        self.assertIn("bin/pw run w --target gpu", s)
        self.assertNotIn("git", s)


if __name__ == "__main__":
    unittest.main()
