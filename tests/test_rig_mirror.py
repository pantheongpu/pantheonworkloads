"""Tests for the model mirror: tools/rig/stage_weights.py (producer), tools/rig/mirror.py (consumer), and the three
consumer paths (lc_fetch_shards, pinned.snapshot, the CLI). A fake Hub (fake curl), a fake `aws s3` (a directory) and
tiny files stand in for the real thing; nothing touches the network.

    python3 -m unittest discover -s tests -p 'test_rig*.py' -v
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKES = ROOT / "tests" / "rig_fakes"
sys.path.insert(0, str(ROOT / "tools" / "rig"))
import mirror  # noqa: E402
import stage_weights  # noqa: E402
import wl  # noqa: E402

REPO, REV = "org/model-GGUF", "a" * 40
FILES = {                       # path -> content
    "Q4/model-00001-of-00002.gguf": b"shard one " * 100,
    "Q4/model-00002-of-00002.gguf": b"shard two " * 120,
    "config.json": b'{"architectures": ["X"]}',
    "tokenizer.json": b'{"model": {}}',
    "README.md": b"# not staged",
    "notes/huge.json": b"x" * 100,
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


class MirrorCase(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="mirtest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.hub = self.tmp / "hub"
        for p, b in FILES.items():
            f = self.hub / REPO / REV / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b)
        self.pins = {p: sha(FILES[p]) for p in ("Q4/model-00001-of-00002.gguf", "Q4/model-00002-of-00002.gguf")}
        # a workloads tree with one workload and its pins
        self.wroot = self.tmp / "workloads"
        w = self.wroot / "llamacpp-toy"
        w.mkdir(parents=True)
        (w / "manifest.yaml").write_text("name: llamacpp-toy\nkind: functional\nruntime: llama.cpp\ntargets: [gpu]\nrequires:\n  gpus: 8\n  gpu_memory_gb: 80\n")
        self.sums = w / "model.sha256"
        self.sums.write_text(f"# {REPO} @ {REV} (test)\n" + "".join(f"{s}  {p}\n" for p, s in self.pins.items()))
        env = {"PW_AWS": str(FAKES / "aws"), "PW_CURL": str(FAKES / "curl"), "FAKE_AWS_STATE": str(self.tmp / "aws.json"),
               "FAKE_S3_ROOT": str(self.tmp / "s3"), "FAKE_HF_ROOT": str(self.hub), "FAKE_CURL_LOG": str(self.tmp / "curl.log"),
               "PW_PGREP": str(FAKES / "pgrep"), "FAKE_PGREP_FILE": str(self.tmp / "pgrep"), "FAKE_NICE_LOG": str(self.tmp / "nice.log"),
               "PATH": f"{FAKES}:{os.environ['PATH']}"}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        for k in ("FAKE_CURL_FLAKY_N", "FAKE_S3_FLAKY_N", "FAKE_CURL_FAIL_STATUS", "FAKE_S3_FAIL_MSG", "FAKE_S3_TRUNC", "FAKE_S3_SLEEP", "FAKE_CURL_SLEEP", "FAKE_S3_FAIL", "FAKE_S3_CORRUPT", "FAKE_CURL_FAIL", "FAKE_CURL_CORRUPT", "PW_MODEL_MIRROR", "PW_MODEL_MIRROR_STRICT",
                  "PW_HF_REVISION", "HF_TOKEN"):
            os.environ.pop(k, None)
        pw = mock.patch.object(wl, "WORKLOADS", self.wroot)
        pw.start()
        self.addCleanup(pw.stop)
        listing = {p: {"size": len(b), "lfs": sha(b) if p.endswith(".gguf") else None} for p, b in FILES.items()}
        listing["notes/huge.json"]["size"] = 200 << 20      # matches a small pattern but is not small
        ph = mock.patch.object(stage_weights, "hub_files", lambda repo, rev: listing if (repo, rev) == (REPO, REV) else None)
        ph.start()
        self.addCleanup(ph.stop)
        self.mir = self.tmp / "mirror"

    def stage(self, *extra, mirror_arg=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = stage_weights.main(["llamacpp-toy", "--mirror", mirror_arg or str(self.mir), "--hf-base", "http://fake",
                                     "--scratch", str(self.tmp / "scratch"), "--state-dir", str(self.tmp / "state"), "--poll-s", "0.05",
                                     "--min-free-gb", "0", "--max-load", "1e9",
                                     "--retries", "2", "--retry-base-s", "0.01", "--retry-max-s", "0.02", *extra])
        return rc, out.getvalue(), err.getvalue()

    def curl_urls(self):
        f = self.tmp / "curl.log"
        return [json.loads(l)["url"] for l in f.read_text().splitlines()] if f.exists() else []

    def manifest(self):
        return json.loads((self.mir / REPO / REV / "manifest.json").read_text())


class TestPlan(MirrorCase):
    def test_plan_lists_pins_and_small_files_only(self):
        files = stage_weights.plan_model(REPO, REV, self.pins)
        self.assertEqual([f["path"] for f in files if f["pinned"]], sorted(self.pins))
        small = sorted(f["path"] for f in files if not f["pinned"])
        self.assertEqual(small, ["config.json", "tokenizer.json"])      # no README, no 200 MiB json

    def test_pin_that_disagrees_with_the_hub_oid_is_refused(self):
        bad = dict(self.pins)
        bad["Q4/model-00001-of-00002.gguf"] = "0" * 64
        with self.assertRaises(ValueError) as cm:
            stage_weights.plan_model(REPO, REV, bad)
        self.assertIn("LFS oid", str(cm.exception))

    def test_pin_header_is_parsed(self):
        self.assertEqual(wl.read_pin("llamacpp-toy")[:2], (REPO, REV))

    def test_twins_share_one_model(self):
        twin = self.wroot / "llamacpp-bench-toy"
        shutil.copytree(self.wroot / "llamacpp-toy", twin)
        g, _ = stage_weights.groups(["llamacpp-toy", "llamacpp-bench-toy"])
        self.assertEqual(list(g), [(REPO, REV)])
        self.assertEqual(sorted(g[(REPO, REV)]["workloads"]), ["llamacpp-bench-toy", "llamacpp-toy"])


class TestStage(MirrorCase):
    def test_stages_verifies_and_writes_the_manifest(self):
        rc, out, err = self.stage()
        self.assertEqual(rc, 0, err)
        m = self.manifest()
        self.assertTrue(m["complete"])
        self.assertEqual(sorted(m["files"]), ["Q4/model-00001-of-00002.gguf", "Q4/model-00002-of-00002.gguf", "config.json", "tokenizer.json"])
        for p, meta in m["files"].items():
            self.assertEqual(meta["sha256"], sha(FILES[p]))
            self.assertEqual((self.mir / REPO / REV / p).read_bytes(), FILES[p])
        self.assertTrue(m["files"]["Q4/model-00001-of-00002.gguf"]["pinned"])
        self.assertIn(f"PW_MODEL_MIRROR={self.mir}", out)
        self.assertEqual(len(self.curl_urls()), 4)

    def test_a_second_run_downloads_nothing(self):
        self.stage()
        n = len(self.curl_urls())
        rc, out, _ = self.stage()
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.curl_urls()), n)
        self.assertIn("staged", out)

    def test_an_interrupted_run_resumes_with_only_the_missing_files(self):
        os.environ["FAKE_CURL_FAIL"] = "model-00002"
        rc, _, err = self.stage()
        self.assertEqual(rc, 1)
        self.assertIn("model-00002", err)
        m = self.manifest()
        self.assertFalse(m["complete"])
        self.assertIn("Q4/model-00001-of-00002.gguf", m["files"])
        self.assertNotIn("Q4/model-00002-of-00002.gguf", m["files"])
        before = self.curl_urls()
        os.environ.pop("FAKE_CURL_FAIL")
        rc, _, err = self.stage()
        self.assertEqual(rc, 0, err)
        after = self.curl_urls()[len(before):]
        self.assertTrue(all("model-00001" not in u for u in after), after)
        self.assertTrue(self.manifest()["complete"])

    def test_a_corrupt_download_is_never_uploaded(self):
        os.environ["FAKE_CURL_CORRUPT"] = "model-00001"
        rc, _, err = self.stage()
        self.assertEqual(rc, 1)
        self.assertIn("sha256 of the download", err)
        self.assertFalse((self.mir / REPO / REV / "Q4/model-00001-of-00002.gguf").exists())
        self.assertNotIn("Q4/model-00001-of-00002.gguf", self.manifest()["files"])
        self.assertFalse(self.manifest()["complete"])

    def test_dry_run_and_check_touch_nothing(self):
        rc, out, _ = self.stage("--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("files to stage", out)
        self.assertFalse(self.mir.exists())
        self.assertEqual(self.curl_urls(), [])
        self.assertEqual(self.stage("--check")[0], 1)          # nothing staged yet
        self.stage()
        self.assertEqual(self.stage("--check")[0], 0)

    def test_token_is_sent_when_set(self):
        os.environ["HF_TOKEN"] = "hf_secret"
        self.stage()
        headers = [json.loads(l)["headers"] for l in (self.tmp / "curl.log").read_text().splitlines()]
        self.assertTrue(all(h == ["Authorization: Bearer hf_secret"] for h in headers))

    def test_s3_mirror_goes_through_aws_s3_cp(self):
        rc, out, err = self.stage(mirror_arg="s3://bkt/pw-weights")
        self.assertEqual(rc, 0, err)
        root = self.tmp / "s3" / "bkt" / "pw-weights" / REPO / REV
        self.assertTrue((root / "manifest.json").is_file())
        self.assertEqual((root / "config.json").read_bytes(), FILES["config.json"])
        calls = [json.loads(l) for l in pathlib.Path(str(self.tmp / "aws.json") + ".calls").read_text().splitlines()]
        self.assertTrue(calls and all(c[:2] in (["s3", "cp"], ["s3api", "head-object"]) for c in calls))
        self.assertIn("export PW_MODEL_MIRROR=s3://bkt/pw-weights", out)


def read_log(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text().splitlines()] if pathlib.Path(path).exists() else []


class TestPolite(MirrorCase):
    """The shared-box mode: nice/ionice, rate caps, the pause gates, the disk budget, delete-after-verify, status."""

    def setUp(self):
        super().setUp()
        self.load = [0.5]                 # what the gates see; tests change these instead of the real machine's
        self.free = [10 << 40]
        for target, fn in (("os.getloadavg", lambda: (self.load[0], 1, 1)),
                           ("shutil.disk_usage", lambda p: types.SimpleNamespace(total=20 << 40, used=0, free=self.free[0]))):
            pp = mock.patch(target, fn)
            pp.start()
            self.addCleanup(pp.stop)

    def uploads(self):
        """nice.log entries of the upload processes (the ones started with the generated AWS config)."""
        return [c for c in read_log(self.tmp / "nice.log") if c["config"]]

    def s3(self, *extra):
        return self.stage(*extra, mirror_arg="s3://bkt/pw-weights")

    def s3_manifest(self):
        return json.loads((self.tmp / "s3/bkt/pw-weights" / REPO / REV / "manifest.json").read_text())

    def run_in_thread(self, *extra):
        res = {}
        t = threading.Thread(target=lambda: res.update(r=self.s3(*extra)))
        t.start()
        return t, res

    def test_curl_and_aws_run_niced_and_rate_limited_by_default(self):
        rc, _, err = self.s3()
        self.assertEqual(rc, 0, err)
        curls = read_log(self.tmp / "curl.log")
        self.assertTrue(curls and all(c["nice"] == 19 for c in curls))
        calls = self.uploads()
        self.assertTrue(calls and all(c["nice"] == 19 for c in calls))
        # default 20 MiB/s over 4 downloads: 5 MiB/s each
        self.assertTrue(all(c["limit_rate"] == "5120k" for c in curls))
        cfg = pathlib.Path(calls[0]["config"]).read_text()
        self.assertIn("max_bandwidth = 10240KB/s", cfg)          # 20 MiB/s over --aws-max-concurrent 2
        self.assertIn("max_concurrent_requests", cfg)

    def test_no_nice_and_no_rate_limit(self):
        rc, _, err = self.s3("--no-nice", "--rate-limit", "0")
        self.assertEqual(rc, 0, err)
        me = os.nice(0)
        self.assertTrue(all(c["nice"] == me and c["limit_rate"] is None for c in read_log(self.tmp / "curl.log")))
        self.assertNotIn("max_bandwidth", pathlib.Path(self.uploads()[0]["config"]).read_text())

    def test_local_copy_is_deleted_only_after_the_upload_is_verified(self):
        rc, _, err = self.s3()
        self.assertEqual(rc, 0, err)
        left = [p for p in (self.tmp / "scratch").rglob("*") if p.is_file()]
        self.assertEqual([p.name for p in left], ["manifest.json"], left)       # only the tiny manifest staging file
        heads = [c for c in read_log(self.tmp / "aws.json.calls") if c[:2] == ["s3api", "head-object"]]
        self.assertEqual(len(heads), 4)

    def test_a_short_object_in_the_mirror_keeps_the_local_copy_and_is_not_recorded(self):
        os.environ["FAKE_S3_TRUNC"] = "model-00001"
        rc, _, err = self.s3()
        self.assertEqual(rc, 1)
        self.assertIn("mirror holds", err)
        m = self.s3_manifest()
        self.assertNotIn("Q4/model-00001-of-00002.gguf", m["files"])
        self.assertFalse(m["complete"])
        self.assertTrue(list((self.tmp / "scratch").rglob("*model-00001*.part")))
        os.environ.pop("FAKE_S3_TRUNC")
        rc, _, err = self.s3()                       # the retry uploads the kept copy again
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.s3_manifest()["complete"])
        self.assertEqual([p.name for p in (self.tmp / "scratch").rglob("*") if p.is_file()], ["manifest.json"])

    def test_ci_runner_worker_pauses_everything_and_it_resumes(self):
        (self.tmp / "pgrep").write_text("1")
        t, res = self.run_in_thread("--strict-ci-pause")
        time.sleep(0.5)
        self.assertEqual(self.curl_urls(), [])                  # nothing started while Runner.Worker exists
        (self.tmp / "pgrep").write_text("0")
        t.join(30)
        rc, out, err = res["r"]
        self.assertEqual(rc, 0, err)
        self.assertRegex(out, r"PAUSE: 1 Runner.Worker process\(es\) running")
        self.assertRegex(out, r"RESUME after \d+s \(was: 1 Runner.Worker")
        self.assertLess(out.index("PAUSE"), out.index("RESUME"))
        self.assertEqual(len(self.curl_urls()), 4)

    def test_a_worker_appearing_mid_download_stops_curl_and_resumes_it_later(self):
        os.environ["FAKE_CURL_SLEEP"] = "0.6"
        t, res = self.run_in_thread("--threads", "1", "--strict-ci-pause")
        time.sleep(0.3)
        (self.tmp / "pgrep").write_text("2")
        time.sleep(0.5)
        (self.tmp / "pgrep").write_text("0")
        t.join(60)
        rc, out, err = res["r"]
        self.assertEqual(rc, 0, err)
        self.assertIn("PAUSE: 2 Runner.Worker", out)
        self.assertIn("RESUME", out)
        self.assertGreater(len(self.curl_urls()), 4)            # the interrupted file was fetched again (curl -C - resumes it)
        self.assertTrue(self.s3_manifest()["complete"])

    def test_dry_run_is_not_throttled_or_paused(self):
        (self.tmp / "pgrep").write_text("3")
        self.assertEqual(self.stage("--dry-run")[0], 0)

    def test_ci_throttles_instead_of_pausing(self):
        (self.tmp / "pgrep").write_text("1")
        rc, out, err = self.s3()
        self.assertEqual(rc, 0, err)
        self.assertNotIn("PAUSE", out)
        self.assertIn("THROTTLE busy (initial): 1 Runner.Worker process(es); limit now 8 MiB/s", out)
        curls = read_log(self.tmp / "curl.log")
        self.assertEqual(len(curls), 4)                          # it transferred, with CI running
        self.assertTrue(all(c["limit_rate"] == "2048k" for c in curls))     # 8 MiB/s over 4 downloads
        self.assertIn("max_bandwidth = 4096KB/s", pathlib.Path(self.uploads()[0]["config"]).read_text())   # 8 MiB/s over 2 uploads
        self.assertTrue(self.s3_manifest()["complete"])

    def test_busy_rate_zero_does_not_throttle(self):
        (self.tmp / "pgrep").write_text("1")
        self.assertEqual(self.s3("--busy-rate", "0")[0], 0)
        self.assertTrue(all(c["limit_rate"] == "5120k" for c in read_log(self.tmp / "curl.log")))

    def test_throttle_follows_ci_live_and_restarts_the_running_curl(self):
        os.environ["FAKE_CURL_SLEEP"] = "0.8"
        t, res = self.run_in_thread("--threads", "1")
        time.sleep(0.3)
        (self.tmp / "pgrep").write_text("1")                    # CI starts while the first file is downloading
        time.sleep(0.5)
        (self.tmp / "pgrep").write_text("0")                    # and ends again
        t.join(60)
        rc, out, err = res["r"]
        self.assertEqual(rc, 0, err)
        self.assertRegex(out, r"THROTTLE busy: 1 Runner.Worker process\(es\) running; downloads and uploads limited to 8 MiB/s \(was 20 MiB/s\)")
        self.assertRegex(out, r"THROTTLE free: no Runner.Worker; downloads and uploads back to 20 MiB/s")
        self.assertNotIn("PAUSE", out)
        limits = [c["limit_rate"] for c in read_log(self.tmp / "curl.log")]
        self.assertIn("20480k", limits)                          # threads 1: 20 MiB/s, then 8 MiB/s, then 20 again
        self.assertIn("8192k", limits)
        self.assertTrue(self.s3_manifest()["complete"])

    def test_an_upload_in_flight_is_slowed_by_a_duty_cycle_when_ci_starts(self):
        os.environ["FAKE_S3_SLEEP"] = "1.5"
        t, res = self.run_in_thread("--threads", "1", "--aws-max-concurrent", "1")
        time.sleep(0.5)
        (self.tmp / "pgrep").write_text("1")
        t.join(60)
        rc, out, err = res["r"]
        self.assertEqual(rc, 0, err)
        self.assertIn("in-flight upload slowed to a 40% duty cycle", out)
        self.assertTrue(self.s3_manifest()["complete"])

    def test_status_shows_the_throttle(self):
        (self.tmp / "pgrep").write_text("2")
        self.s3()
        out = io.StringIO()
        st = json.loads((self.tmp / "state" / "status.json").read_text())
        self.assertEqual(st["throttle"]["mode"], "busy")
        st["running"] = True
        st["pid"] = os.getpid()
        (self.tmp / "state" / "status.json").write_text(json.dumps(st))
        with contextlib.redirect_stdout(out):
            stage_weights.main(["--status", "--state-dir", str(self.tmp / "state")])
        self.assertIn("throttle: BUSY, 2 Runner.Worker process(es) running, transfers limited to 8 MiB/s", out.getvalue())

    def test_transient_curl_errors_are_retried_and_the_run_succeeds(self):
        os.environ["FAKE_CURL_FLAKY_N"] = "2"                   # exit 92 (CDN stream cancel) twice per file
        rc, out, err = self.stage("--retries", "5")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(self.curl_urls()), 12)
        self.assertIn("RETRY download", out)
        self.assertTrue(self.manifest()["complete"])

    def test_retries_give_up_and_the_run_fails(self):
        os.environ["FAKE_CURL_FLAKY_N"] = "99"
        rc, out, err = self.stage("--retries", "3")
        self.assertEqual(rc, 1)
        self.assertIn("giving up after 3 tries", err)
        self.assertEqual(len(self.curl_urls()), 12)

    def test_permanent_http_errors_are_not_retried(self):
        for status in ("404", "403", "401"):
            os.environ["FAKE_CURL_FAIL"] = "model-00001"
            os.environ["FAKE_CURL_FAIL_STATUS"] = status
            rc, out, err = self.stage("--retries", "5")
            self.assertEqual(rc, 1)
            self.assertNotIn("RETRY", out)
            self.assertIn(f"error: {status}", err)
        self.assertEqual(sum("model-00001" in u for u in self.curl_urls()), 3)      # one try per run

    def test_a_503_is_transient(self):
        os.environ["FAKE_CURL_FAIL"] = "model-00001"
        rc, out, err = self.stage("--retries", "3")
        self.assertEqual(rc, 1)
        self.assertEqual(sum("model-00001" in u for u in self.curl_urls()), 3)
        self.assertIn("RETRY download Q4/model-00001", out)

    def test_curl_restarts_resume_the_partial_file_and_resolve_the_url_again(self):
        os.environ["FAKE_CURL_FLAKY_N"] = "1"
        self.stage()
        for l in read_log(self.tmp / "curl.log"):
            self.assertTrue(l["url"].startswith("http://fake/org/model-GGUF/resolve/"))   # the resolve URL every time, never a cached CDN URL

    def test_transient_s3_errors_are_retried(self):
        os.environ["FAKE_S3_FLAKY_N"] = "2"
        rc, out, err = self.s3("--retries", "5")
        self.assertEqual(rc, 0, err)
        self.assertIn("RETRY upload", out)
        self.assertTrue(self.s3_manifest()["complete"])

    def test_s3_access_denied_is_permanent(self):
        os.environ["FAKE_S3_FAIL"] = "model-00001"
        os.environ["FAKE_S3_FAIL_MSG"] = "An error occurred (AccessDenied) when calling the PutObject operation"
        rc, out, err = self.s3("--retries", "5")
        self.assertEqual(rc, 1)
        self.assertNotIn("RETRY upload Q4/model-00001", out)
        self.assertIn("AccessDenied", err)

    def test_sha256_mismatch_is_permanent(self):
        os.environ["FAKE_CURL_CORRUPT"] = "model-00001"
        rc, out, err = self.stage("--retries", "5")
        self.assertEqual(rc, 1)
        self.assertNotIn("RETRY", out)

    def test_load_gate(self):
        self.load[0] = 50.0
        t, res = self.run_in_thread("--max-load", "12")
        time.sleep(0.4)
        self.assertEqual(self.curl_urls(), [])
        self.load[0] = 1.0
        t.join(30)
        self.assertEqual(res["r"][0], 0)
        self.assertIn("PAUSE: 1-minute load average 50.0 is above --max-load 12", res["r"][1])
        self.assertIn("RESUME", res["r"][1])

    def test_free_space_gate(self):
        self.free[0] = 100 << 30
        t, res = self.run_in_thread("--min-free-gb", "500")
        time.sleep(0.4)
        self.assertEqual(self.curl_urls(), [])
        self.free[0] = 1 << 40
        t.join(30)
        self.assertEqual(res["r"][0], 0)
        self.assertIn("PAUSE: free space 100 GiB is below --min-free-gb 500", res["r"][1])

    def test_dry_run_and_check_never_pause(self):
        (self.tmp / "pgrep").write_text("3")
        self.assertEqual(self.stage("--dry-run")[0], 0)
        self.assertEqual(self.stage("--check")[0], 1)

    def test_batch_budget_serialises_files_that_do_not_fit_together(self):
        os.environ["FAKE_CURL_SLEEP"] = "0.2"
        # 107 bytes: each shard (1000, 1200 bytes) runs alone; the two tiny json files (24 + 13) fit together
        t0 = time.monotonic()
        rc, _, err = self.stage("--batch-gb", "1e-7", "--threads", "4")
        self.assertEqual(rc, 0, err)
        self.assertGreaterEqual(time.monotonic() - t0, 0.55)    # three rounds of 0.2 s, not one
        self.assertEqual(len(self.curl_urls()), 4)

    def test_smallest_first_orders_models(self):
        big_repo = "org/big-GGUF"
        w = self.wroot / "llamacpp-big"
        shutil.copytree(self.wroot / "llamacpp-toy", w)
        big = b"B" * 5000
        f = self.hub / big_repo / REV / "big.gguf"
        f.parent.mkdir(parents=True)
        f.write_bytes(big)
        (w / "model.sha256").write_text(f"# {big_repo} @ {REV} (test)\n{sha(big)}  big.gguf\n")
        listing = {"big.gguf": {"size": len(big), "lfs": sha(big)}}
        orig = stage_weights.hub_files
        with mock.patch.object(stage_weights, "hub_files", lambda r, v: listing if r == big_repo else orig(r, v)):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = stage_weights.main(["llamacpp-big", "llamacpp-toy", "--mirror", str(self.mir), "--hf-base", "http://fake", "--scratch", str(self.tmp / "sc"),
                                         "--state-dir", str(self.tmp / "st"), "--threads", "1", "--min-free-gb", "0", "--max-load", "1e9"])
        self.assertEqual(rc, 0, err.getvalue())
        started = [l.split()[3] for l in out.getvalue().splitlines() if " START " in l]
        self.assertEqual(started[0], REPO)                      # the toy model (a few KB) before the 5 KB one... listed first by size
        self.assertEqual(started[-1], big_repo)

    def test_status_reports_progress_rate_and_eta(self):
        self.stage()
        st = json.loads((self.tmp / "state" / "status.json").read_text())
        self.assertEqual((st["models_staged"], st["models_total"]), (1, 1))
        self.assertEqual(st["done_bytes"], st["total_bytes"])
        self.assertIsNotNone(st["rate_sustained"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = stage_weights.main(["--status", "--state-dir", str(self.tmp / "state")])
        self.assertEqual(rc, 0)
        for word in ("NOT RUNNING", "models staged: 1/1", "sustained", "ETA: 0h 0m"):
            self.assertIn(word, out.getvalue())

    def test_status_with_no_run(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(stage_weights.main(["--status", "--state-dir", str(self.tmp / "none")]), 1)

    def test_a_second_run_in_the_same_state_dir_is_refused(self):
        import fcntl
        (self.tmp / "state").mkdir()
        with open(self.tmp / "state" / "stage.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            rc, _, err = self.stage()
        self.assertEqual(rc, 4)
        self.assertIn("holds", err)

    def test_log_file_is_written(self):
        log = self.tmp / "logs" / "stage.log"
        self.assertEqual(self.stage("--log-file", str(log))[0], 0)
        self.assertIn("MODEL STAGED", log.read_text())


class TestSupervisor(unittest.TestCase):
    SUP = ROOT / "tools" / "rig" / "stage_supervisor.sh"

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="suptest-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.env = dict(os.environ, PW_STAGE_HOME=str(self.tmp / "home"), PW_STAGE_NO_SYSTEMD="1", PW_STAGE_BACKOFF_S="0", PW_CRONTAB="/nonexistent-crontab",
                        PW_STAGE_PY=str(self.tmp / "fake_stage.py"), HOME=str(self.tmp))

    def sup(self, *args):
        return subprocess.run(["bash", str(self.SUP), *args], capture_output=True, text=True, env=self.env, timeout=60)

    def fake(self, body):
        (self.tmp / "fake_stage.py").write_text("import sys\n" + body)

    def test_restarts_on_failure_and_exits_when_complete(self):
        self.fake("import pathlib\np = pathlib.Path(sys.argv[0]).with_name('n')\nn = int(p.read_text()) if p.exists() else 0\n"
                  "p.write_text(str(n + 1))\nsys.exit(0 if n >= 2 else 1)\n")
        r = self.sup("run", "--tier", "2b")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.tmp / "n").read_text(), "3")
        log = (self.tmp / "home" / "logs" / "supervisor.log").read_text()
        self.assertIn("complete", log)
        self.assertIn("nice/ionice only", log)                 # systemd-run was disabled for the test

    def test_a_usage_error_is_not_retried(self):
        self.fake("sys.exit(2)\n")
        r = self.sup("run", "--tier", "2b")
        self.assertEqual(r.returncode, 2)

    def test_second_instance_is_refused(self):
        self.fake("import time\ntime.sleep(30)\n")
        p = subprocess.Popen(["bash", str(self.SUP), "run"], env=self.env, start_new_session=True)
        try:
            for _ in range(100):
                if (self.tmp / "home" / "supervisor.lock").exists():
                    break
                time.sleep(0.05)
            time.sleep(0.5)
            r = self.sup("run")
            self.assertEqual(r.returncode, 9)
            self.assertIn("already running", r.stderr + r.stdout)
        finally:
            os.killpg(p.pid, 15)
            p.wait()

    def test_cron_install_and_remove(self):
        bindir = self.tmp / "bin"
        bindir.mkdir()
        store = self.tmp / "crontab.txt"
        (bindir / "crontab").write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat {store} 2>/dev/null || exit 1; else cat > {store}; fi\n')
        (bindir / "crontab").chmod(0o755)
        self.env["PW_CRONTAB"] = str(bindir / "crontab")      # never the real crontab
        self.assertEqual(self.sup("install-cron", "--tier", "2b", "--mirror", "s3://b/pw-weights").returncode, 0)
        text = store.read_text()
        self.assertEqual(text.count("@reboot"), 1)
        self.assertIn("--mirror s3://b/pw-weights", text)
        self.sup("install-cron", "--tier", "2b")                 # idempotent: replaces, never duplicates
        self.assertEqual(store.read_text().count("@reboot"), 1)
        store.write_text("0 3 * * * other-job\n" + store.read_text())
        self.assertEqual(self.sup("remove-cron").returncode, 0)
        self.assertEqual(store.read_text().strip(), "0 3 * * * other-job")


class TestFetch(MirrorCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.stage()[0], 0)
        self.dest = self.tmp / "dest"

    def fetch(self, mirror_arg=None, **kw):
        return mirror.fetch(mirror_arg or str(self.mir), REPO, REV, self.dest, **kw)

    def test_copies_and_verifies_from_a_directory(self):
        self.fetch(names=list(self.pins), patterns=["*.json"], expected=self.pins)
        for p in ("Q4/model-00001-of-00002.gguf", "config.json", "tokenizer.json"):
            self.assertEqual((self.dest / p).read_bytes(), FILES[p])
        self.assertFalse(any(self.dest.rglob("*.part")))

    def test_copies_from_s3_in_parallel_and_skips_what_is_already_verified(self):
        self.assertEqual(self.stage(mirror_arg="s3://bkt/pw-weights")[0], 0)
        calls_file = pathlib.Path(str(self.tmp / "aws.json") + ".calls")
        self.fetch("s3://bkt/pw-weights", names=list(self.pins), expected=self.pins, jobs=4)
        n = len(calls_file.read_text().splitlines())
        self.fetch("s3://bkt/pw-weights", names=list(self.pins), expected=self.pins, jobs=4)
        added = [json.loads(l) for l in calls_file.read_text().splitlines()[n:]]
        self.assertEqual(len(added), 1)                          # only manifest.json is read again
        self.assertIn("manifest.json", " ".join(added[0]))

    def test_a_file_changed_in_the_destination_is_copied_again(self):
        self.fetch(names=list(self.pins))
        victim = self.dest / "Q4/model-00001-of-00002.gguf"
        victim.write_bytes(b"truncated")
        self.fetch(names=list(self.pins), expected=self.pins)
        self.assertEqual(victim.read_bytes(), FILES["Q4/model-00001-of-00002.gguf"])

    def test_a_file_corrupt_in_the_mirror_fails_and_is_not_left_behind(self):
        os.environ["FAKE_S3_CORRUPT"] = "model-00002"
        self.assertEqual(self.stage(mirror_arg="s3://bkt/pw-weights")[0], 0)   # uploads, then the fake rots it in the bucket
        with self.assertRaises(mirror.Mismatch) as cm:
            self.fetch("s3://bkt/pw-weights", names=list(self.pins), expected=self.pins)
        self.assertIn("model-00002", str(cm.exception))
        self.assertFalse((self.dest / "Q4/model-00002-of-00002.gguf").exists())
        self.assertFalse((self.dest / "Q4/model-00002-of-00002.gguf.part").exists())

    def test_a_manifest_that_disagrees_with_the_pin_is_a_mismatch(self):
        bad = dict(self.pins)
        bad["config.json"] = "f" * 64
        with self.assertRaises(mirror.Mismatch):
            self.fetch(names=["config.json"], expected=bad)

    def test_missing_model_or_file_is_a_miss_not_a_mismatch(self):
        with self.assertRaises(mirror.Miss):
            mirror.fetch(str(self.mir), "org/other", REV, self.dest)
        with self.assertRaises(mirror.Miss):
            self.fetch(names=["not-there.safetensors"])

    def test_flat_layout_puts_shards_beside_each_other(self):
        self.fetch(names=list(self.pins), flat=True)
        self.assertTrue((self.dest / "model-00001-of-00002.gguf").is_file())
        self.assertFalse((self.dest / "Q4").exists())

    def test_cli_exit_codes(self):
        base = ["fetch", "--mirror", str(self.mir), "--repo", REPO, "--rev", REV, "--dest", str(self.dest), "--sums", str(self.sums)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(mirror.main(base), 0)
            self.assertEqual(mirror.main(base[:4] + ["org/other"] + base[5:]), 3)
            (self.mir / REPO / REV / "Q4/model-00001-of-00002.gguf").write_bytes(b"rot")
            shutil.rmtree(self.dest)
            self.assertEqual(mirror.main(base), 1)


class TestLcFetchShards(MirrorCase):
    """The bash consumer: tools/llamacpp/common.sh lc_fetch_shards with PW_MODEL_MIRROR."""

    def setUp(self):
        super().setUp()
        self.assertEqual(self.stage()[0], 0)
        self.cache = self.tmp / "cache"

    def run_fetch(self, mirror_value, strict=False, rev=REV):
        env = dict(os.environ, PW_CACHE=str(self.cache), PW_MODEL_MIRROR=mirror_value, PW_TARGET="gpu", PW_OUT=str(self.tmp / "out"))
        if strict:
            env["PW_MODEL_MIRROR_STRICT"] = "1"
        env.pop("PW_MODEL_FILE", None)
        script = (f'. "{ROOT}/tools/llamacpp/common.sh"; '
                  f'lc_fetch_shards "https://huggingface.co/{REPO}/resolve/{rev}" "{self.sums}" 0; echo "MODEL=$LC_MODEL"')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)

    def test_shards_come_from_the_mirror_and_the_first_one_is_the_model(self):
        p = self.run_fetch(str(self.mir))
        self.assertEqual(p.returncode, 0, p.stderr)
        d = self.cache / "models" / "mirror" / f"org__model-GGUF-{REV}"      # keyed by repository and commit, so twins share it
        self.assertIn(f"MODEL={d}/model-00001-of-00002.gguf", p.stdout)
        self.assertEqual((d / "model-00002-of-00002.gguf").read_bytes(), FILES["Q4/model-00002-of-00002.gguf"])
        self.assertEqual(len(self.curl_urls()), 4)                # only staging used the Hub

    def test_a_shard_that_is_not_the_pinned_one_fails_the_run(self):
        (self.mir / REPO / REV / "Q4/model-00002-of-00002.gguf").write_bytes(b"rotten")
        p = self.run_fetch(str(self.mir))
        self.assertEqual(p.returncode, 1)
        self.assertIn("model-00002", p.stderr)
        self.assertFalse(any(self.cache.rglob("model-00002-of-00002.gguf")))

    def test_a_mirror_that_lies_in_its_manifest_is_caught_against_model_sha256(self):
        mf = self.mir / REPO / REV / "manifest.json"
        m = json.loads(mf.read_text())
        wrong = b"different shard"
        m["files"]["Q4/model-00002-of-00002.gguf"].update(sha256=sha(wrong), size=len(wrong))
        (self.mir / REPO / REV / "Q4/model-00002-of-00002.gguf").write_bytes(wrong)
        mf.write_text(json.dumps(m))
        p = self.run_fetch(str(self.mir))
        self.assertEqual(p.returncode, 1)
        self.assertIn("pin is", p.stderr)

    def test_unmirrored_model_falls_back_to_the_hub_unless_strict(self):
        other = self.tmp / "emptymirror"
        other.mkdir()
        os.environ["FAKE_HF_ROOT"] = str(self.hub)
        # the Hub fallback is lc_fetch_shards' own curl loop, served by the fake curl on PATH
        p = self.run_fetch(str(other))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("not in the model mirror; downloading from the Hub", p.stderr)
        self.assertTrue((self.cache / "models" / "llamacpp-toy" / "model-00001-of-00002.gguf").is_file())
        shutil.rmtree(self.cache)
        s = self.run_fetch(str(other), strict=True)
        self.assertEqual(s.returncode, 77)
        self.assertIn("PW_MODEL_MIRROR_STRICT=1", s.stdout)

    def test_without_a_mirror_the_old_path_is_unchanged(self):
        env = dict(os.environ, PW_CACHE=str(self.cache), PW_TARGET="gpu", PW_OUT=str(self.tmp / "out"))
        env.pop("PW_MODEL_MIRROR", None)
        script = (f'. "{ROOT}/tools/llamacpp/common.sh"; '
                  f'lc_fetch_shards "http://fake/{REPO}/resolve/{REV}" "{self.sums}" 0; echo "MODEL=$LC_MODEL"')
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("model-00001-of-00002.gguf", p.stdout)


class TestPinnedSnapshot(MirrorCase):
    """The Python consumer: workloads/_pytorch/pinned.py snapshot()."""

    def setUp(self):
        super().setUp()
        self.assertEqual(self.stage()[0], 0)
        spec = importlib.util.spec_from_file_location("pinned_under_test", ROOT / "workloads" / "_pytorch" / "pinned.py")
        self.pinned = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pinned)
        os.environ["PW_WORKLOAD_DIR"] = str(self.wroot / "llamacpp-toy")
        self.cache = self.tmp / "hfcache"
        self.hf_calls = []
        fake = types.ModuleType("huggingface_hub")
        fake.snapshot_download = lambda *a, **k: self.hf_calls.append((a, k)) or "/hub/snapshot"
        p = mock.patch.dict(sys.modules, {"huggingface_hub": fake})
        p.start()
        self.addCleanup(p.stop)

    def test_without_a_mirror_the_hub_is_used(self):
        got = self.pinned.snapshot(REPO, REV, str(self.cache), ["*.json"])
        self.assertEqual(got, "/hub/snapshot")
        self.assertEqual(self.hf_calls[0][1]["allow_patterns"], ["*.json"])

    def test_with_a_mirror_the_files_are_copied_and_the_hub_is_not_called(self):
        os.environ["PW_MODEL_MIRROR"] = str(self.mir)
        got = self.pinned.snapshot(REPO, REV, str(self.cache), ["*.json"])
        self.assertEqual(self.hf_calls, [])
        for p in list(self.pins) + ["config.json"]:
            self.assertEqual((pathlib.Path(got) / p).read_bytes(), FILES[p])
        self.pinned.verify_snapshot(got, str(self.sums))         # the existing post-check passes on the copy

    def test_sha256_mismatch_is_a_failure_not_a_skip(self):
        os.environ["PW_MODEL_MIRROR"] = str(self.mir)
        (self.mir / REPO / REV / "Q4/model-00001-of-00002.gguf").write_bytes(b"rotten")
        with self.assertRaises(SystemExit) as cm:
            self.pinned.snapshot(REPO, REV, str(self.cache), ["*.json"])
        self.assertIn("model-00001", str(cm.exception))
        self.assertEqual(self.hf_calls, [])                      # and never "fixed" by quietly going to the Hub

    def test_missing_model_falls_back_or_raises_when_strict(self):
        os.environ["PW_MODEL_MIRROR"] = str(self.tmp / "empty")
        self.assertEqual(self.pinned.snapshot(REPO, REV, str(self.cache), ["*.json"]), "/hub/snapshot")
        os.environ["PW_MODEL_MIRROR_STRICT"] = "1"
        with self.assertRaises(RuntimeError):
            self.pinned.snapshot(REPO, REV, str(self.cache), ["*.json"])

    def test_an_unpinned_revision_override_skips_the_mirror(self):
        os.environ["PW_MODEL_MIRROR"] = str(self.mir)
        os.environ["PW_HF_REVISION"] = "main"
        self.assertEqual(self.pinned.snapshot(REPO, "main", str(self.cache), ["*.json"]), "/hub/snapshot")


class TestTiers(unittest.TestCase):
    """Against the real manifests of this repository."""

    def test_tiers_follow_requires(self):
        a, b, c = wl.in_tier("A"), wl.in_tier("B"), wl.in_tier("2b")
        self.assertTrue(a and b and c)
        self.assertFalse(set(a) & set(b) or set(a) & set(c) or set(b) & set(c))
        for n in a:
            r = wl.requires(n)
            self.assertEqual((r["gpus"], r["gpu_memory_gb"] <= 22), (1, True))
        for n in c:
            r = wl.requires(n)
            self.assertTrue(r["gpus"] <= 8 and r.get("gpu_memory_gb", 0) <= 80, n)
            self.assertFalse(wl.graduated(wl.manifest(n)), n)

    def test_every_stage_2b_workload_has_a_stageable_pin(self):
        g, unstageable = stage_weights.groups(wl.in_tier("2b"))
        self.assertEqual(unstageable, [])
        for (repo, rev), e in g.items():
            self.assertEqual(len(rev), 40, repo)
            stage_weights.plan_model(repo, rev, e["pins"])       # raises when the catalog and the pins disagree


if __name__ == "__main__":
    unittest.main()
