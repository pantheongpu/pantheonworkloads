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
               "PATH": f"{FAKES}:{os.environ['PATH']}"}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        for k in ("FAKE_S3_FAIL", "FAKE_S3_CORRUPT", "FAKE_CURL_FAIL", "FAKE_CURL_CORRUPT", "PW_MODEL_MIRROR", "PW_MODEL_MIRROR_STRICT",
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
                                     "--scratch", str(self.tmp / "scratch"), *extra])
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
        self.assertTrue(calls and all(c[:2] == ["s3", "cp"] for c in calls))
        self.assertIn("export PW_MODEL_MIRROR=s3://bkt/pw-weights", out)


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
