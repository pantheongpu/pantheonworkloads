"""Tests for tools/rig/fastfetch.py (parallel ranged Hub downloader), its wiring behind PW_FAST_FETCH=1, and tools/rig/nvme-setup.sh.
A local http.server with Range support, a redirecting /resolve/ endpoint that hands out "signed" CDN URLs, and tiny files stand in
for the Hub; nothing touches the network.

    python3 -m unittest tests.test_rig_fastfetch -v
"""
import hashlib
import http.server
import json
import os
import pathlib
import re
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "rig"))
import fastfetch  # noqa: E402

os.environ["PW_FAST_QUIET"] = "1"
os.environ["PW_FAST_BACKOFF_S"] = "0.01"
CH = 64 * 1024                                       # small chunks so a tiny file has many


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(n, seed=1):
    out, x = bytearray(), seed
    while len(out) < n:
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out += x.to_bytes(4, "little")
    return bytes(out[:n])


class Hub:
    """A fake Hub + CDN on one port. /<repo>/resolve/<rev>/<path> -> 302 to /cdn/<token>/<path>;
    /cdn/... serves ranges. Knobs: expire_after (requests before a token turns into 403), fail (status for the next n range
    requests), delay (seconds per range request), a count of in-flight requests."""
    def __init__(self, files):
        self.files = files
        self.lock = threading.Lock()
        self.reset()
        hub = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                hub.handle(self)
        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def reset(self):
        self.expire_after = None
        self.token_expiring = 0
        self.fail = 0
        self.fail_status = 500
        self.delay = 0.0
        self.range_requests = []
        self.resolves = 0
        self.token = 0
        self.token_hits = 0
        self.inflight = 0
        self.max_inflight = 0
        self.auth_seen = []

    def send(self, h, code, body=b"", headers=()):
        h.send_response(code)
        for k, v in headers:
            h.send_header(k, v)
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def handle(self, h):
        path = h.path
        if "/api/models/" in path:
            sib = [{"rfilename": p, "size": len(b), "lfs": {"sha256": sha(b)} if len(b) > 1000 else None} for p, b in self.files.items()]
            return self.send(h, 200, json.dumps({"siblings": sib}).encode(), [("Content-Type", "application/json")])
        m = re.match(r"^/(.+)/resolve/([0-9a-f]+)/(.+)$", path)
        if m:
            with self.lock:
                self.resolves += 1
                self.auth_seen.append(h.headers.get("Authorization"))
                self.token += 1
                tok = self.token
            name = m.group(3)
            if name not in self.files:
                return self.send(h, 404)
            return self.send(h, 302, headers=[("Location", f"{self.base}/cdn/{tok}/{name}")])
        m = re.match(r"^/cdn/(\d+)/(.+)$", path)
        if not m:
            return self.send(h, 404)
        tok, name = int(m.group(1)), m.group(2)
        data = self.files[name]
        rng = re.match(r"bytes=(\d+)-(\d+)", h.headers.get("Range", ""))
        with self.lock:
            self.range_requests.append((name, int(rng.group(1)) if rng else None))
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            if h.headers.get("Authorization"):
                self.auth_seen.append("CDN:" + h.headers["Authorization"])
            expired = self.expire_after is not None and len(self.range_requests) > self.expire_after and tok <= self.token_expiring
            failing = self.fail > 0
            if failing:
                self.fail -= 1
        try:
            if self.delay:
                time.sleep(self.delay)
            if expired:
                return self.send(h, 403, b"expired")
            if failing:
                return self.send(h, self.fail_status)
            if not rng:
                return self.send(h, 200, data)
            s, e = int(rng.group(1)), min(int(rng.group(2)), len(data) - 1)
            if s >= len(data):
                return self.send(h, 416, headers=[("Content-Range", f"bytes */{len(data)}")])
            self.send(h, 206, data[s:e + 1], [("Content-Range", f"bytes {s}-{e}/{len(data)}")])
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.inflight -= 1

    def url(self, name, repo="org/model", rev="a" * 40):
        return f"{self.base}/{repo}/resolve/{rev}/{name}"


class FetchCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = blob(CH * 40 + 1234)              # 41 chunks, last one short
        cls.hub = Hub({"big.bin": cls.data, "cfg.json": b'{"a": 1}', "sub/shard.gguf": cls.data[:CH * 5 + 7], "empty": b""})

    def setUp(self):
        self.hub.reset()
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="ffetch-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dest = str(self.tmp / "big.bin")

    def fetch(self, **kw):
        kw.setdefault("streams", 6)
        kw.setdefault("chunk", CH)
        kw.setdefault("flush_s", 0.02)
        return fastfetch.fetch_file(self.hub.url("big.bin"), self.dest, sha(self.data), **kw)

    def test_assembled_bytes_and_sidecars_gone(self):
        r = self.fetch()
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)
        self.assertEqual(r["sha256"], sha(self.data))
        self.assertEqual(sorted(os.listdir(self.tmp)), ["big.bin"])
        self.assertEqual(self.hub.resolves, 1)          # resolved once, not per chunk

    def test_empty_file(self):
        r = fastfetch.fetch_file(self.hub.url("empty"), str(self.tmp / "e"), sha(b""), 4, CH)
        self.assertEqual(pathlib.Path(self.tmp / "e").read_bytes(), b"")
        self.assertEqual(r["size"], 0)

    def test_concurrency_limit(self):
        self.hub.delay = 0.03
        r = self.fetch(streams=5)
        self.assertLessEqual(self.hub.max_inflight, 5 + 1)   # +1: the resolving probe never overlaps, slack for accounting
        self.assertGreaterEqual(self.hub.max_inflight, 3)    # and it really is parallel
        self.assertLessEqual(r["max_inflight"], 5)

    def test_one_stream(self):
        self.fetch(streams=1)
        self.assertEqual(self.hub.max_inflight, 1)
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)

    def test_transient_errors_retry(self):
        self.hub.fail, self.hub.fail_status = 6, 503
        self.fetch(retries=10)
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)
        self.hub.reset()
        self.hub.fail, self.hub.fail_status = 3, 429
        self.fetch(retries=10)

    def test_persistent_failure_keeps_partial_then_resume_fetches_only_the_rest(self):
        # chunks 0..n succeed, then the server fails for good: the call raises and leaves .part + .part.map
        self.hub.delay = 0.01

        def break_later():
            while len(self.hub.range_requests) < 14:
                time.sleep(0.002)
            self.hub.fail, self.hub.fail_status = 10 ** 6, 500
        t = threading.Thread(target=break_later)
        t.start()
        with self.assertRaises(fastfetch.FetchError):
            self.fetch(retries=1, streams=3)
        t.join()
        self.assertFalse(os.path.exists(self.dest))                 # never marked complete
        self.assertTrue(os.path.exists(self.dest + ".part.map"))
        done = sum(self.hub_map_done())
        self.assertGreater(done, 2)
        self.hub.reset()
        r = self.fetch()
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)
        self.assertGreater(r["resumed_chunks"], 0)
        self.assertLess(len(self.hub.range_requests), 41 - r["resumed_chunks"] + 1 + 1)   # resumed chunks were not fetched again
        self.assertLess(len(self.hub.range_requests), 41)

    def hub_map_done(self):
        raw = pathlib.Path(self.dest + ".part.map").read_bytes()
        return [1 if b else 0 for b in raw[fastfetch.HEADER.size:]]

    def test_sigkill_then_resume(self):
        self.hub.delay = 0.05
        env = dict(os.environ, PW_FAST_FLUSH_S="0.05", PW_FAST_STREAMS="3", PW_FAST_CHUNK_MIB=str(CH / (1 << 20)))
        p = subprocess.Popen([sys.executable, str(ROOT / "tools/rig/fastfetch.py"), "file", "--url", self.hub.url("big.bin"),
                              "--dest", self.dest, "--sha256", sha(self.data)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline and not (os.path.exists(self.dest + ".part.map") and sum(self.hub_map_done()) >= 8):
            time.sleep(0.02)
        p.send_signal(signal.SIGKILL)
        p.wait()
        self.assertFalse(os.path.exists(self.dest))
        before = len(self.hub.range_requests)
        self.hub.delay = 0
        r = self.fetch()
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)
        self.assertGreater(r["resumed_chunks"], 0)
        self.assertLess(len(self.hub.range_requests) - before, 41)

    def _plant_partial(self, damage):
        """A partial whose sidecar claims every chunk is done, with `damage(buf)` applied to the data."""
        n = (len(self.data) + CH - 1) // CH
        buf = bytearray(self.data)
        damage(buf)
        pathlib.Path(self.dest + ".part").write_bytes(bytes(buf))
        pathlib.Path(self.dest + ".part.map").write_bytes(
            fastfetch.HEADER.pack(fastfetch.MAGIC, len(self.data), CH, bytes.fromhex(sha(self.data))) + b"\x01" * n)

    def test_corrupt_tail_is_repaired(self):
        def zero_tail(b):
            b[-CH:] = bytes(CH)                      # the last full-or-short chunk zeroed, as after a reboot
        self._plant_partial(zero_tail)
        r = self.fetch()
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)
        self.assertEqual(r["resumed_chunks"], 41 - fastfetch.REDO)

    def test_corruption_beyond_the_redo_window_is_caught_and_refetched_once(self):
        def zero_middle(b):
            b[CH * 3:CH * 4] = bytes(CH)
        self._plant_partial(zero_middle)
        r = self.fetch()                              # resumed, hash mismatch, deleted, fetched again from scratch
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)
        self.assertEqual(r["resumed_chunks"], 0)

    def test_sha_mismatch_deletes_and_fails(self):
        with self.assertRaises(fastfetch.ShaMismatch):
            fastfetch.fetch_file(self.hub.url("big.bin"), self.dest, "0" * 64, 4, CH, flush_s=0.02)
        self.assertEqual(os.listdir(self.tmp), [])    # no file, no .part, no .part.map
        self.assertEqual(len([r for r in self.hub.range_requests if r[1] == 0]), 2)   # probe + chunk 0, once: a clean mismatch is not retried

    def test_existing_good_file_is_kept_bad_one_replaced(self):
        pathlib.Path(self.dest).write_bytes(self.data)
        r = self.fetch()
        self.assertTrue(r.get("cached"))
        self.assertEqual(self.hub.range_requests, [])
        pathlib.Path(self.dest).write_bytes(self.data[:-1] + b"x")
        self.fetch()
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)

    def test_expired_url_is_re_resolved(self):
        self.hub.delay = 0.005
        self.hub.expire_after = 10          # after 10 range requests, tokens issued so far answer 403
        self.hub.token_expiring = 1
        self.fetch()
        self.assertEqual(pathlib.Path(self.dest).read_bytes(), self.data)
        self.assertGreaterEqual(self.hub.resolves, 2)
        self.assertLessEqual(self.hub.resolves, 2 + 1)   # one shared re-resolve, not one per worker

    def test_missing_file_is_permanent(self):
        with self.assertRaises(fastfetch.Permanent):
            fastfetch.fetch_file(self.hub.url("nope.bin"), str(self.tmp / "x"), None, 2, CH)

    def test_token_goes_to_the_origin_only(self):
        # the fake CDN is the same host:port, so use a different host name for the redirect target through 127.0.0.1 vs localhost
        orig = self.hub.handle

        def redirecting(h):
            if "/resolve/" in h.path:
                self.hub.token += 1
                return self.hub.send(h, 302, headers=[("Location", f"http://localhost:{self.hub.port}/cdn/1/big.bin")])
            return orig(h)
        self.hub.handle = redirecting
        try:
            fastfetch.fetch_file(self.hub.url("big.bin"), self.dest, sha(self.data), 2, CH, token="sekret", flush_s=0.02)
        finally:
            self.hub.handle = orig
        self.assertFalse([a for a in self.hub.auth_seen if a and a.startswith("CDN:")])

    def test_files_cli_flat_and_snapshot(self):
        sums = self.tmp / "model.sha256"
        sums.write_text(f"# c\n{sha(self.data[:CH * 5 + 7])}  sub/shard.gguf\n")
        out = self.tmp / "out"
        env = dict(os.environ, PW_FAST_CHUNK_MIB=str(CH / (1 << 20)))
        base = f"{self.hub.base}/org/model/resolve/{'a' * 40}"
        r = subprocess.run([sys.executable, str(ROOT / "tools/rig/fastfetch.py"), "files", "--base", base, "--sums", str(sums),
                            "--dest", str(out), "--flat"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((out / "shard.gguf").read_bytes(), self.data[:CH * 5 + 7])
        sums.write_text(f"{'0' * 64}  sub/shard.gguf\n")
        shutil.rmtree(out)
        r = subprocess.run([sys.executable, str(ROOT / "tools/rig/fastfetch.py"), "files", "--base", base, "--sums", str(sums),
                            "--dest", str(out), "--flat"], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertFalse(list(out.glob("*")))
        # snapshot: pinned shard + glob for the small config; the big non-matching file is left out
        os.environ["PW_HF_ENDPOINT"] = self.hub.base
        try:
            d, verified = fastfetch.snapshot("org/model", "a" * 40, str(self.tmp / "snap"), ["*.json"],
                                             {"sub/shard.gguf": sha(self.data[:CH * 5 + 7])}, streams=3, chunk=CH)
        finally:
            del os.environ["PW_HF_ENDPOINT"]
        self.assertEqual(sorted(os.listdir(d)), ["cfg.json", "sub"])
        self.assertEqual(verified, {"sub/shard.gguf"})


class WiringCase(unittest.TestCase):
    """PW_FAST_FETCH is off by default and only then touches the old download paths."""
    def test_lc_fetch_shards_uses_curl_without_the_flag_and_fastfetch_with_it(self):
        hub = Hub({"m-00001-of-00001.gguf": blob(CH * 3 + 5)})
        data = hub.files["m-00001-of-00001.gguf"]
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="ffwire-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        wl = tmp / "llamacpp-toy"
        wl.mkdir()
        (wl / "model.sha256").write_text(f"{sha(data)}  m-00001-of-00001.gguf\n")
        fake = tmp / "bin"
        fake.mkdir()
        (fake / "curl").write_text(f"#!/bin/sh\necho CURL \"$@\" >> {tmp}/curl.log\nexit 22\n")
        (fake / "curl").chmod(0o755)
        script = f'''
set -u
PW_TARGET=cpu; PW_OUT={tmp}/out; PW_SIM_PROFILE=x
. {ROOT}/tools/llamacpp/common.sh
lc_fetch_shards "{hub.base}/org/m/resolve/{'b' * 40}" "{wl}/model.sha256" 1
echo "MODEL=$LC_MODEL"
'''
        base_env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", PW_CACHE=str(tmp / "cache"), PW_FAST_CHUNK_MIB=str(CH / (1 << 20)))
        base_env.pop("PW_MODEL_MIRROR", None)
        off = subprocess.run(["bash", "-c", script], env=dict(base_env, PW_FAST_FETCH="0"), capture_output=True, text=True)
        self.assertEqual(off.returncode, 77, off.stdout + off.stderr)          # old path: fake curl fails -> SKIP
        self.assertIn("CURL", (tmp / "curl.log").read_text())
        (tmp / "curl.log").unlink()
        on = subprocess.run(["bash", "-c", script], env=dict(base_env, PW_FAST_FETCH="1", PW_FAST_STREAMS="3"), capture_output=True, text=True)
        self.assertEqual(on.returncode, 0, on.stdout + on.stderr)
        self.assertFalse((tmp / "curl.log").exists())                           # curl never ran
        model = re.search(r"MODEL=(.*)", on.stdout).group(1)
        self.assertEqual(pathlib.Path(model).read_bytes(), data)
        # a wrong pin fails the run (exit 1), never a SKIP
        (wl / "model.sha256").write_text(f"{'0' * 64}  m-00001-of-00001.gguf\n")
        shutil.rmtree(tmp / "cache")
        bad = subprocess.run(["bash", "-c", script], env=dict(base_env, PW_FAST_FETCH="1"), capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)

    def test_pinned_snapshot_flag(self):
        sys.path.insert(0, str(ROOT / "workloads" / "_pytorch"))
        import pinned
        hub = Hub({"w.safetensors": blob(CH * 2 + 3), "config.json": b'{"x": 1}'})
        data = hub.files["w.safetensors"]
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="ffsnap-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        wd = tmp / "w"
        wd.mkdir()
        (wd / "model.sha256").write_text(f"{sha(data)}  w.safetensors\n")
        env = {"PW_WORKLOAD_DIR": str(wd), "PW_HF_ENDPOINT": hub.base, "PW_FAST_FETCH": "1", "PW_FAST_CHUNK_MIB": str(CH / (1 << 20))}
        old = {k: os.environ.get(k) for k in [*env, "PW_MODEL_MIRROR", "PW_HF_REVISION"]}
        os.environ.pop("PW_MODEL_MIRROR", None)
        os.environ.pop("PW_HF_REVISION", None)
        os.environ.update(env)
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v) for k, v in old.items()])
        path = pinned.snapshot("org/w", "c" * 40, str(tmp / "cache"), ["*.json", "w.safetensors"])
        self.assertEqual(pathlib.Path(path, "w.safetensors").read_bytes(), data)
        self.assertTrue(pathlib.Path(path, "config.json").exists())
        pinned.verify_snapshot(path, str(wd / "model.sha256"))            # still passes; skips the re-hash it already did
        # off: the old path is taken (huggingface_hub), the hub fake sees nothing
        os.environ["PW_FAST_FETCH"] = "0"
        hub.reset()
        try:
            pinned.snapshot("org/w", "c" * 40, str(tmp / "cache2"), ["*.json"])
        except Exception:   # noqa: BLE001  huggingface_hub missing or offline: either way not our server
            pass
        self.assertEqual(hub.resolves, 0)




class NvmeCase(unittest.TestCase):
    """tools/rig/nvme-setup.sh against fake lsblk/mdadm/mkfs/mount (commands are logged to $LOG)."""
    SCRIPT = ROOT / "tools" / "rig" / "nvme-setup.sh"

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="nvme-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.log = self.tmp / "log"
        self.mnt = self.tmp / "mnt"
        self.mounts = self.tmp / "mounts"
        self.mounts.write_text("/dev/nvme0n1p1 / ext4 rw 0 0\n")
        self.bin = self.tmp / "bin"
        self.bin.mkdir()

    def fake(self, name, body):
        p = self.bin / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)

    def lsblk(self, rows):
        out = "\n".join(rows)
        self.fake("lsblk", f"""case "$*" in
  *-r*) cat <<'EOF'
{out}
EOF
;; esac
""")

    def run_script(self, *args, **env):
        for n in ("mdadm", "mkfs.xfs", "mkfs.ext4", "mount", "chown", "blkid", "udevadm"):
            if not (self.bin / n).exists():
                self.fake(n, f'echo "{n} $*" >> {self.log}\n' + ('exit 1\n' if n == "blkid" else "exit 0\n"))
        e = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", PW_NVME_MNT=str(self.mnt), PW_NVME_PROC_MOUNTS=str(self.mounts),
                 PW_NVME_SKIP_ROOT_CHECK="1", PW_NVME_USER=os.environ.get("USER", "root"))
        e.update(env)
        return subprocess.run(["bash", str(self.SCRIPT), *args], env=e, capture_output=True, text=True)

    def calls(self):
        return [c for c in self.log.read_text().splitlines() if not c.startswith("chown")] if self.log.exists() else []

    ROOT_EBS = "/dev/nvme0n1 disk Amazon\\x20Elastic\\x20Block\\x20Store"

    def test_no_instance_store_is_a_clean_exit_zero(self):
        self.lsblk([self.ROOT_EBS])
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no instance-store", r.stdout)
        self.assertIn("gp3", r.stdout)
        self.assertEqual(self.calls(), [])

    def test_one_device_gets_a_filesystem_and_a_mount(self):
        self.lsblk([self.ROOT_EBS, "/dev/nvme1n1 disk Amazon\\x20EC2\\x20NVMe\\x20Instance\\x20Storage"])
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.calls()
        self.assertTrue(any(c.startswith("mkfs.xfs") and "/dev/nvme1n1" in c for c in calls), calls)
        self.assertFalse(any(c.startswith("mdadm --create") for c in calls))
        self.assertTrue(any(c.startswith("mount") and "noatime" in c and str(self.mnt) in c for c in calls), calls)
        self.assertIn(f"export PW_CACHE={self.mnt}/pw-cache", r.stdout)
        self.assertIn(f"export HF_HOME={self.mnt}/hf", r.stdout)
        self.assertTrue((self.mnt / "pw-cache").is_dir())
        self.assertFalse(any("nvme0n1" in c for c in calls))                 # never the EBS root

    def test_several_devices_make_a_raid0(self):
        self.lsblk([self.ROOT_EBS] + [f"/dev/nvme{i}n1 disk Amazon\\x20EC2\\x20NVMe\\x20Instance\\x20Storage" for i in (1, 2, 3, 4)])
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.calls()
        create = [c for c in calls if c.startswith("mdadm --create")]
        self.assertEqual(len(create), 1, calls)
        self.assertIn("--level=0", create[0])
        self.assertIn("--raid-devices=4", create[0])
        self.assertTrue(any(c.startswith("mkfs.xfs") and "/dev/md" in c for c in calls), calls)

    def test_idempotent_when_already_mounted(self):
        self.lsblk([self.ROOT_EBS, "/dev/nvme1n1 disk Amazon\\x20EC2\\x20NVMe\\x20Instance\\x20Storage"])
        self.mnt.mkdir()
        self.mounts.write_text(f"/dev/md127 {self.mnt} xfs rw,noatime 0 0\n")
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.calls(), [])
        self.assertIn("already mounted", r.stdout)
        self.assertIn("export PW_CACHE", r.stdout)

    def test_busy_device_is_not_touched(self):
        self.lsblk([self.ROOT_EBS, "/dev/nvme1n1 disk Amazon\\x20EC2\\x20NVMe\\x20Instance\\x20Storage"])
        self.mounts.write_text("/dev/nvme0n1p1 / ext4 rw 0 0\n/dev/nvme1n1 /data xfs rw 0 0\n")
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.calls(), [])

    def test_dlami_mount_is_bound_not_reformatted(self):
        self.lsblk([self.ROOT_EBS, "/dev/nvme1n1 disk Amazon\\x20EC2\\x20NVMe\\x20Instance\\x20Storage"])
        dl = self.tmp / "dlami"
        dl.mkdir()
        self.mounts.write_text(f"/dev/nvme0n1p1 / ext4 rw 0 0\n/dev/mapper/vg.01-lv_ephemeral {dl} ext4 rw,noatime 0 0\n")
        r = self.run_script(PW_NVME_DLAMI=str(dl))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.calls()
        self.assertTrue(any(c.startswith("mount --bind") and str(dl) in c and str(self.mnt) in c for c in calls), calls)
        self.assertFalse(any(c.startswith("mkfs") or c.startswith("mdadm") for c in calls), calls)
        self.assertIn("export PW_CACHE", r.stdout)

    def test_not_root_fails(self):
        self.lsblk([self.ROOT_EBS])
        r = self.run_script(PW_NVME_SKIP_ROOT_CHECK="0", PW_NVME_ID="echo 1000")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("root", r.stderr)


if __name__ == "__main__":
    unittest.main()
