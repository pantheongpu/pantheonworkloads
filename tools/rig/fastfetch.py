#!/usr/bin/env python3
"""Resumable parallel ranged downloader for Hugging Face Hub files (docs/rigs.md, "Direct download on a rig").

One stream from the Hub CDN gives ~145 MiB/s, 24 parallel ranged streams several GiB/s (measured on c6in.xlarge, docs/rigs.md);
this fetches one file with N range requests at once, each worker pwrite()ing its chunk at its offset into a preallocated file.
Standard library only (http.client + threads); no aria2c, no hf_transfer.

    fastfetch.py file     --url URL --dest FILE [--sha256 HEX]
    fastfetch.py files    --base https://huggingface.co/<repo>/resolve/<rev> --sums model.sha256 --dest DIR [--flat]
    fastfetch.py snapshot --repo R --rev V --dest DIR [--pattern GLOB ...] [--sums model.sha256]

Exit status: 0 done (every pinned file verified), 1 sha256 mismatch (the file is deleted), 2 usage, 4 the transfer failed
(network, 4xx/5xx after the retries, disk full; partial data is kept for a resume).

How it stays correct:
* The URL is resolved once (the Hub's resolve/<commit>/<file> redirects to a signed CDN URL). When a chunk gets 401/403 (the signature
  expired) every worker waits for ONE re-resolve and carries on; a 4xx on the Hub itself (gated repo, 404) is permanent.
  HF_TOKEN is sent only to the host the first URL names, never to the CDN it redirects to.
* Data goes to <dest>.part; <dest>.part.map is a sidecar with one byte per chunk. A chunk is recorded in the map only after the data file
  was fsync()ed (batched every PW_FAST_FLUSH_S seconds), so a crash or reboot cannot leave a "done" chunk that was never written. On resume
  the 2 highest-numbered done chunks (PW_FAST_REDO) are fetched again anyway, because the earlier staging
  run left damaged partials after a reboot; the final sha256 pass catches anything else. <dest> exists only after the whole file hashed to the pin and was renamed into place.
* sha256 is computed in order by a thread that follows the downloaders (reading the pages just written), so verification costs no extra
  pass. A pinned file already on disk is hashed first and skipped when it matches. On a mismatch after a resumed download the file is
  deleted and fetched once more from scratch; after a clean download the mismatch is final (ShaMismatch, the file is deleted).
  A file with no pin (small configs) is checked by size only.
* Transient errors (connection resets, HTTP/2 stream cancels surfacing as resets, 429, 5xx, short reads) retry with exponential backoff and
  jitter. Memory: one 1 MiB buffer per stream.

Environment: PW_FAST_STREAMS (48), PW_FAST_CHUNK_MIB (48), PW_FAST_RETRIES (8), PW_FAST_FLUSH_S (2), PW_FAST_QUIET=1, HF_TOKEN,
PW_HF_ENDPOINT / HF_ENDPOINT (https://huggingface.co).
"""
import argparse
import concurrent.futures
import errno
import fnmatch
import hashlib
import http.client
import json
import os
import random
import ssl
import struct
import sys
import threading
import time
import urllib.parse

MAGIC = b"PWFF0001"
HEADER = struct.Struct("<8sQI32s")      # magic, file size, chunk size, expected sha256 (zeros when unpinned)
MiB = 1 << 20
REDO = 2                                # resumed: this many of the highest-numbered done chunks are fetched again
BUF = 1 * MiB


class FetchError(Exception):
    """The transfer failed (partial data is kept for a resume)."""


class Permanent(FetchError):
    """A retry cannot help: 404, a gated repository, a refused re-resolve."""


class ShaMismatch(Exception):
    """The downloaded file does not hash to the pin; it was deleted."""


class Transient(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * MiB), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def default_streams():
    return max(1, _env_int("PW_FAST_STREAMS", 48))


def default_chunk():
    return max(1, int(float(os.environ.get("PW_FAST_CHUNK_MIB", "48")) * MiB))


def hub_endpoint():
    return (os.environ.get("PW_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def hub_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


class Target:
    """A resolved request target: where chunks are fetched from right now."""
    def __init__(self, url, send_auth):
        u = urllib.parse.urlsplit(url)
        self.scheme, self.host = u.scheme, u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.path = (u.path or "/") + (("?" + u.query) if u.query else "")
        self.send_auth = send_auth

    def key(self):
        return (self.scheme, self.host, self.port)


_local = threading.local()


class Conns:
    """The keep-alive connections one download (or listing) made, one per thread and host; close() ends them all."""
    def __init__(self):
        self.lock = threading.Lock()
        self.all = []

    def close(self):
        with self.lock:
            conns, self.all = self.all, []
        for c in conns:
            try:
                c.close()
            except Exception:   # noqa: BLE001
                pass


def _connection(t, timeout, owner):
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    c = conns.get((id(owner), *t.key()))
    if c is None:
        if t.scheme == "https":
            c = http.client.HTTPSConnection(t.host, t.port, timeout=timeout, context=ssl.create_default_context())
        else:
            c = http.client.HTTPConnection(t.host, t.port, timeout=timeout)
        conns[(id(owner), *t.key())] = c
        with owner.lock:
            owner.all.append(c)
    return c


def _drop_connection(t, owner):
    c = getattr(_local, "conns", {}).pop((id(owner), *t.key()), None)
    if c is not None:
        try:
            c.close()
        except Exception:   # noqa: BLE001
            pass


def _headers(t, token, rng):
    h = {"Range": rng, "User-Agent": "pantheonworkloads-fastfetch/1", "Accept-Encoding": "identity"}
    if token and t.send_auth:
        h["Authorization"] = f"Bearer {token}"
    return h


def _backoff(attempt, retry_after=None):
    if retry_after is not None:
        return min(float(retry_after), 120.0)
    base = float(os.environ.get("PW_FAST_BACKOFF_S", "1.0"))
    return min(base * (2 ** attempt), 60.0) * (0.5 + random.random() / 2)


class Resolver:
    """Resolves the Hub URL to a CDN target once; refresh() re-resolves (one caller does it, the rest reuse the result)."""
    def __init__(self, url, token, timeout, retries, owner):
        self.url, self.token, self.timeout, self.retries, self.owner = url, token, timeout, retries, owner
        self.lock = threading.Lock()
        self.gen = 0
        self.target = None
        self.size = None

    def get(self):
        with self.lock:
            if self.target is None:
                self.target, self.size = self._resolve()
            return self.gen, self.target

    def refresh(self, seen_gen):
        with self.lock:
            if self.gen == seen_gen:
                t, size = self._resolve()
                if self.size is not None and size != self.size:
                    raise Permanent(f"{self.url}: the file changed size while downloading ({self.size} -> {size})")
                self.target, self.size = t, size
                self.gen += 1
            return self.gen, self.target

    def _resolve(self):
        last = None
        for attempt in range(self.retries + 1):
            try:
                return self._resolve_once()
            except Transient as e:
                last = e
                time.sleep(_backoff(attempt))
        raise FetchError(f"cannot resolve {self.url}: {last}")

    def _resolve_once(self):
        origin = urllib.parse.urlsplit(self.url).hostname
        url = self.url
        for _ in range(8):
            t = Target(url, send_auth=(urllib.parse.urlsplit(url).hostname == origin))
            try:
                c = _connection(t, self.timeout, self.owner)
                c.request("GET", t.path, headers=_headers(t, self.token, "bytes=0-0"))
                r = c.getresponse()
                status, loc, cr = r.status, r.getheader("Location"), r.getheader("Content-Range")
                if status == 200:
                    _drop_connection(t, self.owner)      # the server ignores Range: do not read the whole body
                else:
                    r.read()
            except (OSError, http.client.HTTPException) as e:
                _drop_connection(t, self.owner)
                raise Transient(f"{type(e).__name__}: {e}")
            if status in (301, 302, 303, 307, 308):
                if not loc:
                    raise Permanent(f"{url}: redirect without Location")
                url = urllib.parse.urljoin(url, loc)
                continue
            if status == 206 and cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                return t, int(cr.rsplit("/", 1)[1])
            if status == 416:     # an empty file
                return t, 0
            if status in (401, 403, 404, 410):
                raise Permanent(f"{url}: HTTP {status} (404: no such file or revision; 401/403: gated or private, set HF_TOKEN)")
            if status == 200:
                raise Permanent(f"{url}: the server ignored the Range header (HTTP 200); parallel ranged download is not possible")
            if status == 429 or status >= 500:
                raise Transient(f"HTTP {status}")
            raise Permanent(f"{url}: unexpected HTTP {status}")
        raise Permanent(f"{self.url}: too many redirects")


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.bytes = 0
        self.inflight = 0
        self.max_inflight = 0
        self.requests = 0

    def add(self, n):
        with self.lock:
            self.bytes += n


class Download:
    """One file: chunks, sidecar map, hasher, workers. Use fetch_file()."""
    def __init__(self, url, dest, size_hint, sha256, streams, chunk, token, retries, flush_s, label, quiet):
        self.url, self.dest = url, dest
        self.part, self.map_path = dest + ".part", dest + ".part.map"
        self.expect = sha256.lower() if sha256 else None
        self.streams, self.chunk, self.token, self.retries = streams, chunk, token, retries
        self.flush_s, self.label, self.quiet = flush_s, label, quiet
        self.timeout = 60
        self.stats = Stats()
        self.abort = threading.Event()
        self.cond = threading.Condition()
        self.done = []
        self.pending = []           # done in memory, not yet durable in the map
        self.resumed = 0
        self.owner = Conns()
        self.resolver = Resolver(url, token, self.timeout, retries, self.owner)

    # ---- sidecar -------------------------------------------------------------------------------
    def _load_map(self, size, n):
        """Done flags from a matching sidecar, else all zero (and any stale partial removed)."""
        done = bytearray(n)
        try:
            with open(self.map_path, "rb") as f:
                raw = f.read()
            magic, msize, mchunk, msha = HEADER.unpack_from(raw)
            want_sha = bytes.fromhex(self.expect) if self.expect else b"\0" * 32
            if (magic == MAGIC and msize == size and mchunk == self.chunk and msha == want_sha
                    and len(raw) == HEADER.size + n and os.path.getsize(self.part) == size):
                done = bytearray(1 if b else 0 for b in raw[HEADER.size:])
        except (OSError, struct.error, ValueError):
            pass
        if not any(done):
            for p in (self.part, self.map_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
            return done, False
        redo = min(sum(done), max(0, _env_int("PW_FAST_REDO", REDO)))
        for i in sorted((i for i, b in enumerate(done) if b), reverse=True)[:redo]:
            done[i] = 0
        return done, True

    def _init_files(self, size, n):
        done, resumed = self._load_map(size, n)
        want_sha = bytes.fromhex(self.expect) if self.expect else b"\0" * 32
        if not resumed:
            with open(self.map_path, "wb") as m:
                m.write(HEADER.pack(MAGIC, size, self.chunk, want_sha) + bytes(n))
                m.flush()
                os.fsync(m.fileno())
            fd = os.open(self.part, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.posix_fallocate(fd, 0, size) if size else None
            except OSError as e:
                if e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EFBIG):
                    os.close(fd)
                    raise FetchError(f"no room for {size} bytes at {self.part}: {e}")
                os.ftruncate(fd, size)      # a file system without fallocate
            self.fd = fd
        else:
            self.fd = os.open(self.part, os.O_RDWR)
        self.mapfd = os.open(self.map_path, os.O_RDWR)
        self.done = done
        self.resumed = sum(done)

    def _flush(self):
        with self.cond:
            batch, self.pending = self.pending, []
        if not batch:
            return
        os.fsync(self.fd)                  # data first, then the records that say it is there
        for i in batch:
            os.pwrite(self.mapfd, b"\x01", HEADER.size + i)
        os.fsync(self.mapfd)

    # ---- workers -------------------------------------------------------------------------------
    def _chunk_bounds(self, i, size):
        s = i * self.chunk
        return s, min(size, s + self.chunk) - 1

    def _fetch_chunk(self, i, size):
        s, e = self._chunk_bounds(i, size)
        last = None
        for attempt in range(self.retries + 1):
            if self.abort.is_set():
                return
            gen, t = self.resolver.get()
            written = 0
            try:
                with self.stats.lock:
                    self.stats.requests += 1
                    self.stats.inflight += 1
                    self.stats.max_inflight = max(self.stats.max_inflight, self.stats.inflight)
                try:
                    c = _connection(t, self.timeout, self.owner)
                    c.request("GET", t.path, headers=_headers(t, self.token, f"bytes={s}-{e}"))
                    r = c.getresponse()
                    if r.status in (401, 403):
                        r.read()
                        self.resolver.refresh(gen)      # expired signature: one worker re-resolves, all reuse it
                        last = f"HTTP {r.status} (re-resolved)"
                        continue
                    if r.status == 429 or r.status >= 500:
                        ra = r.getheader("Retry-After")
                        r.read()
                        time.sleep(_backoff(attempt, float(ra) if ra and ra.isdigit() else None))
                        last = f"HTTP {r.status}"
                        continue
                    if r.status != 206:
                        raise Permanent(f"chunk {i}: HTTP {r.status}")
                    cr = r.getheader("Content-Range") or ""
                    if not cr.startswith(f"bytes {s}-{e}/"):
                        raise Transient(f"Content-Range {cr!r}, wanted bytes {s}-{e}")
                    buf = bytearray(min(BUF, e - s + 1))
                    mv = memoryview(buf)
                    pos = s
                    while pos <= e:
                        n = r.readinto(mv[:min(len(buf), e - pos + 1)])
                        if not n:
                            raise Transient(f"short read at {pos} of {e}")
                        os.pwrite(self.fd, mv[:n], pos)
                        pos += n
                        written += n
                    r.read()
                finally:
                    with self.stats.lock:
                        self.stats.inflight -= 1
                self.stats.add(written)
                with self.cond:
                    self.done[i] = 1
                    self.pending.append(i)
                    self.cond.notify_all()
                return
            except (OSError, http.client.HTTPException, Transient) as ex:
                if isinstance(ex, OSError) and ex.errno in (errno.ENOSPC, errno.EDQUOT):
                    raise FetchError(f"disk full writing {self.part}: {ex}")
                _drop_connection(t, self.owner)
                last = f"{type(ex).__name__}: {ex}"
                time.sleep(_backoff(attempt))
        raise FetchError(f"chunk {i} of {self.url} failed after {self.retries + 1} tries: {last}")

    def _hasher(self, size, n, out):
        h = hashlib.sha256()
        try:
            for i in range(n):
                with self.cond:
                    while not self.done[i]:
                        if self.abort.is_set():
                            return
                        self.cond.wait(0.5)
                s, e = self._chunk_bounds(i, size)
                pos = s
                while pos <= e:
                    data = os.pread(self.fd, min(4 * MiB, e - pos + 1), pos)
                    if not data:
                        raise OSError("short pread")
                    h.update(data)
                    pos += len(data)
            out.append(h.hexdigest())
        except Exception as ex:   # noqa: BLE001
            out.append(ex)

    # ---- driver --------------------------------------------------------------------------------
    def run(self):
        try:
            return self._run()
        finally:
            self.owner.close()

    def _run(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.dest)), exist_ok=True)
        _, _t = self.resolver.get()
        size = self.resolver.size
        n = (size + self.chunk - 1) // self.chunk
        self._init_files(size, n)
        result, t0 = [], time.time()
        hasher = threading.Thread(target=self._hasher, args=(size, n, result), daemon=True)
        hasher.start()
        todo = [i for i in range(n) if not self.done[i]]
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(self.streams, len(todo) or 1)))
        futs = [pool.submit(self._fetch_chunk, i, size) for i in todo]
        try:
            last_log = time.time()
            pending = set(futs)
            while pending:
                d, pending = concurrent.futures.wait(pending, timeout=self.flush_s,
                                                     return_when=concurrent.futures.FIRST_EXCEPTION)
                for f in d:
                    f.result()
                self._flush()
                if not self.quiet and time.time() - last_log > 10:
                    last_log = time.time()
                    rate = self.stats.bytes / max(1e-9, last_log - t0) / MiB
                    print(f"fastfetch {self.label}: {sum(self.done) * self.chunk / 2**30:.1f}/{size / 2**30:.1f} GiB, {rate:.0f} MiB/s",
                          file=sys.stderr, flush=True)
        except BaseException:
            self.abort.set()
            for f in futs:
                f.cancel()
            pool.shutdown(wait=True)
            try:
                self._flush()
            finally:
                with self.cond:
                    self.cond.notify_all()
                hasher.join(5)
                os.close(self.fd)
                os.close(self.mapfd)
            raise
        pool.shutdown(wait=True)
        self._flush()
        hasher.join()
        os.fsync(self.fd)
        os.close(self.fd)
        os.close(self.mapfd)
        got = result[0] if result else None
        if isinstance(got, Exception) or got is None:
            raise FetchError(f"hashing {self.part} failed: {got}")
        if self.expect and got != self.expect:
            for p in (self.part, self.map_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
            raise _Mismatch(got, self.resumed > 0)
        os.replace(self.part, self.dest)
        try:
            os.unlink(self.map_path)
        except FileNotFoundError:
            pass
        _fsync_dir(os.path.dirname(os.path.abspath(self.dest)))
        return {"path": self.dest, "size": size, "sha256": got, "downloaded": self.stats.bytes, "resumed_chunks": self.resumed,
                "requests": self.stats.requests, "max_inflight": self.stats.max_inflight, "seconds": time.time() - t0}


class _Mismatch(Exception):
    def __init__(self, got, was_resumed):
        self.got, self.was_resumed = got, was_resumed


def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def fetch_file(url, dest, sha256=None, streams=None, chunk=None, token=None, retries=None, flush_s=None, label=None, quiet=None):
    """Download url to dest (parallel ranged, resumable); returns {path, size, sha256, downloaded, ...}.
    With sha256 given, the result is verified: a mismatch deletes the file and raises ShaMismatch (after one fresh retry when the
    download had been resumed). A pinned file already at dest is hashed and kept when it matches, else replaced."""
    streams = streams or default_streams()
    chunk = chunk or default_chunk()
    retries = _env_int("PW_FAST_RETRIES", 8) if retries is None else retries
    flush_s = float(os.environ.get("PW_FAST_FLUSH_S", "2")) if flush_s is None else flush_s
    quiet = (os.environ.get("PW_FAST_QUIET") == "1") if quiet is None else quiet
    token = hub_token() if token is None else token
    label = label or os.path.basename(dest)
    if sha256 and os.path.isfile(dest):
        got = sha256_file(dest)
        if got == sha256.lower():
            return {"path": dest, "size": os.path.getsize(dest), "sha256": got, "downloaded": 0, "resumed_chunks": 0,
                    "requests": 0, "max_inflight": 0, "seconds": 0.0, "cached": True}
        os.unlink(dest)
    elif not sha256 and os.path.isfile(dest):
        return {"path": dest, "size": os.path.getsize(dest), "sha256": None, "downloaded": 0, "resumed_chunks": 0,
                "requests": 0, "max_inflight": 0, "seconds": 0.0, "cached": True}   # unpinned: dest only exists once complete
    for tries_left in (1, 0):
        d = Download(url, dest, None, sha256, streams, chunk, token, retries, flush_s, label, quiet)
        try:
            return d.run()
        except _Mismatch as m:
            if m.was_resumed and tries_left:
                print(f"fastfetch {label}: sha256 mismatch on a resumed file; fetching it again from scratch", file=sys.stderr)
                continue
            raise ShaMismatch(f"{dest} has sha256 {m.got}, expected {sha256}")
    raise AssertionError("unreachable")


def hub_listing(repo, rev, token=None, timeout=60):
    """{path: {"size": n, "sha256": lfs oid or None}} of the files of repo@rev, from the Hub API."""
    url = f"{hub_endpoint()}/api/models/{urllib.parse.quote(repo, safe='/')}/revision/{rev}?blobs=true"
    t = Target(url, True)
    token = hub_token() if token is None else token
    last = None
    owner = Conns()
    for attempt in range(6):
        try:
            c = _connection(t, timeout, owner)
            h = {"User-Agent": "pantheonworkloads-fastfetch/1"}
            if token:
                h["Authorization"] = f"Bearer {token}"
            c.request("GET", t.path, headers=h)
            r = c.getresponse()
            body = r.read()
            if r.status == 200:
                break
            if r.status in (401, 403, 404):
                raise Permanent(f"{url}: HTTP {r.status} (404: no such repository or revision; 401/403: gated or private, set HF_TOKEN)")
            last = f"HTTP {r.status}"
        except (OSError, http.client.HTTPException) as e:
            _drop_connection(t, owner)
            last = f"{type(e).__name__}: {e}"
        time.sleep(_backoff(attempt))
    else:
        owner.close()
        raise FetchError(f"cannot list {repo}@{rev}: {last}")
    owner.close()
    out = {}
    for s in json.loads(body).get("siblings", []):
        lfs = s.get("lfs") or {}
        out[s["rfilename"]] = {"size": s.get("size"), "sha256": lfs.get("sha256") or lfs.get("oid")}
    return out


def fetch_many(items, streams=None, chunk=None, small_pool=8, **kw):
    """items: [(url, dest, sha256 or None, size or None)]. Files smaller than one chunk go through a pool of small_pool
    single-stream downloads; larger ones run one at a time with all the streams. Returns {dest: result}."""
    chunk = chunk or default_chunk()
    out = {}
    small = [i for i in items if i[3] is not None and i[3] <= chunk]
    big = [i for i in items if i not in small]
    if small:
        with concurrent.futures.ThreadPoolExecutor(max_workers=small_pool) as ex:
            futs = {ex.submit(fetch_file, u, d, sha, 1, chunk, **kw): d for (u, d, sha, _s) in small}
            for f in concurrent.futures.as_completed(futs):
                out[futs[f]] = f.result()
    for (u, d, sha, _s) in big:
        out[d] = fetch_file(u, d, sha, streams, chunk, **kw)
    return out


def read_sums(path):
    sums = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                s, n = line.split(None, 1)
                sums[n.strip()] = s
    return sums


def resolve_url(repo, rev, path):
    return f"{hub_endpoint()}/{urllib.parse.quote(repo, safe='/')}/resolve/{rev}/{urllib.parse.quote(path)}"


def snapshot(repo, rev, dest, patterns, expected=None, streams=None, chunk=None, **kw):
    """The files of repo@rev that match the glob patterns (and every pinned path in `expected`) into dest/<path>.
    Pinned files are verified against expected[path]; others by size. Returns (dest, set of verified paths)."""
    expected = dict(expected or {})
    listing = hub_listing(repo, rev)
    names = sorted(p for p in listing if p in expected or any(fnmatch.fnmatch(p, g) for g in patterns))
    missing = sorted(set(expected) - set(listing))
    if missing:
        raise Permanent(f"{repo}@{rev[:8]}: pinned file(s) not in the repository: {', '.join(missing)}")
    items = [(resolve_url(repo, rev, p), os.path.join(dest, p), expected.get(p), listing[p]["size"]) for p in names]
    fetch_many(items, streams, chunk, **kw)
    return dest, {p for p in names if p in expected}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("file", "files", "snapshot"):
        p = sub.add_parser(name)
        p.add_argument("--streams", type=int)
        p.add_argument("--chunk-mib", type=float)
        if name == "file":
            p.add_argument("--url", required=True)
            p.add_argument("--dest", required=True)
            p.add_argument("--sha256")
        elif name == "files":
            p.add_argument("--base", required=True)
            p.add_argument("--sums", required=True)
            p.add_argument("--dest", required=True)
            p.add_argument("--flat", action="store_true")
        else:
            p.add_argument("--repo", required=True)
            p.add_argument("--rev", required=True)
            p.add_argument("--dest", required=True)
            p.add_argument("--pattern", action="append", default=[])
            p.add_argument("--sums")
    a = ap.parse_args(argv)
    streams = a.streams
    chunk = int(a.chunk_mib * MiB) if a.chunk_mib else None
    try:
        if a.cmd == "file":
            r = fetch_file(a.url, a.dest, a.sha256, streams, chunk)
            print(json.dumps(r))
        elif a.cmd == "files":
            sums = read_sums(a.sums)
            items = [(f"{a.base.rstrip('/')}/{urllib.parse.quote(p)}",
                      os.path.join(a.dest, os.path.basename(p) if a.flat else p), s, None) for p, s in sums.items()]
            fetch_many(items, streams, chunk)
        else:
            snapshot(a.repo, a.rev, a.dest, a.pattern, read_sums(a.sums) if a.sums else {}, streams, chunk)
            print(a.dest)
    except ShaMismatch as e:
        print(f"fastfetch: {e}", file=sys.stderr)
        return 1
    except (FetchError, OSError, http.client.HTTPException) as e:
        print(f"fastfetch: transfer failed: {e}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
