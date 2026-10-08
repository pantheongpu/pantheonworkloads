"""Checksum helpers for workloads whose model is several files (imported by their main.py, beside common.py).

model.sha256 is in sha256sum format ("<sha256>  <path>"; lines starting with # are comments), paths relative
to the model snapshot directory.
"""
import hashlib
import os
import sys
import urllib.request


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path):
    sums = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                s, n = line.split(None, 1)
                sums[n.strip()] = s
    return sums


def verify_snapshot(snapshot_dir, sums_path):
    """Exit 1 (a failure, not a SKIP) when a pinned file of the snapshot has another sha256, or is missing."""
    for name, want in read_sums(sums_path).items():
        got = sha256_file(os.path.join(snapshot_dir, name))
        if got != want:
            sys.exit(f"{name} has sha256 {got}, the pin is {want}")


def fetch_asset(url, sha256, dest):
    """Download url to dest unless it is there; check sha256. Returns dest. Raises on a mismatch."""
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as f:
            f.write(r.read())
        os.replace(dest + ".part", dest)
    got = sha256_file(dest)
    if got != sha256:
        raise RuntimeError(f"{dest} has sha256 {got}, pinned {sha256}")
    return dest
