#!/usr/bin/env python3
"""Consumer side of the model mirror (docs/rigs.md): copy a pinned model's files from PW_MODEL_MIRROR instead of the Hub.

A mirror is a prefix, an `s3://bucket/pw-weights` URL or a local directory, holding `<repo>/<revision>/manifest.json`
and the files beside it (written by tools/rig/stage_weights.py). `fetch` copies the files you ask for into a local
directory, in parallel, resumable (a file already there that was verified is skipped), and verifies the sha256 of every
file it copies against manifest.json and, when you pass them, against the pins (model.sha256). A mismatch deletes the
copy and fails; it is never retried from the Hub, because the pin is the truth and the mirror is not.

    mirror.py fetch --mirror M --repo R --rev V --dest DIR [--name PATH ...] [--pattern GLOB ...] [--sums model.sha256]

Exit status: 0 fetched (DIR on stdout), 3 miss (no manifest or a wanted file is not mirrored: the caller may use the Hub),
1 sha256 mismatch or a copy failed. Needs the `aws` CLI for s3:// mirrors (PW_AWS names another command, for tests).
"""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

SCHEMA = "pw-mirror/0"
MISS, MISMATCH = 3, 1


class Miss(Exception):
    """The mirror has no manifest for the model, or not a file that was asked for."""


class Mismatch(Exception):
    """A file's sha256 is not the one the manifest or the pin says."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path):
    """model.sha256: '<sha256>  <path>' lines, # comments."""
    sums = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                s, n = line.split(None, 1)
                sums[n.strip()] = s
    return sums


def is_s3(mirror):
    return str(mirror).startswith("s3://")


def location(mirror, repo, rev):
    return str(mirror).rstrip("/") + f"/{repo}/{rev}"


def _aws(args, capture=False):
    exe = os.environ.get("PW_AWS") or "aws"
    return subprocess.run([exe, *args], capture_output=True, text=capture)


def read_manifest(mirror, repo, rev):
    loc = location(mirror, repo, rev)
    if is_s3(mirror):
        p = _aws(["s3", "cp", "--only-show-errors", loc + "/manifest.json", "-"], capture=True)
        if p.returncode != 0:
            raise Miss(f"no manifest at {loc}/manifest.json ({(p.stderr or '').strip()[-120:]})")
        text = p.stdout
    else:
        f = pathlib.Path(loc) / "manifest.json"
        if not f.is_file():
            raise Miss(f"no manifest at {f}")
        text = f.read_text()
    m = json.loads(text)
    if m.get("schema") != SCHEMA:
        raise Miss(f"{loc}/manifest.json has schema {m.get('schema')!r}, expected {SCHEMA}")
    return m


def _copy(mirror, repo, rev, name, dest):
    src = location(mirror, repo, rev) + "/" + name
    if is_s3(mirror):
        p = _aws(["s3", "cp", "--only-show-errors", src, str(dest)], capture=True)
        if p.returncode != 0:
            raise OSError(f"aws s3 cp {src} failed: {(p.stderr or '').strip()[-200:]}")
    else:
        shutil.copyfile(src, dest)


def select(manifest, names, patterns):
    """The manifest paths wanted: every exact name (a name the mirror lacks is a Miss) plus every path matching a glob
    (fnmatch on the whole path, as huggingface_hub's allow_patterns does)."""
    files = manifest["files"]
    want = []
    for n in names:
        if n not in files:
            raise Miss(f"{n} is not in the mirror's manifest")
        want.append(n)
    for p in files:
        if p not in want and any(fnmatch.fnmatch(p, g) for g in patterns):
            want.append(p)
    return want


def fetch(mirror, repo, rev, dest, names=(), patterns=(), expected=None, jobs=8, flat=False):
    """Copy the wanted files into dest; returns dest (a pathlib.Path). Raises Miss or Mismatch. flat: every file goes
    directly in dest under its base name (the llama.cpp shard layout); two wanted files with one base name are refused."""
    manifest = read_manifest(mirror, repo, rev)
    expected = dict(expected or {})
    names = list(names) + [n for n in expected if n not in names]
    want = select(manifest, names, patterns)
    for n, pin in expected.items():
        have = manifest["files"][n]["sha256"]
        if have != pin:
            raise Mismatch(f"{n}: the mirror's manifest says sha256 {have}, the pin is {pin}")
    dest = pathlib.Path(dest)
    if flat and len({os.path.basename(n) for n in want}) != len(want):
        raise Mismatch("flat layout: two wanted files have the same base name")
    state_dir = dest / ".pw-mirror"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "verified.json"
    try:
        verified = json.loads(state_file.read_text())
    except (OSError, ValueError):
        verified = {}

    def one(name):
        meta = manifest["files"][name]
        target = dest / (os.path.basename(name) if flat else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        # already here and verified before (same size and mtime as then): skip
        if target.is_file():
            st = target.stat()
            v = verified.get(name)
            if v and v["sha256"] == meta["sha256"] and v["size"] == st.st_size == meta["size"] and v["mtime_ns"] == st.st_mtime_ns:
                return name, None
        part = target.with_name(target.name + ".part")
        _copy(mirror, repo, rev, name, part)
        got = sha256_file(part)
        if got != meta["sha256"]:
            part.unlink(missing_ok=True)
            raise Mismatch(f"{name} copied from the mirror has sha256 {got}, the manifest says {meta['sha256']}")
        os.replace(part, target)
        st = target.stat()
        return name, {"sha256": got, "size": st.st_size, "mtime_ns": st.st_mtime_ns}

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(one, n) for n in want]):
            try:
                name, rec = fut.result()
                if rec:
                    verified[name] = rec
            except (Mismatch, OSError) as e:
                errors.append(e)
    state_file.write_text(json.dumps(verified))
    if errors:
        mism = [e for e in errors if isinstance(e, Mismatch)]
        raise (mism or errors)[0]
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--mirror", default=os.environ.get("PW_MODEL_MIRROR"))
    f.add_argument("--repo", required=True)
    f.add_argument("--rev", required=True)
    f.add_argument("--dest", required=True)
    f.add_argument("--name", action="append", default=[], help="an exact path in the repository (repeatable)")
    f.add_argument("--pattern", action="append", default=[], help="a glob over repository paths (repeatable)")
    f.add_argument("--sums", help="model.sha256: its files are fetched and pinned")
    f.add_argument("--flat", action="store_true", help="put every file directly in --dest under its base name")
    f.add_argument("--jobs", type=int, default=int(os.environ.get("PW_MIRROR_JOBS", "8")))
    a = ap.parse_args(argv)
    if not a.mirror:
        print("no mirror: set PW_MODEL_MIRROR or pass --mirror", file=sys.stderr)
        return MISS
    try:
        d = fetch(a.mirror, a.repo, a.rev, a.dest, a.name, a.pattern, read_sums(a.sums) if a.sums else None, a.jobs, a.flat)
    except Miss as e:
        print(f"mirror miss: {e}", file=sys.stderr)
        return MISS
    except (Mismatch, OSError) as e:
        print(f"mirror failure: {e}", file=sys.stderr)
        return MISMATCH
    print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
