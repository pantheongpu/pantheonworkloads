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
    --scratch   where files are downloaded (default ~/.cache/pw-stage); needs room for the largest file times --jobs

Consumers set PW_MODEL_MIRROR to the same --mirror value (tools/rig/mirror.py, docs/rigs.md). HF_TOKEN is sent when set.
Environment for tests: PW_AWS (aws command), PW_CURL (curl command).
"""
import argparse
import fnmatch
import json
import os
import pathlib
import shutil
import subprocess
import sys
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


def _aws(args):
    p = subprocess.run([os.environ.get("PW_AWS") or "aws", *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise OSError(f"aws {' '.join(args[:3])} failed: {p.stderr.strip()[-200:]}")


def put(mirror, repo, rev, name, src):
    dst = mirrorlib.location(mirror, repo, rev) + "/" + name
    if mirrorlib.is_s3(mirror):
        _aws(["s3", "cp", "--only-show-errors", str(src), dst])
    else:
        pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)


def download(hf_base, repo, rev, path, dest, expect_size):
    url = f"{hf_base.rstrip('/')}/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
    if os.path.exists(dest):
        have = os.path.getsize(dest)
        if have == expect_size:
            return          # an earlier run finished the download but not the upload; the sha256 check follows
        if have > expect_size:
            os.unlink(dest)
    cmd = [os.environ.get("PW_CURL") or "curl", "-fL", "--retry", "3", "-C", "-", "-sS", "-o", str(dest)]
    if os.environ.get("HF_TOKEN"):
        cmd += ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
    p = subprocess.run(cmd + [url], capture_output=True, text=True)
    if p.returncode != 0:
        raise OSError(f"download of {url} failed (curl exit {p.returncode}): {p.stderr.strip()[-200:]}; the partial file is kept for a resume")
    if os.path.getsize(dest) != expect_size:
        raise OSError(f"{path}: downloaded {os.path.getsize(dest)} bytes, the catalog says {expect_size}")


def write_manifest(mirror, repo, rev, files, complete, scratch):
    m = {"schema": mirrorlib.SCHEMA, "repo": repo, "revision": rev, "complete": complete,
         "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "total_bytes": sum(f["size"] for f in files.values()), "files": files}
    tmp = pathlib.Path(scratch) / "manifest.json"
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    put(mirror, repo, rev, "manifest.json", tmp)


def stage_one(mirror, repo, rev, plan, hf_base, scratch):
    """Returns (staged_count, skipped_count, [errors])."""
    manifest = staged_manifest(mirror, repo, rev)
    files = dict((manifest or {}).get("files", {}))
    staged = skipped = 0
    errors = []
    d = pathlib.Path(scratch) / repo.replace("/", "__") / rev
    d.mkdir(parents=True, exist_ok=True)
    for f in plan:
        if already(manifest, f):
            skipped += 1
            continue
        part = d / (f["path"].replace("/", "__") + ".part")
        try:
            download(hf_base, repo, rev, f["path"], part, f["size"])
            got = mirrorlib.sha256_file(part)
            if f["sha256"] and got != f["sha256"]:
                part.unlink(missing_ok=True)   # a corrupt download is not resumed
                raise ValueError(f"{f['path']}: sha256 of the download is {got}, expected {f['sha256']}; not uploaded")
            put(mirror, repo, rev, f["path"], part)
            files[f["path"]] = {"size": f["size"], "sha256": got, "pinned": f["pinned"],
                                "source": "pin" if f["pinned"] else ("lfs-oid" if f["sha256"] else "computed")}
            write_manifest(mirror, repo, rev, files, False, d)
            part.unlink(missing_ok=True)
            staged += 1
        except (OSError, ValueError) as e:
            errors.append(str(e))
    if not errors:
        write_manifest(mirror, repo, rev, files, True, d)
    return staged, skipped, errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workload", nargs="*")
    ap.add_argument("--tier", choices=["A", "B", "2b", "big"])
    ap.add_argument("--mirror", default=os.environ.get("PW_MODEL_MIRROR"), required=False)
    ap.add_argument("--hf-base", default=HUB)
    ap.add_argument("--scratch", default=os.path.expanduser("~/.cache/pw-stage"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
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
            continue
        if a.dry_run or done:
            continue
        os.makedirs(a.scratch, exist_ok=True)
        staged, skipped, errors = stage_one(a.mirror, repo, rev, plan, a.hf_base, a.scratch)
        print(f"  staged {staged}, already there {skipped}" + (f", {len(errors)} FAILED" if errors else ""))
        for err in errors:
            print(f"  ERROR {err}", file=sys.stderr)
        bad += bool(errors)
    print(f"total {total / 2**30:.1f} GiB in {len(g)} model(s); {todo / 2**30:.1f} GiB not yet staged")
    if not a.check and not a.dry_run:
        print(f"consumers: export PW_MODEL_MIRROR={a.mirror}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
