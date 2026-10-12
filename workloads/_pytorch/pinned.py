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


def _mirror_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "rig", "mirror.py")
    spec = importlib.util.spec_from_file_location("pw_rig_mirror", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pw_rig_mirror"] = mod   # dataclass-free, but a worker thread may import by name
    spec.loader.exec_module(mod)
    return mod


def snapshot(model_id, revision, cache_dir, allow_patterns):
    """huggingface_hub.snapshot_download(model_id, revision, cache_dir, allow_patterns), or, when PW_MODEL_MIRROR is set
    (an s3:// prefix or a directory staged by tools/rig/stage_weights.py), a parallel copy from the mirror, verified against
    the mirror's manifest and against model.sha256. A model the mirror lacks falls back to the Hub (the call raises, so
    hf_load SKIPs, when PW_MODEL_MIRROR_STRICT=1). A sha256 mismatch fails the run. PW_HF_REVISION (an unpinned run) uses the Hub."""
    mirror = os.environ.get("PW_MODEL_MIRROR")
    if mirror and not os.environ.get("PW_HF_REVISION"):
        mod = _mirror_module()
        sums_path = os.path.join(os.environ.get("PW_WORKLOAD_DIR", ""), "model.sha256")
        pins = read_sums(sums_path) if os.path.isfile(sums_path) else {}
        dest = os.path.join(cache_dir, "pw-mirror", model_id.replace("/", "--"), revision)
        try:
            return str(mod.fetch(mirror, model_id, revision, dest, patterns=list(allow_patterns), expected=pins,
                                 jobs=int(os.environ.get("PW_MIRROR_JOBS", "8"))))
        except mod.Miss as e:
            if os.environ.get("PW_MODEL_MIRROR_STRICT") == "1":
                raise RuntimeError(f"not in the model mirror {mirror}: {e}")
            print(f"warning: {model_id}@{revision[:8]} is not in the model mirror ({e}); downloading from the Hub", file=sys.stderr)
        except (mod.Mismatch, OSError) as e:
            sys.exit(f"copy from the model mirror {mirror} failed: {e}")   # SystemExit: a failure, never a SKIP
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, revision=revision, cache_dir=cache_dir, allow_patterns=allow_patterns)


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
