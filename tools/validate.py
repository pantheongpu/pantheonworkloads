"""Manifest checks for workloads/<name>/manifest.yaml (rules: docs/manifest.md)."""
import pathlib
import re

import yaml

KINDS = {"functional", "benchmark"}
COMPARE = {"exact", "tolerance"}
NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
TARGET = re.compile(r"^(cpu|gpu|sim:(nvidia|amd|\*)/(\*|[a-z0-9-]+))$")


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_restricted(manifest):
    """True for a workload whose model is under a non-permissive or gated licence (`restricted: true`)."""
    return isinstance(manifest, dict) and manifest.get("restricted") is True


REQUIRES_KEYS = {"gpus", "gpu_memory_gb", "min_compute_capability", "vendor", "interconnect", "notes"}
VENDORS = {"nvidia", "amd"}
CAPABILITY = re.compile(r"^\d+\.\d+$")


def requirements(manifest):
    """The normalised `requires` of a manifest: {gpus, gpu_memory_gb, min_compute_capability, vendor, ...}
    with gpus defaulting to 1, or None when the manifest declares none (any GPU will do)."""
    req = manifest.get("requires") if isinstance(manifest, dict) else None
    if not isinstance(req, dict):
        return None
    out = dict(req)
    out.setdefault("gpus", 1)
    return out


def needs_text(req):
    """Short form for tables: '8 x 80 GiB', '1 x 24 GiB', '2 x any' (no memory requirement); 'any' for no requires."""
    if not req:
        return "any"
    mem = req.get("gpu_memory_gb")
    return f"{req.get('gpus', 1)} x " + (f"{mem:g} GiB" if mem is not None else "any")


def check_requires(req):
    """Errors for a `requires` value (rules: docs/manifest.md)."""
    if not isinstance(req, dict):
        return ["requires must be a mapping"]
    errors = [f"requires.{k}: unknown key (known: {sorted(REQUIRES_KEYS)})" for k in req if k not in REQUIRES_KEYS]
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if "gpus" in req and not (isinstance(req["gpus"], int) and not isinstance(req["gpus"], bool) and req["gpus"] >= 1):
        errors.append("requires.gpus must be an integer >= 1")
    if "gpu_memory_gb" in req and not (num(req["gpu_memory_gb"]) and req["gpu_memory_gb"] > 0):
        errors.append("requires.gpu_memory_gb must be a positive number (GiB per GPU)")
    if "min_compute_capability" in req:
        c = req["min_compute_capability"]
        if not (isinstance(c, str) and CAPABILITY.match(c)):
            errors.append('requires.min_compute_capability must be a string "major.minor", e.g. "9.0"')
        elif req.get("vendor") == "amd":
            errors.append("requires.min_compute_capability is NVIDIA only (vendor: amd conflicts)")
    if "vendor" in req and req["vendor"] not in VENDORS:
        errors.append(f"requires.vendor {req['vendor']!r}: one of {sorted(VENDORS)}")
    for k in ("interconnect", "notes"):
        if k in req and not (isinstance(req[k], str) and req[k].strip()):
            errors.append(f"requires.{k} must be a non-empty string")
    return errors


def check(path):
    """Return (errors, warnings) for one manifest file."""
    path = pathlib.Path(path)
    errors, warnings = [], []
    try:
        m = load(path)
    except yaml.YAMLError as e:
        return [f"not valid YAML: {e}"], []
    if not isinstance(m, dict):
        return ["the manifest must be a mapping"], []

    for key in ("name", "description", "kind", "runtime", "targets"):
        if key not in m:
            errors.append(f"missing required key '{key}'")
    name = m.get("name")
    if name is not None:
        if not isinstance(name, str) or not NAME.match(name):
            errors.append(f"name {name!r}: lower-case letters, digits and '-' only")
        elif name != path.parent.name:
            errors.append(f"name {name!r} does not match its directory {path.parent.name!r}")
    if m.get("kind") not in KINDS and "kind" in m:
        errors.append(f"kind {m.get('kind')!r}: one of {sorted(KINDS)}")

    targets = m.get("targets")
    if targets is not None:
        if not isinstance(targets, list) or not targets:
            errors.append("targets must be a non-empty list")
        else:
            for t in targets:
                if not isinstance(t, str) or not TARGET.match(t):
                    errors.append(f"target {t!r}: cpu, gpu, or sim:<nvidia|amd>/<profile or *>")

    if "restricted" in m:
        if not isinstance(m["restricted"], bool):
            errors.append("restricted must be true or false")
        elif m["restricted"] and not isinstance(m.get("model"), dict):
            errors.append("restricted: true needs a 'model' (the restriction is the model's licence)")
        elif m["restricted"] and not m.get("notes"):
            errors.append("restricted: true needs 'notes' saying which licence or gate applies and what was read")

    if "requires" in m:
        errors += check_requires(m["requires"])

    model = m.get("model")
    # `model: null` written out says "this workload downloads no model" (a pure-compute suite on a runtime).
    if m.get("runtime") not in (None, "none") and model is None and "model" not in m:
        errors.append("a workload with a runtime other than 'none' needs a 'model' (or set runtime: none, or model: null for a model-free suite)")
    if model is not None:
        if not isinstance(model, dict):
            errors.append("model must be a mapping")
        else:
            for key in ("id", "licence", "source_url"):
                if not model.get(key):
                    errors.append(f"model.{key} is required")
            if str(model.get("licence", "")).upper().startswith("UNVERIFIED"):
                warnings.append("model.licence is marked UNVERIFIED: read the model's own card before relying on it")
            if "revision" not in model:
                errors.append("model.revision is required (null while unpinned)")
            elif model["revision"] is None:
                warnings.append("model.revision is null: results are not reproducible until it is pinned")

    if m.get("kind") == "functional":
        if m.get("compare") not in COMPARE:
            errors.append(f"a functional workload needs compare: one of {sorted(COMPARE)}")
        if not (path.parent / "reference.json").exists():
            warnings.append("no reference.json recorded yet (bin/pw record): runs will be reported as SKIP")
    if not (path.parent / "run.sh").exists():
        errors.append("run.sh is missing")
    t = m.get("timeout_s", 600)
    if not isinstance(t, int) or t <= 0:
        errors.append("timeout_s must be a positive integer")
    return errors, warnings
