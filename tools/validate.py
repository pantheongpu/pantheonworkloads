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
