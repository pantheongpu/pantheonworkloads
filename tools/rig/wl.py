"""Workload facts the rig tools need (shared by stage_weights.py and batch.py): manifests, `requires`, tiers, pins.

Tiers (docs/rigs.md), all computed from the manifests, so they cannot drift from them:
  A    requires 1 GPU of at most 22 GiB                       (stage 2a, tier A: one A10G / L4 class card)
  B    requires 1 GPU of more than 22 and at most 44 GiB      (stage 2a, tier B: one L40S)
  2b   everything else that fits 8 x 80 GiB (gpus <= 8, gpu_memory_gb <= 80) and is not yet graduated
       (notes do not start with VERIFIED ON / BLOCKED)        (stage 2b: one 8 x A100 80 GB node)
  big  needs more than 8 x 80 GiB (or a vendor/capability this does not model)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

WORKLOADS = ROOT / "workloads"
PIN_HEADER = re.compile(r"^#\s*(\S+/\S+)\s+@\s+([0-9a-f]{7,40})\b")


def names():
    return sorted(p.parent.name for p in WORKLOADS.glob("*/manifest.yaml") if not p.parent.name.startswith("_"))


def manifest(name):
    return validate.load(WORKLOADS / name / "manifest.yaml")


def requires(name):
    return validate.requirements(manifest(name))


def graduated(m):
    notes = str(m.get("notes") or "")
    return notes.startswith("VERIFIED ON") or notes.startswith("BLOCKED")


def tier(name):
    m = manifest(name)
    if "gpu" not in (m.get("targets") or []):
        return None
    req = validate.requirements(m)
    gpus = (req or {}).get("gpus", 1)
    mem = (req or {}).get("gpu_memory_gb")
    if req and gpus == 1 and mem is not None and mem <= 22:
        return "A"
    if req and gpus == 1 and mem is not None and mem <= 44:
        return "B"
    if req and (gpus > 8 or (mem or 0) > 80 or req.get("vendor") == "amd"):
        return "big"
    if req and not graduated(m):
        return "2b"
    return None


def in_tier(tier_name):
    return [n for n in names() if tier(n) == tier_name]


def read_pin(name):
    """(repo, revision, {path: sha256}) from the workload's model.sha256 header comment ('# <repo> @ <commit> ...') and
    its lines; (None, None, {}) when the workload has no model.sha256."""
    f = WORKLOADS / name / "model.sha256"
    if not f.is_file():
        return None, None, {}
    repo = rev = None
    sums = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            m = PIN_HEADER.match(line)
            if m and repo is None:
                repo, rev = m.group(1), m.group(2)
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:   # older single-file pins are a bare hash: no path to stage
            sums[parts[1].strip()] = parts[0]
    return repo, rev, sums
