#!/usr/bin/env python3
"""Count the GPUs of this host that have at least a given memory: `hwcheck.py <gib> [nvidia|amd]`.

Prints one line, "<count> <description of all GPUs>", for tools/hwcheck.sh (the workloads' own check of what
`requires` in their manifest says; bin/pw makes the same check before it starts a workload). Same detection and
the same 98% allowance as bin/pw: nvidia-smi first, then rocm-smi, then sysfs for AMD. The environment variables
PW_NVIDIA_SMI, PW_ROCM_SMI and PW_DRM_ROOT replace the tools for tests. Exits 2 on a usage error.
"""
import json
import os
import pathlib
import subprocess
import sys

SLACK = 0.98


def nvidia():
    try:
        p = subprocess.run([os.environ.get("PW_NVIDIA_SMI") or "nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if p.returncode != 0:
        return []
    out = []
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            try:
                out.append(("nvidia", parts[0], float(parts[1]) / 1024))
            except ValueError:
                pass
    return out


def amd():
    out = []
    try:
        p = subprocess.run([os.environ.get("PW_ROCM_SMI") or "rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
                           capture_output=True, text=True, timeout=30)
        data = json.loads(p.stdout) if p.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        data = {}
    for key in sorted(k for k in data if k.startswith("card") and isinstance(data[k], dict)):
        card = data[key]
        try:
            out.append(("amd", card.get("Card Series") or card.get("Card series") or "AMD GPU", int(card["VRAM Total Memory (B)"]) / 2**30))
        except (KeyError, ValueError, TypeError):
            pass
    if out:
        return out
    root = pathlib.Path(os.environ.get("PW_DRM_ROOT") or "/sys/class/drm")
    for dev in sorted(root.glob("card[0-9]*/device")):
        try:
            if (dev / "vendor").read_text().strip() == "0x1002":
                out.append(("amd", "AMD GPU", int((dev / "mem_info_vram_total").read_text()) / 2**30))
        except (OSError, ValueError):
            pass
    return out


def main(argv):
    if len(argv) not in (2, 3) or (len(argv) == 3 and argv[2] not in ("nvidia", "amd")):
        print("usage: hwcheck.py <gib> [nvidia|amd]", file=sys.stderr)
        return 2
    gib = float(argv[1])
    vendor = argv[2] if len(argv) == 3 else None
    gpus = nvidia() or amd()
    ok = [g for g in gpus if (not vendor or g[0] == vendor) and g[2] >= gib * SLACK]
    groups = {}
    for _, name, mem in gpus:
        groups[(name, round(mem, 1))] = groups.get((name, round(mem, 1)), 0) + 1
    text = ", ".join(f"{n} x {name} {mem:g} GiB" for (name, mem), n in groups.items()) or "no GPU detected"
    print(f"{len(ok)} {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
