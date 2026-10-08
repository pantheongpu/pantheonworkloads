#!/usr/bin/env python3
"""Probe how this PyTorch converts out-of-range values to float8 (e4m3fn, e5m2) and back, on CPU and CUDA.

    python3 tools/fp8_probe.py > probe.json

Prints one JSON object: torch version, device name, compute capability, and for every
(device, source dtype, fp8 dtype) the round-tripped values of the probe inputs.
NaN / inf are written as the strings "nan", "inf", "-inf" so the output is valid JSON.
Background: docs/fp8-cast-semantics.md. Needs torch only.
"""
import json
import math
import sys

import torch

INPUTS = [448.0, 449.0, 480.0, 500.0, 57344.0, -449.0, math.inf, math.nan, 2.0 ** -10, 1e-30]
FP8 = {"float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}
SRC = {"float32": torch.float32, "float64": torch.float64, "float16": torch.float16, "bfloat16": torch.bfloat16}


def num(v):
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def main():
    out = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "inputs": [num(v) for v in INPUTS], "results": {}}
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if "cuda" in devs:
        out["device"] = torch.cuda.get_device_name(0)
        out["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
    else:
        out["device"], out["capability"] = None, None
    for dev in devs:
        for sname, sdt in SRC.items():
            src = torch.tensor(INPUTS, dtype=torch.float64).to(sdt).to(dev)   # the source dtype may already have rounded the input
            out["results"][f"{dev}/{sname} source"] = [num(v) for v in src.double().cpu().tolist()]
            for fname, fdt in FP8.items():
                try:
                    back = src.to(fdt).to(torch.float64).cpu().tolist()
                    out["results"][f"{dev}/{sname}->{fname}"] = [num(v) for v in back]
                except Exception as e:   # noqa: BLE001
                    out["results"][f"{dev}/{sname}->{fname}"] = "error: %s" % str(e).splitlines()[0]
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
