#!/usr/bin/env python3
"""Turns whisper-bench's log (stderr, read from stdin) into pw metrics JSON on stdout.

whisper.cpp prints lines such as
    whisper_print_timings:   encode time =  1234.56 ms /     3 runs ( 411.52 ms per run)
Only the encoder and decoder per-run times are kept. Exit 1 when they are missing.
"""
import json
import re
import sys

LINE = re.compile(r"(encode|decode|batchd|prompt) time\s*=\s*([\d.]+) ms\s*/\s*(\d+) runs\s*\(\s*([\d.]+) ms per run\)")


def parse(text):
    found = {}
    for m in LINE.finditer(text):
        kind, _total, runs, per_run = m.groups()
        if int(runs) > 0:
            found[kind] = float(per_run)
    return {f"{k}_ms_per_run": found[k] for k in ("encode", "decode") if k in found}


if __name__ == "__main__":
    metrics = parse(sys.stdin.read())
    if "encode_ms_per_run" not in metrics:
        sys.exit("no encode timing in whisper-bench output")
    print(json.dumps(metrics))
