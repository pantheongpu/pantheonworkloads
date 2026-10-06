#!/usr/bin/env python3
"""Turn `llama-bench -o json` output into the metrics a workload reports.

    python3 bench_parse.py < llama-bench.json   -> {"pp512_tokens_per_s": ..., "tg128_tokens_per_s": ...}

llama-bench prints a JSON list with one object per test: n_prompt > 0 and n_gen == 0 is prompt
processing, n_gen > 0 and n_prompt == 0 is token generation, and avg_ts is tokens/s (mean over
its repetitions). A prompt-processing test of N tokens becomes `ppN_tokens_per_s`, a generation
test of N tokens `tgN_tokens_per_s`. Tests with a context depth (n_depth > 0) get a `_dD` suffix.
"""
import json
import sys


def metrics(rows):
    out = {}
    for r in rows:
        n_prompt, n_gen = int(r.get("n_prompt", 0)), int(r.get("n_gen", 0))
        if n_prompt and not n_gen:
            key = f"pp{n_prompt}"
        elif n_gen and not n_prompt:
            key = f"tg{n_gen}"
        else:
            key = f"pp{n_prompt}_tg{n_gen}"       # a combined test: reported as one rate
        depth = int(r.get("n_depth", 0) or 0)
        if depth:
            key += f"_d{depth}"
        out[key + "_tokens_per_s"] = round(float(r["avg_ts"]), 2)
    if not out:
        raise ValueError("llama-bench reported no tests")
    return out


def parse(text):
    start = text.find("[")                         # tolerate log lines before the JSON list
    if start < 0:
        raise ValueError("no JSON list in llama-bench output")
    rows, _ = json.JSONDecoder().raw_decode(text[start:])
    return metrics(rows)


if __name__ == "__main__":
    print(json.dumps(parse(sys.stdin.read())))
