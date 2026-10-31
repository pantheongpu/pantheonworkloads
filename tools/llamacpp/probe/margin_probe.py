#!/usr/bin/env python3
"""Run pw-margin (pw-margin.cpp) on a model with the candidate prompts and print the top-1/top-2 gap table.

  margin_probe.py --bin /path/to/pw-margin --model M.gguf [--prompts prompts.txt] [-n 3] [--min-gap 2.0] [--log probe.log]

Prints, per prompt: output text, the gap at each step, the minimum gap, and the word PIN when output names Paris and the
minimum gap is >= --min-gap. The minimum over the generated steps is what matters: a greedy run flips at its smallest gap.
"""
import argparse, json, subprocess, sys, os

DEFAULT = [
    "The capital of France is",
    "Question: What is the capital of France?\\nAnswer:",
    "Q: What is the capital of France? Answer in one word.\\nA:",
    "The capital of France is Paris. The capital of France is Paris. The capital of France is",
    "Italy: Rome\\nSpain: Madrid\\nGermany: Berlin\\nFrance:",
    "Paris, Paris, Paris, Paris, Paris, Paris,",
    "The Eiffel Tower is in the city of",
    "France's capital city is called",
    "The French capital, home to the Louvre and the Eiffel Tower, is the city of",
    "Capital of Italy: Rome\\nCapital of France:",
    "What is the capital of France? The answer is",
    "Paris is the capital of",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", help="file, one prompt per line, \\n for a newline (default: built-in candidates)")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--log")
    a = ap.parse_args()
    pf = a.prompts
    if not pf:
        pf = "/tmp/pw-margin-prompts.%d.txt" % os.getpid()
        open(pf, "w").write("\n".join(DEFAULT) + "\n")
    p = subprocess.run([a.bin, "-m", a.model, "-f", pf, "-n", str(a.n), "-ngl", str(a.ngl)], capture_output=True, text=True)
    if p.returncode:
        sys.exit("pw-margin failed:\n" + p.stderr[-2000:])
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    if a.log:
        open(a.log, "a").write(p.stdout)
    for r in lines:
        gaps = [s["gap"] for s in r["steps"]]
        mn = min(gaps)
        flag = "PIN" if "Paris" in r["output"] and mn >= a.min_gap else ("low-gap" if mn < a.min_gap else "no-Paris")
        print("%-8s min %6.2f gaps %s output %r  <- %r" % (flag, mn, [round(g, 2) for g in gaps], r["output"], r["prompt"]))

main()
