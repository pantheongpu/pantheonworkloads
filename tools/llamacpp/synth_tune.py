#!/usr/bin/env python3
"""Choose a robust prompt and measure how much the CPU results move between settings.

    synth_tune.py --bin DIR --cache DIR --arch llama --quants F16,Q8_0,Q4_0 [--candidates 80] [--out spreads.json]

1. Prompt: tries seeded three-word prompts and keeps the one whose smallest top-1/top-2 logit margin
   (over the generated tokens and over every quantisation type given) is largest, so greedy decoding
   does not hinge on a near-tie.
2. Spread: runs the chosen prompt on the CPU backend under settings that change the arithmetic but not the
   model (threads, micro-batch size, flash attention on/off, f32 KV cache, weight repacking off) and records,
   per quantisation type, the largest difference of each reported number from the baseline run, and whether
   the greedy token ids changed. docs/llamacpp-synth.md explains how the manifests' tolerances are derived
   from these numbers; they are a CPU-only lower bound for what a GPU backend will show.
Developer tool: not used by the workloads at run time.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth_gguf  # noqa: E402
import synth_run  # noqa: E402

PROBE_VARIANTS = {
    "threads1": ["-t", "1"], "threads4": ["-t", "4"],
    "ubatch2": ["-ub", "2"],
    "fa_on": ["--fa", "on"], "fa_off": ["--fa", "off"],
    "kv_f32": ["--kv", "f32"], "no_repack": ["--no-repack"],
}
PPL_VARIANTS = {
    "threads1": ["-t", "1"], "threads4": ["-t", "4"],
    "fa_on": ["-fa", "on"], "fa_off": ["-fa", "off"],
    "no_repack": ["--no-repack"],
}


def candidate_prompts(n, seed=1):
    draw = synth_gguf._splitmix64(synth_gguf.stream_key(seed, "prompts") + np.arange(3 * n, dtype=np.uint64))
    idx = [int(x % np.uint64(len(synth_gguf.WORDS))) for x in draw]
    return [" ".join(synth_gguf.WORDS[i] for i in idx[3 * k:3 * k + 3]) for k in range(n)]


def tune(bin_dir, cache, arch, quants, size="tiny", candidates=80, prompt=None):
    """Return the report dict: chosen prompt and, per quantisation type, margin, CPU spreads and sha256."""
    b = pathlib.Path(bin_dir)
    probe, perplexity, quantize = str(b / "pw-probe"), str(b / "llama-perplexity"), str(b / "llama-quantize")
    seed = synth_gguf.DEFAULT_SEED
    models = {q: synth_run.ensure_quant(cache, arch, size, seed, q, quantize) for q in quants}
    text = pathlib.Path(cache) / "synth" / f"text-seed{seed}-gen{synth_gguf.GENERATOR_VERSION}.txt"
    text.parent.mkdir(parents=True, exist_ok=True)
    if not text.exists():
        text.write_text(synth_gguf.synth_text(seed), encoding="utf-8", newline="\n")

    best = (prompt, -1.0)
    if not prompt:
        for p in candidate_prompts(candidates):
            m = min(min(s["margin"] for s in synth_run.run_probe(probe, models[q], 1, 0, prompt=p)["steps"]) for q in quants)
            if m > best[1]:
                best = (p, m)
    prompt = best[0]
    report = {"arch": arch, "size": size, "prompt": prompt, "quants": {}}
    for q in quants:
        base_probe = synth_run.run_probe(probe, models[q], 2, 0, prompt=prompt)
        base_ppl = synth_run.run_ppl(perplexity, models[q], text, 2, 0)
        margin = min(s["margin"] for s in base_probe["steps"])
        spread = {"top1_logit": 0.0, "logsumexp": 0.0, "ppl_rel": 0.0}
        ids_changed = []
        for name, extra in PROBE_VARIANTS.items():
            d = synth_run.run_probe(probe, models[q], 2, 0, extra, prompt)
            if d["generated"] != base_probe["generated"]:
                ids_changed.append(name)
            for key in ("top1_logit", "logsumexp"):
                diff = max(abs(x[key] - y[key]) for x, y in zip(d["steps"], base_probe["steps"]))
                spread[key] = max(spread[key], diff)
        for name, extra in PPL_VARIANTS.items():
            v = synth_run.run_ppl(perplexity, models[q], text, 2, 0, extra)
            spread["ppl_rel"] = max(spread["ppl_rel"], abs(v - base_ppl) / base_ppl)
        report["quants"][q] = {"min_margin": round(margin, 4), "ids_changed_by": ids_changed,
                               "sha256": synth_run.sha256(models[q]), **{k: float(f"{v:.4g}") for k, v in spread.items()}}
        print(f"{arch} {q:8} margin {margin:7.3f}  d_top1 {spread['top1_logit']:.2e}  d_lse {spread['logsumexp']:.2e}  "
              f"d_ppl_rel {spread['ppl_rel']:.2e}  ids changed by {ids_changed or '-'}", file=sys.stderr)
    print(f"prompt: {prompt!r}", file=sys.stderr)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--quants", required=True)
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--candidates", type=int, default=80)
    ap.add_argument("--prompt", help="skip the search and measure this prompt")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    report = tune(a.bin, a.cache, a.arch, a.quants.split(","), a.size, a.candidates, a.prompt)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
