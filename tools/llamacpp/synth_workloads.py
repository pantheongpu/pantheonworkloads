#!/usr/bin/env python3
"""Single source of truth for the llamacpp-synth-* workloads: their manifests, run.sh and models.sha256.

    synth_workloads.py generate           write workloads/llamacpp-synth-*/ from the tables below and synth_spreads.json
    synth_workloads.py check              fail when the files on disk differ from what `generate` would write
    synth_workloads.py tune --bin DIR     measure prompts, CPU spreads and model hashes (needs the synth tools build)
                                          and rewrite tools/llamacpp/synth_spreads.json

The models are RANDOM-WEIGHT models for kernel and graph coverage, not any named pretrained model.
Tolerances: per quantisation type, max(class floor, 5 x the largest CPU spread measured by `tune`),
rounded up to two significant figures (docs/llamacpp-synth.md, "Tolerances").
"""
import argparse
import json
import math
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import synth_gguf  # noqa: E402

SPREADS = HERE / "synth_spreads.json"
TARGETS_FUNCTIONAL = ["cpu", "gpu", "sim:nvidia/*", "sim:amd/*"]
TARGETS_BENCH = ["cpu", "gpu"]
RUNTIME_VERSION = "llama.cpp b11447 (da263e7275dfbaeefcd61504eaa4fd5247540e11)"
SOURCE_URL = "https://github.com/pantheongpu/pantheonworkloads/blob/main/tools/llamacpp/synth_gguf.py"
SAFETY = 5.0

DISCLAIMER = ("These are RANDOM-WEIGHT models for kernel and graph coverage: seeded pseudo-random numbers in the "
              "tensor shapes of an architecture. They are NOT the quality or behaviour of any named pretrained model, "
              "and nothing here says how Llama, Mistral, Mixtral, Qwen2, Gemma or Phi-3 models perform.")

# quantisation type -> (class, what a CUDA/HIP backend runs for it)
QUANT_CLASS = {
    "F32": "float", "F16": "float", "BF16": "float",
    "Q8_0": "q8", "Q6_K": "q8", "Q5_0": "q8", "Q5_1": "q8", "Q5_K_S": "q8", "Q5_K_M": "q8",
    "Q4_0": "q4", "Q4_1": "q4", "Q4_K_S": "q4", "Q4_K_M": "q4", "IQ4_NL": "q4", "IQ4_XS": "q4", "MXFP4_MOE": "q4",
    "Q3_K_S": "low", "Q3_K_M": "low", "Q3_K_L": "low", "Q2_K": "low", "Q2_K_S": "low", "IQ3_S": "low", "IQ3_M": "low",
    "IQ3_XXS": "low", "IQ3_XS": "low", "IQ2_XXS": "low", "IQ2_XS": "low", "IQ2_S": "low", "IQ2_M": "low",
    "IQ1_S": "low", "IQ1_M": "low", "TQ1_0": "low", "TQ2_0": "low", "Q1_0": "low", "Q2_0": "low",
}
# floors: (absolute logit tolerance, relative perplexity tolerance). Reasoned, not measured: a GPU backend
# quantises the activations differently from the CPU one (q8_1 blocks, different block groupings) and may
# accumulate in a different order, so the CPU-only spreads are a lower bound.
FLOORS = {"float": (0.05, 0.005), "q8": (0.15, 0.01), "q4": (0.3, 0.02), "low": (0.5, 0.05)}

ARCH_QUANTS = ["F16", "Q8_0", "Q4_0", "Q4_K_M"]
ARCH_NOTES = {
    "llama": "Llama-architecture graph: RMSNorm, multi-head attention (8 heads, no grouped-query sharing), RoPE base 1e4, SwiGLU, untied output matrix.",
    "mistral": "llama.cpp's `llama` architecture with Mistral-style shapes: grouped-query attention (8 heads, 2 KV heads), RoPE base 1e6, SwiGLU.",
    "mixtral": "llama.cpp's `llama` architecture with Mixtral-style shapes: 8 experts, 2 used per token, grouped-query attention (8 heads, 2 KV heads), RoPE base 1e6. Exercises ffn_gate_inp routing and the expert matmuls (MUL_MAT_ID: mmq/mmvq with an expert index).",
    "qwen2": "Qwen2 architecture: grouped-query attention (8 heads, 2 KV heads) with biases on Q, K and V, RoPE base 1e6, SwiGLU, byte-level BPE tokenizer (qwen2 pre-tokenizer) built from a generated vocabulary.",
    "gemma": "Gemma (v1) architecture: multi-query attention (8 heads, 1 KV head), embeddings scaled by sqrt(n_embd), GeGLU, output matrix tied to the token embeddings.",
    "phi3": "Phi-3 architecture: fused QKV projection (attn_qkv) and a fused gate+up projection (ffn_up with 2*n_ff rows), SwiGLU, RoPE base 1e4.",
}
FUNCTIONAL = [dict(name=f"llamacpp-synth-{a}", arch=a, quants=ARCH_QUANTS, kind="arch") for a in
              ("llama", "mistral", "mixtral", "qwen2", "gemma", "phi3")]
GROUPS = [
    ("legacy", "llama", ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"],
     "legacy block quantisations (32-weight blocks): CUDA/HIP mmvq for single-token decode, mmq for prompt processing, dequantise-to-cuBLAS fallbacks"),
    ("kquant", "llama", ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"],
     "k-quants (256-weight super-blocks, mixed per tensor by llama-quantize): mmvq and mmq kernels per super-block type"),
    ("iquant", "llama", ["IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS"],
     "importance-quant family that needs no importance matrix: lookup-table dequantisation (IQ4 non-linear codebooks, IQ3 grids)"),
    ("iquant-imatrix", "llama", ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "Q2_K_S"],
     "the low-bit importance-quant types that llama-quantize only writes with an importance matrix (a seeded synthetic one is generated): grid lookup kernels"),
    ("float", "llama", ["F32", "F16", "BF16"],
     "unquantised weights: cuBLAS/rocBLAS GEMM and the float mat-vec kernels, F32 and BF16 conversion paths"),
    ("lowbit", "llama", ["Q1_0", "Q2_0", "TQ1_0", "TQ2_0"],
     "1- and 2-bit formats: Q1_0 and Q2_0 block quantisations and the ternary TQ1_0/TQ2_0 types (several have no CUDA/HIP mmq kernel and use the generic dequantise path)"),
    ("moe", "mixtral", ["Q4_0", "Q8_0", "Q4_K_M", "Q6_K", "IQ4_XS", "MXFP4_MOE"],
     "quantised expert matmuls (MUL_MAT_ID) for each block family, plus MXFP4 experts"),
]
FUNCTIONAL += [dict(name=f"llamacpp-synth-quant-{g}", arch=arch, quants=q, kind="group", covers=c) for g, arch, q, c in GROUPS]
BENCH = [dict(name=f"llamacpp-synth-bench-{a}", arch=a) for a in ("llama", "mistral", "mixtral", "qwen2", "gemma", "phi3")]


def round_up(x, sig=2):
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - sig + 1
    return math.ceil(x / 10 ** e) * 10 ** e


def tolerances(spread):
    """{field key: {abs, rel}} for one workload's spread record."""
    fields = {}
    for q, rec in spread["quants"].items():
        floor_abs, floor_ppl = FLOORS[QUANT_CLASS[q]]
        a = max(floor_abs, round_up(SAFETY * max(rec["top1_logit"], rec["logsumexp"])))
        r = max(floor_ppl, round_up(SAFETY * rec["ppl_rel"]))
        fields[f"{q}.top1_logit"] = {"abs": float(f"{a:.6g}"), "rel": 0.0}
        fields[f"{q}.logsumexp"] = {"abs": float(f"{a:.6g}"), "rel": 0.0}
        fields[f"{q}.ppl"] = {"abs": 0.0, "rel": float(f"{r:.6g}")}
    return fields


def _literal(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|" if "\n" in data else None)


class Dumper(yaml.SafeDumper):
    pass


Dumper.add_representer(str, _literal)


def functional_manifest(w, spread):
    arch, quants = w["arch"], w["quants"]
    what = ARCH_NOTES[arch] if w["kind"] == "arch" else \
        f"{ARCH_NOTES[arch].split(':')[0]} model quantised to {', '.join(quants)}: {w['covers']}."
    notes = (
        f"{DISCLAIMER}\n"
        f"Generated by tools/llamacpp/synth_gguf.py (seed {synth_gguf.DEFAULT_SEED}, size tiny, {synth_gguf.DEFAULT_VOCAB}-token generated "
        f"vocabulary; same seed gives byte-identical files, pinned in models.sha256), quantised with the pinned build's llama-quantize.\n"
        f"{what}\n"
        f"Per quantisation type the output holds the 6 greedy token ids (compared exactly), the top-1 logit and log-sum-exp "
        f"of every step and llama-perplexity's PPL on a fixed generated text. Reference recorded on the cpu target. "
        f"Tolerances are max(class floor, 5 x CPU spread across threads/ubatch/flash-attention/KV type/repacking); "
        f"a GPU backend's spread was not measurable here. See docs/llamacpp-synth.md.\n"
        f"Backends and sim: wiring as llamacpp-smollm2-135m (docs/llamacpp.md); the sim: targets have never been run.")
    return {
        "name": w["name"],
        "description": f"llama.cpp on seeded random-weight {arch} GGUFs ({', '.join(quants)}): greedy token ids, logits, perplexity (not a pretrained model).",
        "kind": "functional",
        "runtime": "llama.cpp",
        "runtime_version": RUNTIME_VERSION,
        "targets": TARGETS_FUNCTIONAL,
        "compare": "tolerance",
        "tolerance": {"abs": 0.0, "rel": 0.0, "fields": tolerances(spread)},
        "timeout_s": 3600,
        "model": model_block(arch, "tiny", ", ".join(quants)),
        "notes": notes,
    }


def model_block(arch, size, quant):
    return {
        "id": f"pantheonworkloads synthetic random-weight {arch} GGUF ({size}; {quant}; NOT a pretrained model)",
        "revision": f"synth-gen{synth_gguf.GENERATOR_VERSION}-seed{synth_gguf.DEFAULT_SEED}",
        "licence": "Apache-2.0 (generated by this repository's generator from seeded random numbers; no third-party weights, vocabulary or text)",
        "source_url": SOURCE_URL,
    }


def bench_manifest(w):
    arch = w["arch"]
    notes = (
        f"{DISCLAIMER}\n"
        f"Real targets only: bin/pw refuses --bench for sim: targets. llama-bench -o json on a seeded random-weight {arch} GGUF "
        f"made by tools/llamacpp/synth_gguf.py and quantised with llama-quantize; metrics pp<N>_tokens_per_s and tg<M>_tokens_per_s "
        f"(default pp512, tg128). The recorded model is the default size m (see `python3 tools/llamacpp/synth_gguf.py --list`) "
        f"at Q4_K_M; PW_SYNTH_SIZE (tiny, s, m, l, xl) and PW_SYNTH_QUANT select others, but a record always carries "
        f"this manifest's model pointer, so only record with the defaults. The larger sizes pad the vocabulary to 32000 so the "
        f"output layer is realistic. {ARCH_NOTES[arch]}\n"
        f"NOT YET RUN on any GPU: bench/ has no record for it. AMD devices are not auto-detected: pass --device. "
        f"See docs/llamacpp-synth.md and docs/llamacpp.md.")
    return {
        "name": w["name"],
        "description": f"llama-bench pp/tg tokens/s on a seeded random-weight {arch} GGUF, size m, Q4_K_M by default (not a pretrained model).",
        "kind": "benchmark",
        "runtime": "llama.cpp",
        "runtime_version": RUNTIME_VERSION,
        "targets": TARGETS_BENCH,
        "timeout_s": 7200,
        "model": model_block(arch, "m", "Q4_K_M"),
        "notes": notes,
    }


def functional_run(w, spread):
    return f"""#!/usr/bin/env bash
# {w['name']}: random-weight {w['arch']} GGUFs ({', '.join(w['quants'])}) for kernel and graph coverage.
# NOT a pretrained model. Generated by tools/llamacpp/synth_workloads.py: edit that, not this file.
# Contract: docs/workload-contract.md. Backends, build and knobs: docs/llamacpp.md, docs/llamacpp-synth.md.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
. "$here/../../tools/llamacpp/synth.sh"
lc_setup
lc_synth_prepare
lc_synth_functional {w['arch']} {','.join(w['quants'])} '{spread['prompt']}' "$here/models.sha256"
"""


def bench_run(w):
    return f"""#!/usr/bin/env bash
# {w['name']}: llama-bench on a random-weight {w['arch']} GGUF. NOT a pretrained model. Real GPUs only:
#   bin/pw run {w['name']} --target gpu --bench --repeat 3 [--device NAME]
# Generated by tools/llamacpp/synth_workloads.py: edit that, not this file.
#   PW_SYNTH_SIZE   tiny | s | m (default) | l | xl      PW_SYNTH_QUANT   quantisation type (default Q4_K_M)
#   PW_BENCH_PP / PW_BENCH_TG / PW_BENCH_REPS / PW_BENCH_EXTRA / PW_NGL / PW_THREADS: as llamacpp-bench-smollm2-135m
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/../../tools/llamacpp/common.sh"
. "$here/../../tools/llamacpp/synth.sh"
lc_setup
lc_synth_prepare
lc_synth_bench {w['arch']}
"""


def render(spreads):
    """{relative path: text} for every generated file."""
    files = {}
    for w in FUNCTIONAL:
        s = spreads[w["name"]]
        d = f"workloads/{w['name']}"
        files[f"{d}/manifest.yaml"] = yaml.dump(functional_manifest(w, s), Dumper=Dumper, sort_keys=False, width=110)
        files[f"{d}/run.sh"] = functional_run(w, s)
        files[f"{d}/models.sha256"] = "# sha256 of each generated model file: <arch>-<size>-<quant> <hash> (tools/llamacpp/synth_workloads.py tune)\n" + \
            "".join(f"{w['arch']}-tiny-{q} {s['quants'][q]['sha256']}\n" for q in w["quants"])
    for w in BENCH:
        d = f"workloads/{w['name']}"
        files[f"{d}/manifest.yaml"] = yaml.dump(bench_manifest(w), Dumper=Dumper, sort_keys=False, width=110)
        files[f"{d}/run.sh"] = bench_run(w)
    return files


def load_spreads():
    return json.loads(SPREADS.read_text())


def cmd_generate(_):
    for rel, text in render(load_spreads()).items():
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        if rel.endswith("run.sh"):
            p.chmod(0o755)
    print(f"wrote {len(render(load_spreads()))} files")
    return 0


def cmd_check(_):
    bad = [rel for rel, text in render(load_spreads()).items()
           if not (ROOT / rel).exists() or (ROOT / rel).read_text(encoding="utf-8") != text]
    for rel in bad:
        print(f"differs from the generator: {rel}", file=sys.stderr)
    return 1 if bad else 0


def cmd_tune(a):
    import synth_tune
    out = json.loads(SPREADS.read_text()) if SPREADS.exists() else {}
    for w in FUNCTIONAL:
        if a.only and a.only not in w["name"]:
            continue
        print(f"== {w['name']}", file=sys.stderr)
        out[w["name"]] = synth_tune.tune(a.bin, a.cache, w["arch"], w["quants"], candidates=a.candidates)
        SPREADS.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=cmd_generate)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    t = sub.add_parser("tune")
    t.add_argument("--bin", required=True)
    t.add_argument("--cache", required=True)
    t.add_argument("--candidates", type=int, default=80)
    t.add_argument("--only", help="only workloads whose name contains this")
    t.set_defaults(fn=cmd_tune)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
