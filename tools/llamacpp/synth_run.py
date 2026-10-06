#!/usr/bin/env python3
"""Run one synthetic-GGUF functional workload (driven by workloads/llamacpp-synth-*/run.sh).

    synth_run.py --arch llama --quants F16,Q8_0 --bin DIR --cache DIR [--threads 2] [--ngl 0]

For every quantisation type: build (and cache) the random-weight model, greedy-decode with pw-probe
(exact token ids, logit statistics per step) and run llama-perplexity on a fixed generated text.
Prints the workload contract's JSON object on the last line. The models are random-weight models for
kernel and graph coverage, not any named pretrained model (docs/llamacpp-synth.md).

Environment: PW_LC_ENV is a newline-separated list of KEY=VALUE pairs (the simulated-GPU preloads from
tools/llamacpp/common.sh); it is applied to pw-probe and llama-perplexity only, never to the generator
or llama-quantize.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth_gguf  # noqa: E402

PROMPT = "the water is"
N_TOKENS = 6
PPL_CTX = 128
PPL_CHUNKS = 8
# Types the pinned llama-quantize refuses without an importance matrix; they get the generator's synthetic one.
NEEDS_IMATRIX = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "Q2_K_S"}


class SynthError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_dir(cache, arch, size, seed):
    return pathlib.Path(cache) / "synth" / f"{arch}-{size}-seed{seed}-gen{synth_gguf.GENERATOR_VERSION}"


def ensure_f16(cache, arch, size, seed, vocab=None):
    d = model_dir(cache, arch, size, seed)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{arch}-{size}-F16.gguf"
    if not out.exists():
        tmp = d / f"{arch}-{size}-F16.gguf.part"
        synth_gguf.write_model(str(tmp), arch, seed, size, vocab)
        tmp.rename(out)
    return out


def ensure_quant(cache, arch, size, seed, quant, quantize, threads=2, vocab=None):
    f16 = ensure_f16(cache, arch, size, seed, vocab)
    if quant == "F16":
        return f16
    out = f16.with_name(f"{arch}-{size}-{quant}.gguf")
    if not out.exists():
        tmp = f16.with_name(f"{arch}-{size}-{quant}.gguf.part")
        tmp.unlink(missing_ok=True)
        cmd = [quantize, str(f16), str(tmp), quant, str(threads)]
        if quant in NEEDS_IMATRIX:
            im = f16.with_name("imatrix.gguf")
            if not im.exists():
                synth_gguf.write_imatrix(str(im) + ".part", arch, seed, size, vocab)
                os.replace(str(im) + ".part", im)
            # llama-quantize stores the imatrix path in the output's metadata: use a relative one (run from the
            # model directory) so the same inputs give the same bytes wherever the cache lives
            cmd[1:1] = ["--imatrix", im.name]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(f16.parent))
        if p.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            raise SynthError(f"llama-quantize {quant} failed: " + (p.stderr or p.stdout).strip().splitlines()[-1])
        tmp.rename(out)
    return out


def lc_env():
    env = dict(os.environ)
    for line in os.environ.get("PW_LC_ENV", "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def run_probe(probe, model, threads, ngl, extra=(), prompt=PROMPT, n=N_TOKENS):
    cmd = [probe, "-m", str(model), "-p", prompt, "-n", str(n), "-t", str(threads), "-ngl", str(ngl), "-c", "256", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, env=lc_env())
    if p.returncode != 0:
        raise SynthError(f"pw-probe failed on {model.name}: " + (p.stderr.strip().splitlines() or ["?"])[-1])
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        raise SynthError(f"pw-probe printed no JSON for {model.name}: {e}")


def run_ppl(perplexity, model, text_file, threads, ngl, extra=()):
    cmd = [perplexity, "-m", str(model), "-f", str(text_file), "-c", str(PPL_CTX), "-b", str(PPL_CTX),
           "--chunks", str(PPL_CHUNKS), "-t", str(threads), "-ngl", str(ngl), *extra]
    # llama.cpp's asynchronous logger sometimes drops the last lines when the process exits under load
    # (exit status 0, no "Final estimate"): the run is deterministic, so just run it again.
    for _ in range(4):
        p = subprocess.run(cmd, capture_output=True, text=True, env=lc_env())
        m = re.search(r"Final estimate: PPL = ([0-9.]+(?:e[+-]?\d+)?)", p.stdout + p.stderr)
        if p.returncode != 0 or m:
            break
    if p.returncode != 0 or not m:
        raise SynthError(f"llama-perplexity failed (exit {p.returncode}) on {model.name}: "
                         + ((p.stdout + p.stderr).strip().splitlines() or ["?"])[-1])
    return float(m.group(1))


def measure(probe, perplexity, model, text_file, threads, ngl, probe_extra=(), ppl_extra=(), prompt=PROMPT):
    """The flat result dict for one model file."""
    d = run_probe(probe, model, threads, ngl, probe_extra, prompt)
    steps = d["steps"]
    return {
        "tokens": " ".join(str(t) for t in d["generated"]),
        "top1_logit": [s["top1_logit"] for s in steps],
        "logsumexp": [s["logsumexp"] for s in steps],
        "ppl": run_ppl(perplexity, model, text_file, threads, ngl, ppl_extra),
    }, [s["margin"] for s in steps]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--quants", required=True, help="comma-separated llama-quantize type names (F16 is the base file)")
    ap.add_argument("--bin", required=True, help="directory with pw-probe, llama-perplexity, llama-quantize")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--seed", type=int, default=synth_gguf.DEFAULT_SEED)
    ap.add_argument("--prompt", default=PROMPT, help="prompt text (tokenised by the model's own vocabulary)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--expected-shas", help="file of '<arch>-<quant> <sha256>' lines the generated models must match")
    ap.add_argument("--prepare-only", action="store_true", help="build the (first) model and print its path")
    ap.add_argument("--show-margins", action="store_true", help="put the logit margins in the detail text")
    a = ap.parse_args(argv)
    binp = pathlib.Path(a.bin)
    probe, perplexity, quantize = (str(binp / n) for n in ("pw-probe", "llama-perplexity", "llama-quantize"))
    expected = {}
    if a.expected_shas and os.path.exists(a.expected_shas):
        for line in open(a.expected_shas, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                k, v = line.split()[:2]
                expected[k] = v
    cache = pathlib.Path(a.cache)
    text_file = cache / "synth" / f"text-seed{a.seed}-gen{synth_gguf.GENERATOR_VERSION}.txt"
    text_file.parent.mkdir(parents=True, exist_ok=True)
    if not text_file.exists():
        text_file.write_text(synth_gguf.synth_text(a.seed), encoding="utf-8", newline="\n")
    out, margins, shas = {}, {}, []
    if a.prepare_only:
        try:
            q = a.quants.split(",")[0]
            print(ensure_quant(cache, a.arch, a.size, a.seed, q, quantize, max(a.threads, 1)))
        except SynthError as e:
            print(f"synth: {e}", file=sys.stderr)
            return 1
        return 0
    try:
        for q in a.quants.split(","):
            model = ensure_quant(cache, a.arch, a.size, a.seed, q, quantize, max(a.threads, 1))
            digest = sha256(model)
            key = f"{a.arch}-{a.size}-{q}"
            if key in expected and expected[key] != digest:
                print(f"sha256 of {model} is {digest}, expected {expected[key]}: the generator or quantizer "
                      f"produced a different file", file=sys.stderr)
                return 1
            shas.append(f"{key} {digest}")
            res, mg = measure(probe, perplexity, model, text_file, a.threads, a.ngl, prompt=a.prompt)
            for k, v in res.items():
                out[f"{q}.{k}"] = v
            margins[q] = min(mg)
    except SynthError as e:
        print(f"synth: {e}", file=sys.stderr)
        return 1
    sys.stderr.write("\n".join(shas) + "\n")
    detail = f"synthetic random-weight {a.arch} ({a.size}), seed {a.seed}, " + "/".join(a.quants.split(","))
    if a.show_margins:
        detail += "; min margins " + ", ".join(f"{q}={m:.3g}" for q, m in margins.items())
    print(json.dumps({"output": out, "detail": detail, "metrics": {}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
