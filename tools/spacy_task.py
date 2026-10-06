#!/usr/bin/env python3
"""spaCy en_core_web_sm payload of the spacy-* workloads.

    python3 -I tools/spacy_task.py [--bench]

Runs the pretrained pipeline over fixed sentences (written for this repo) and prints, as the last
stdout line, {"output", "detail", "metrics"}: tokens, coarse POS tags, dependency labels, head
offsets and named entities, all compared exactly. Metrics only with --bench on the gpu target.
Exit 77 when the target cannot run it (no GPU, spaCy without a GPU build); the gpu target never
falls back to the CPU.
"""
import json
import os
import statistics
import sys
import time

TEXTS = [
    "Apple is looking at buying a U.K. startup for $1 billion in London on Monday.",
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "Dr. Maria Lopez moved from Madrid to Austin, Texas, in March 2019 to join the University of Texas.",
    "Engineers at the observatory measured the comet's tail at roughly 40 million kilometres.",
]


def digest(doc):
    """One line per sentence of exact, comparable facts (no floats: they would be platform noise)."""
    return {
        "tokens": " ".join(t.text for t in doc),
        "pos": " ".join(t.pos_ for t in doc),
        "dep": " ".join(f"{t.dep_}>{t.head.i}" for t in doc),
        "ents": "; ".join(f"{e.text}/{e.label_}" for e in doc.ents),
    }


def main(argv):
    bench = "--bench" in argv[1:]
    target = os.environ.get("PW_TARGET", "cpu")
    if bench and target != "gpu":
        print("SKIP: benchmarks are for the real gpu target only")
        return 77
    try:
        import spacy
        if target == "gpu":
            if not spacy.prefer_gpu():
                print("SKIP: target gpu, but spaCy found no usable GPU (cupy missing, or no CUDA device)")
                return 77
        elif target != "cpu":
            print(f"SKIP: target {target} is not supported")
            return 77
        nlp = spacy.load("en_core_web_sm")
    except ImportError as e:
        print(f"SKIP: missing Python package ({e})")
        return 77
    docs = [digest(d) for d in nlp.pipe(TEXTS)]
    output = {k: " || ".join(d[k] for d in docs) for k in ("tokens", "pos", "dep", "ents")}
    output["model_version"] = nlp.meta["version"]
    metrics = {}
    if bench:
        corpus = TEXTS * 250
        words = sum(len(d) for d in nlp.pipe(corpus))
        times = []
        for _ in range(int(os.environ.get("PW_SPACY_REPEATS", "5"))):
            t = time.perf_counter()
            for _ in nlp.pipe(corpus, batch_size=64):
                pass
            times.append(time.perf_counter() - t)
        metrics = {"words_per_s": words / statistics.median(times), "docs_per_s": len(corpus) / statistics.median(times)}
    print(json.dumps({"output": output, "detail": f"{sum(len(t.split()) for t in output['tokens'].split(' || '))} tokens, en_core_web_sm {output['model_version']}, spaCy {spacy.__version__}, target {target}",
                      "metrics": metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
