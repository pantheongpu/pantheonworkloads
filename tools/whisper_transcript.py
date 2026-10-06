#!/usr/bin/env python3
"""Normalises a whisper-cli transcript (env PW_TEXT), checks its word error rate against the
known words of the test clip and prints the pw contract JSON. Exits non-zero over the limit.

    PW_WHISPER_WER_MAX   limit (default 0.15)    PW_BACKEND   label for the detail line
"""
import json
import os
import re
import sys

# The words of President Kennedy's 1961 inaugural address that are in whisper.cpp's samples/jfk.wav.
TRUTH = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def wer(ref, hyp):
    """Word error rate: word-level Levenshtein distance over the reference length."""
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / len(r)


def main():
    out = norm(os.environ["PW_TEXT"])
    w = wer(TRUTH, out)
    limit = float(os.environ.get("PW_WHISPER_WER_MAX", "0.15"))
    if w > limit:
        sys.exit(f"word error rate {w:.3f} exceeds {limit}: {out!r}")
    print(json.dumps({"output": out, "detail": f"backend {os.environ.get('PW_BACKEND', '?')}, WER {w:.3f} vs known text",
                      "metrics": {}}))


if __name__ == "__main__":
    main()
