"""Word error rate, shared by the runner's text comparison (bin/pw, `fields: {key: {wer: X}}`) and the
speech payloads (tools/speech_tasks.py). Pure Python, no dependencies."""
import re


def normalise(text):
    """Lower case, everything but a-z, 0-9 and apostrophes turned into a space (so case, punctuation and
    hyphens do not count), split into words."""
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def word_error_rate(ref, hyp):
    """Word-level edit distance / number of reference words (case, punctuation and hyphens ignored)."""
    r, h = normalise(ref), normalise(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)
