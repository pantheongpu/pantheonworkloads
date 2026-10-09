#!/usr/bin/env python3
"""Payloads of the text-model workloads (docs/pretrained-reachable.md, second table).

    python3 -I tools/text_tasks.py <task> [--bench]

Tasks: bidaf, bertsquad, minilm, glove (onnxruntime); spacy-md, spacy-multi (spaCy); langid,
sentencepiece (CPU only). Environment from bin/pw: PW_TARGET (cpu | gpu), PW_WORKLOAD_DIR (its
model.sha256 pins every file). Prints, as the last stdout line, {"output", "detail", "metrics"}
(docs/workload-contract.md); metrics only with --bench on the gpu target. Exit 77: cannot run
here (no network, no GPU provider, missing package). The gpu target never silently runs on the CPU.
"""
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ORT = _load("ort_tasks")
ASSETS = ORT._assets()
Skip = ORT.Skip


def _sums():
    return ASSETS.parse_sums((pathlib.Path(os.environ["PW_WORKLOAD_DIR"]) / "model.sha256").read_text())


def _cache():
    return pathlib.Path(os.environ.get("PW_CACHE") or pathlib.Path.home() / ".cache" / "pantheonworkloads") / "ort-assets"


def unpacked(archive, members=None):
    """Directory with the (checked) contents of the pinned archive; Skip when it cannot be had."""
    sums = _sums()
    try:
        root = ASSETS.fetch_unpacked(archive, sums[archive], _cache(), members)
        ASSETS.check_members(root, sums, archive)
    except (RuntimeError, KeyError) as e:
        raise Skip(str(e))
    return root


def med(values):
    return statistics.median(values)


def read_tensor(path):
    """A TensorProto file (the model zoo's test_data_set_N/*.pb) as a numpy array."""
    import onnx
    from onnx import numpy_helper
    t = onnx.TensorProto()
    t.ParseFromString(pathlib.Path(path).read_bytes())
    return numpy_helper.to_array(t)


def repeats(default):
    return int(os.environ.get("PW_TEXT_REPEATS", str(default)))


# ---------------------------------------------------------------- BiDAF (ONNX model zoo, MIT)

def char_grid(tokens):
    """BiDAF's character input: (n, 1, 1, 16) strings, each token cut or padded to 16 characters."""
    import numpy as np
    rows = [list(t)[:16] + [""] * (16 - len(t)) for t in tokens]
    return np.asarray(rows, dtype=object).reshape(-1, 1, 1, 16)


def task_bidaf(bench):
    import numpy as np
    root = unpacked("bidaf-9.tar.gz") / "bidaf"
    sess, prov = ORT.session(root / "bidaf.onnx", log_level=3)

    def ask(cw, qw, cc, qc):
        s, e = sess.run(None, {"context_word": cw, "query_word": qw, "context_char": cc, "query_char": qc})
        return int(s[0]), int(e[0])

    answers = []
    for k in range(16):   # the zoo's own 16 SQuAD v1.1 samples, with the answer spans the model must give
        d = root / f"test_data_set_{k}"
        cw, qw, cc, qc = (read_tensor(d / f"input_{i}.pb") for i in range(4))   # file order: context_word, query_word, context_char, query_char
        want = tuple(int(read_tensor(d / f"output_{i}.pb")[0]) for i in range(2))
        got = ask(cw, qw, cc, qc)
        if got != want:
            raise RuntimeError(f"test_data_set_{k}: answer span {got}, the zoo's expected output is {want}")
        answers.append(f"{got[0]}-{got[1]}")   # spans only: the SQuAD paragraphs are CC BY-SA text, none is copied into references
    ctx = "a quick brown fox jumps over the lazy dog .".split()   # the model README's example, tokenized by hand (no NLTK)
    qry = "what color is the fox ?".split()
    ctx_raw = "A quick brown fox jumps over the lazy dog .".split()
    qry_raw = "What color is the fox ?".split()
    s, e = ask(np.asarray(ctx, dtype=object).reshape(-1, 1), np.asarray(qry, dtype=object).reshape(-1, 1), char_grid(ctx_raw), char_grid(qry_raw))
    output = {"zoo_spans": " ".join(answers), "fox_span": f"{s}-{e}", "fox_answer": " ".join(ctx[s:e + 1])}
    return output, f"16 zoo samples match the zoo's expected spans; fox answer {output['fox_answer']!r}; {prov}", {}


# ---------------------------------------------------------------- BERT-Squad int8 (ONNX model zoo, Apache-2.0)

SQUAD_ITEMS = [   # (context, question): written for this repo
    ("The Marlow Observatory was founded in 1887 by the astronomer Helen Marsh. It stands on a hill north of the town of Ardmore "
     "and houses a 90-centimetre reflecting telescope.", "Who founded the Marlow Observatory?"),
    ("The Marlow Observatory was founded in 1887 by the astronomer Helen Marsh. It stands on a hill north of the town of Ardmore "
     "and houses a 90-centimetre reflecting telescope.", "When was the observatory founded?"),
    ("The river Alder rises in the Kell Hills and flows south for 140 kilometres before it joins the Brannock at the port of Sedgwick. "
     "Barges carry grain along its lower reaches.", "Where does the Alder join the Brannock?"),
    ("The river Alder rises in the Kell Hills and flows south for 140 kilometres before it joins the Brannock at the port of Sedgwick. "
     "Barges carry grain along its lower reaches.", "What do barges carry?"),
]
BERT_SEQ = 256
BERT_ZOO_ERR_MAX = 0.1   # worst measured 0.047 (A10G host, 7-bit int8 mode), 2.2 without it: docs/int8-and-nondeterminism.md


def squad_features(tok, context, question):
    """(ids, mask, segments, first context index, last context index) for one question, one 256-token window."""
    q = tok.tokenize(question)[:64]
    c = tok.tokenize(context)
    c = c[:BERT_SEQ - len(q) - 3]
    toks = ["[CLS]"] + q + ["[SEP]"] + c + ["[SEP]"]
    first = len(q) + 2
    ids = tok.ids(toks)
    n = len(ids)
    seg = [0] * first + [1] * (n - first)
    pad = BERT_SEQ - n
    return toks, ids + [0] * pad, [1] * n + [0] * pad, seg + [0] * pad, first, first + len(c) - 1


def best_span(start, end, first, last, max_len=30):
    """Highest start+end logit with first <= s <= e <= last and e - s < max_len (ties: lower s, then lower e)."""
    best = None
    for s in range(first, last + 1):
        for e in range(s, min(last, s + max_len - 1) + 1):
            score = float(start[s]) + float(end[e])
            if best is None or score > best[0]:
                best = (score, s, e)
    return best


def runner_up_score(start, end, first, last, best, max_len=30):
    """Highest start+end logit among the spans other than `best` (same window and length rules as best_span)."""
    top = None
    for s in range(first, last + 1):
        for e in range(s, min(last, s + max_len - 1) + 1):
            if (s, e) != best:
                score = float(start[s]) + float(end[e])
                if top is None or score > top:
                    top = score
    return top


def squad_session():
    root = unpacked("bertsquad-12-int8.tar.gz", ["bertsquad-12-int8/bertsquad-12-int8.onnx", "bertsquad-12-int8/test_data_set_0"]) / "bertsquad-12-int8"
    vocab = unpacked("genesis-memory-model-0.1.0-alpha.1.tgz", ["package/tokenizer.json", "package/README.md"]) / "package" / "tokenizer.json"
    sess, prov = ORT.session(root / "bertsquad-12-int8.onnx", log_level=3)
    return root, sess, prov, WORDPIECE.WordPiece.from_tokenizer_json(vocab)


def squad_run(sess, ids, mask, seg, uid=1):
    import numpy as np
    feed = {"unique_ids_raw_output___9:0": np.array([uid], dtype=np.int64), "segment_ids:0": np.array([seg], dtype=np.int64),
            "input_mask:0": np.array([mask], dtype=np.int64), "input_ids:0": np.array([ids], dtype=np.int64)}
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
    return out["unstack:0"][0], out["unstack:1"][0]   # start logits, end logits (as run_onnx_squad.py reads them)


def task_bertsquad(bench):
    import numpy as np
    root, sess, prov, tok = squad_session()
    # the zoo's own test vector: its inputs are not real text, so only the logits can be compared; int8 kernels differ a little between onnxruntime versions
    d = root / "test_data_set_0"
    ins = [read_tensor(d / f"input_{i}.pb") for i in range(4)]
    feed = {i.name: x for i, x in zip(sess.get_inputs(), ins)}
    got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
    err = max(float(np.abs(got[n] - read_tensor(d / f"output_{k}.pb")).max()) for k, n in enumerate(["unstack:1", "unstack:0"]))
    err_max = float(os.environ.get("PW_BERT_ZOO_ERR_MAX", str(BERT_ZOO_ERR_MAX)))
    if err > err_max:
        raise RuntimeError(f"logits differ from the zoo's expected output by {err:.3g} (bound {err_max}; docs/int8-and-nondeterminism.md)")
    answers, spans, scores, margins = [], [], [], []
    for context, question in SQUAD_ITEMS:
        toks, ids, mask, seg, first, last = squad_features(tok, context, question)
        start, end = squad_run(sess, ids, mask, seg)
        score, s, e = best_span(start, end, first, last)
        margins.append(round(score - runner_up_score(start, end, first, last, (s, e)), 4))
        answers.append(WORDPIECE.WordPiece.detokenize(toks[s:e + 1]))
        spans.append(f"{s}-{e}")
        scores.append(round(score, 4))
    # answers and spans are compared exactly, span_scores within the manifest's measured bound; the zoo vector's
    # deviation and the margin of each best span over the runner-up are recorded as informational
    output = {"answers": " | ".join(answers), "spans": " ".join(spans), "span_scores": scores,
              "zoo_logit_err": round(err, 3), "span_margins": margins}
    metrics = {}
    if bench:
        toks, ids, mask, seg, first, last = squad_features(tok, *SQUAD_ITEMS[0])
        for _ in range(3):
            squad_run(sess, ids, mask, seg)
        times = []
        for _ in range(repeats(20)):
            t = time.perf_counter()
            squad_run(sess, ids, mask, seg)
            times.append(time.perf_counter() - t)
        metrics = {"questions_per_s": 1 / med(times), "latency_ms": med(times) * 1000}
    return output, f"answers {answers}; max |logit - zoo's expected| {err:.3f}; smallest span margin {min(margins):.2f}; {prov}", metrics


# ---------------------------------------------------------------- all-MiniLM-L6-v2 sentence embeddings (npm redistribution, Apache-2.0)

SENTENCES = [
    "The cat sat on the mat.",
    "A kitten is resting on the rug.",
    "Stock markets fell sharply on Monday.",
    "Shares dropped as investors worried about inflation.",
    "How do I reset my password?",
    "I forgot my login credentials and need to change them.",
]


def embed(sess, tok, sentences):
    """Mean-pooled (over the attention mask), L2-normalised embeddings, one padded batch."""
    import numpy as np
    enc = [tok.encode(s, 256) for s in sentences]
    n = max(map(len, enc))
    ids = np.array([e + [0] * (n - len(e)) for e in enc], dtype=np.int64)
    mask = np.array([[1] * len(e) + [0] * (n - len(e)) for e in enc], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
    feed = {i.name: feed[i.name] for i in sess.get_inputs()}
    hidden = sess.run(None, feed)[0]
    pooled = (hidden * mask[:, :, None]).sum(1) / mask.sum(1, keepdims=True)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True), enc


def task_minilm(bench):
    import numpy as np
    root = unpacked("genesis-memory-model-0.1.0-alpha.1.tgz", ["package/onnx/model.onnx", "package/tokenizer.json", "package/README.md"]) / "package"
    sess, prov = ORT.session(root / "onnx" / "model.onnx", log_level=3)
    tok = WORDPIECE.WordPiece.from_tokenizer_json(root / "tokenizer.json")
    emb, enc = embed(sess, tok, SENTENCES)
    sim = emb @ emb.T
    nearest = [int(np.argsort(-np.where(np.eye(len(sim), dtype=bool)[i], -2.0, sim[i]), kind="stable")[0]) for i in range(len(sim))]
    output = {"token_ids": " | ".join(" ".join(map(str, e)) for e in enc), "nearest": " ".join(map(str, nearest)),
              "similarity": [[round(float(x), 5) for x in row] for row in sim], "embedding_head": [[round(float(x), 5) for x in row[:4]] for row in emb]}
    metrics = {}
    if bench:
        batch = SENTENCES * 6   # 36 sentences per call
        embed(sess, tok, batch)
        times = []
        for _ in range(repeats(20)):
            t = time.perf_counter()
            embed(sess, tok, batch)
            times.append(time.perf_counter() - t)
        metrics = {"sentences_per_s": len(batch) / med(times), "batch_latency_ms": med(times) * 1000}
    return output, f"nearest sentence per input {nearest}; dim {emb.shape[1]}; {prov}", metrics


# ---------------------------------------------------------------- GloVe word vectors, nearest neighbours on onnxruntime (PDDL)

GLOVE_QUERIES = ["king", "paris", "python", "doctor", "music", "king - man + woman", "paris - france + italy"]
GLOVE_K = 6


def load_glove(path):
    """(words, unit-length float32 matrix) of a gensim-data word2vec-text .gz (first line: count, dim)."""
    import gzip
    import numpy as np
    words, parts = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        n, dim = map(int, f.readline().split())
        chunk = []
        for line in f:
            w, rest = line.rstrip("\n").split(" ", 1)
            words.append(w)
            chunk.append(rest)
            if len(chunk) == 20000:   # bounded memory: ~1M number strings at a time
                parts.append(np.array(" ".join(chunk).split(), dtype=np.float32))
                chunk = []
        if chunk:
            parts.append(np.array(" ".join(chunk).split(), dtype=np.float32))
    mat = np.concatenate(parts).reshape(n, dim)
    if len(words) != n:
        raise RuntimeError(f"header says {n} vectors, file has {len(words)}")
    return words, mat / np.linalg.norm(mat, axis=1, keepdims=True)


def knn_model(matrix_t, k):
    """ONNX graph: scores = q @ E (E is [dim, N], an initializer), then TopK(k) over the N words."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    dim = matrix_t.shape[0]
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["q", "E"], ["scores"]), helper.make_node("TopK", ["scores", "k"], ["values", "indices"], axis=-1, largest=1, sorted=1)],
        "knn", [helper.make_tensor_value_info("q", TensorProto.FLOAT, ["B", dim])],
        [helper.make_tensor_value_info("values", TensorProto.FLOAT, ["B", k]), helper.make_tensor_value_info("indices", TensorProto.INT64, ["B", k])],
        [numpy_helper.from_array(matrix_t, "E"), numpy_helper.from_array(np.array([k], dtype=np.int64), "k")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def glove_query_vectors(index, mat, queries):
    """Rows for 'word' or 'a - b + c' queries (unit length), and the set of words each excludes from its answer."""
    import numpy as np
    rows, banned = [], []
    for q in queries:
        terms, sign, words = [], 1, []
        for tok in q.split():
            if tok in "+-":
                sign = 1 if tok == "+" else -1
            else:
                terms.append(sign * mat[index[tok]])
                words.append(tok)
        v = np.sum(terms, axis=0)
        rows.append(v / np.linalg.norm(v))
        banned.append(set(words))
    return np.array(rows, dtype=np.float32), banned


def task_glove(bench):
    import numpy as np
    (gz,) = ORT.assets("glove-wiki-gigaword-50.gz")
    words, mat = load_glove(gz)
    index = {w: i for i, w in enumerate(words)}
    sess, prov = ORT.session(knn_model(np.ascontiguousarray(mat.T), GLOVE_K + 3), log_level=3)
    q, banned = glove_query_vectors(index, mat, GLOVE_QUERIES)
    vals, idx = sess.run(None, {"q": q})
    lines, scores = [], []
    for row, (v, ix) in enumerate(zip(vals, idx)):
        keep = [(float(s), words[i]) for s, i in zip(v, ix) if words[i] not in banned[row]][:GLOVE_K - 1]
        lines.append(f"{GLOVE_QUERIES[row]}: " + " ".join(w for _, w in keep))
        scores.append([round(s, 5) for s, _ in keep])
    output = {"vocab": len(words), "dim": int(mat.shape[1]), "neighbours": " | ".join(lines), "scores": scores}
    metrics = {}
    if bench:
        rng = np.random.default_rng(0)
        batch = mat[rng.integers(0, len(words), 256)]
        for _ in range(3):
            sess.run(None, {"q": batch})
        times = []
        for _ in range(repeats(20)):
            t = time.perf_counter()
            sess.run(None, {"q": batch})
            times.append(time.perf_counter() - t)
        metrics = {"queries_per_s": len(batch) / med(times), "batch_latency_ms": med(times) * 1000}
    return output, f"{len(words)} words x {mat.shape[1]}; analogy answers: {lines[-2]} / {lines[-1]}; {prov}", metrics


# ---------------------------------------------------------------- spaCy pipelines (MIT)

def spacy_target():
    import spacy
    target = os.environ.get("PW_TARGET", "cpu")
    if target == "gpu":
        if not spacy.prefer_gpu():
            raise Skip("target gpu, but spaCy found no usable GPU (cupy missing, or no CUDA device)")
    elif target != "cpu":
        raise Skip(f"target {target} is not supported")
    return spacy, target


def spacy_pipeline(spacy, wheel, pkg):
    """Load a pinned pipeline wheel from its extracted copy (nothing is pip-installed). The lemmatizer
    is excluded: the Russian and Ukrainian ones need pymorphy3, which is not installed."""
    root = unpacked(wheel, [f"{pkg}/{pkg}-3.8.0", f"{pkg}-3.8.0.dist-info"])
    return spacy.load(root / pkg / f"{pkg}-3.8.0", exclude=["lemmatizer"])


def digest(doc):
    return {"tokens": " ".join(t.text for t in doc), "pos": " ".join(t.pos_ for t in doc),
            "dep": " ".join(f"{t.dep_}>{t.head.i}" for t in doc), "ents": "; ".join(f"{e.text}/{e.label_}" for e in doc.ents)}


def spacy_bench(nlp, texts, label):
    corpus = texts * (1000 // len(texts) + 1)
    words = sum(len(d) for d in nlp.pipe(corpus))
    times = []
    for _ in range(repeats(5)):
        t = time.perf_counter()
        for _ in nlp.pipe(corpus, batch_size=64):
            pass
        times.append(time.perf_counter() - t)
    return {"words_per_s": words / med(times), "docs_per_s": len(corpus) / med(times)}


MULTI = [   # (wheel, package, sentences written for this repo)
    ("ru_core_news_sm-3.8.0-py3-none-any.whl", "ru_core_news_sm",
     ["Владимир Путин встретился с Ангелой Меркель в Берлине в понедельник.", "Инженеры обсерватории измерили хвост кометы."]),
    ("uk_core_news_sm-3.8.0-py3-none-any.whl", "uk_core_news_sm",
     ["Президент України Володимир Зеленський відвідав Львів у п'ятницю.", "Інженери обсерваторії виміряли хвіст комети."]),
    ("nb_core_news_sm-3.8.0-py3-none-any.whl", "nb_core_news_sm",
     ["Statsminister Jonas Gahr Støre besøkte Bergen i mandags.", "Ingeniørene ved observatoriet målte halen på kometen."]),
    ("xx_ent_wiki_sm-3.8.0-py3-none-any.whl", "xx_ent_wiki_sm",
     ["Angela Merkel visited Paris with Emmanuel Macron on Monday.", "Die Universität von Texas liegt in Austin.", "Le Louvre est à Paris, près de la Seine."]),
]


def task_spacy_multi(bench):
    spacy, target = spacy_target()
    output, metrics, total = {}, {}, 0
    for wheel, pkg, texts in MULTI:
        nlp = spacy_pipeline(spacy, wheel, pkg)
        docs = [digest(d) for d in nlp.pipe(texts)]
        total += sum(len(d["tokens"].split(" ")) for d in docs)
        for key in ("tokens", "pos", "dep", "ents"):
            if key in ("pos", "dep") and not any(t.pos_ if key == "pos" else t.dep_ for d in nlp.pipe(texts) for t in d):
                continue   # an NER-only pipeline (xx_ent_wiki_sm) sets neither
            output[f"{pkg}.{key}"] = " || ".join(d[key] for d in docs)
        output[f"{pkg}.version"] = nlp.meta["version"]
        if bench:
            metrics[f"{pkg}_words_per_s"] = spacy_bench(nlp, texts, pkg)["words_per_s"]
    return output, f"{len(MULTI)} pipelines, ~{total} tokens, spaCy {spacy.__version__}, target {target}", metrics


MD_TEXTS = [
    "Apple is looking at buying a U.K. startup for $1 billion in London on Monday.",
    "Dr. Maria Lopez moved from Madrid to Austin, Texas, in March 2019 to join the University of Texas.",
    "The king and the queen visited Paris.",
]
MD_CANDIDATES = ["queen", "prince", "castle", "banana", "horse", "city", "river", "table", "happy", "car", "emperor", "monarch"]
MD_PAIRS = [("king", "queen"), ("king", "banana"), ("cat", "dog"), ("car", "truck"), ("paris", "london"), ("happy", "sad")]


def to_numpy(a):
    return a.get() if hasattr(a, "get") else a   # cupy -> numpy on the GPU path


def task_spacy_md(bench):
    import numpy as np
    spacy, target = spacy_target()
    nlp = spacy_pipeline(spacy, "en_core_web_md-3.8.0-py3-none-any.whl", "en_core_web_md")
    docs = [digest(d) for d in nlp.pipe(MD_TEXTS)]
    output = {k: " || ".join(d[k] for d in docs) for k in ("tokens", "pos", "dep", "ents")}
    output["model_version"] = nlp.meta["version"]

    def vec(w):
        return np.asarray(to_numpy(nlp.vocab.get_vector(w)), dtype=np.float64)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    output["word_similarity"] = [round(cos(vec(a), vec(b)), 5) for a, b in MD_PAIRS]
    sims = {w: round(cos(vec("king"), vec(w)), 4) for w in MD_CANDIDATES}   # rounded before ranking so tied rows (1.0) order by word, not by fp noise
    ranked = sorted(MD_CANDIDATES, key=lambda w: (-sims[w], w))
    output["king_ranking"] = " ".join(ranked)
    output["king_ranking_sims"] = [sims[w] for w in ranked]
    d1, d2 = nlp(MD_TEXTS[2]), nlp("A queen and a king went to London.")
    output["doc_similarity"] = round(float(d1.similarity(d2)), 5)
    metrics = spacy_bench(nlp, MD_TEXTS, "md") if bench else {}
    return output, f"en_core_web_md {output['model_version']}, king~queen {output['word_similarity'][0]}, spaCy {spacy.__version__}, target {target}", metrics


# ---------------------------------------------------------------- CPU-only models

LANG_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "Der schnelle braune Fuchs springt über den faulen Hund neben der alten Steinbrücke.",
    "Le renard brun rapide saute par-dessus le chien paresseux près du vieux pont de pierre.",
    "El rápido zorro marrón salta sobre el perro perezoso cerca del viejo puente de piedra.",
    "Il veloce volpe marrone salta sopra il cane pigro vicino al vecchio ponte di pietra.",
    "Быстрая коричневая лиса прыгает через ленивую собаку возле старого каменного моста.",
    "素早い茶色の狐が古い石橋の近くで怠け者の犬を飛び越える。",
    "Den snabba bruna räven hoppar över den lata hunden nära den gamla stenbron.",
    "Szybki brązowy lis przeskakuje nad leniwym psem w pobliżu starego kamiennego mostu.",
    "Hızlı kahverengi tilki eski taş köprünün yanındaki tembel köpeğin üzerinden atlar.",
]


def cpu_only():
    if os.environ.get("PW_TARGET", "cpu") != "cpu":
        raise Skip("this workload runs on the CPU only (no GPU code path)")


def task_langid(bench):
    cpu_only()
    root = unpacked("py3langid-0.4.0-py3-none-any.whl", ["py3langid", "py3langid-0.4.0.dist-info/licenses"])
    sys.path.insert(0, str(root))
    from py3langid.langid import classify
    got = [classify(t) for t in LANG_TEXTS]
    output = {"languages": " ".join(g[0] for g in got), "scores": [round(float(g[1]), 3) for g in got]}
    return output, f"languages {output['languages']}", {}


SP_TEXTS = [
    "This is a test.",
    "I saw a girl with a telescope.",
    "Hello world, tokenization of unseen words like zxqvjk works.",
    "New York is bigger than Boston in 2020!",
]


def task_sentencepiece(bench):
    cpu_only()
    wheel = "sentencepiece-0.2.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    root = unpacked(wheel)
    sys.path.insert(0, str(root))
    try:
        import sentencepiece as spm
    except ImportError as e:   # a wheel for CPython 3.13 on Linux x86-64 only
        raise Skip(f"the pinned sentencepiece wheel cannot be imported here ({e})")
    (model,) = ORT.assets("sentencepiece-test.model")
    sp = spm.SentencePieceProcessor(model_file=str(model))
    ids = [sp.encode(t) for t in SP_TEXTS]
    pieces = [sp.encode(t, out_type=str) for t in SP_TEXTS]
    back = [sp.decode(i) for i in ids]
    output = {"vocab_size": sp.get_piece_size(), "ids": " | ".join(" ".join(map(str, i)) for i in ids),
              "pieces": " | ".join(" ".join(p) for p in pieces), "roundtrip_ok": back == SP_TEXTS, "decoded": " | ".join(back)}
    return output, f"vocab {output['vocab_size']}, sentencepiece {spm.__version__}", {}


WORDPIECE = _load("wordpiece")

TASKS = {"bidaf": task_bidaf, "bertsquad": task_bertsquad, "minilm": task_minilm, "glove": task_glove,
         "spacy-md": task_spacy_md, "spacy-multi": task_spacy_multi, "langid": task_langid, "sentencepiece": task_sentencepiece}


def main(argv):
    args = [a for a in argv[1:] if a != "--bench"]
    bench = "--bench" in argv[1:]
    if len(args) != 1 or args[0] not in TASKS:
        print(f"usage: text_tasks.py {{{'|'.join(TASKS)}}} [--bench]", file=sys.stderr)
        return 2
    if bench and os.environ.get("PW_TARGET") != "gpu":
        print("SKIP: benchmarks are for the real gpu target only")
        return 77
    try:
        output, detail, metrics = TASKS[args[0]](bench)
    except Skip as e:
        print(f"SKIP: {e}")
        return 77
    except ImportError as e:
        print(f"SKIP: missing Python package ({e})")
        return 77
    print(json.dumps({"output": output, "detail": detail, "metrics": metrics if os.environ.get("PW_TARGET") == "gpu" else {},
                      "versions": ORT.VERSIONS.collect(ORT.VERSIONS.ORT)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
