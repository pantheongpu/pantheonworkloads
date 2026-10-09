"""Tests for the text-model workloads' helpers (tools/ort_assets.py archive handling,
tools/wordpiece.py, tools/text_tasks.py). No network, no onnxruntime, no model: tiny archives and
vocabularies built on the fly, the pure functions, and the exit-77 paths.

    python3 -m unittest discover -s tests -v
"""
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ort_assets as assets  # noqa: E402
import wordpiece  # noqa: E402

_spec = importlib.util.spec_from_file_location("text_tasks", ROOT / "tools" / "text_tasks.py")
text_tasks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(text_tasks)

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def make_tgz(path, files):
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


class Archives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = pathlib.Path(self.tmp.name)
        self._urls = dict(assets.URLS)

    def tearDown(self):
        assets.URLS.clear()
        assets.URLS.update(self._urls)
        self.tmp.cleanup()

    def archive(self, name, maker, files):
        src = self.t / name
        maker(src, files)
        assets.URLS[name] = src.as_uri()
        return sha(src)

    def test_extract_all_or_some_members(self):
        make_tgz(self.t / "a.tgz", {"pkg/a.txt": b"A", "pkg/sub/b.txt": b"B", "other/c.txt": b"C"})
        out = self.t / "out"
        done = assets.extract_archive(self.t / "a.tgz", out, ["pkg/sub", "other/c.txt"])
        self.assertEqual(sorted(done), ["other/c.txt", "pkg/sub/b.txt"])
        self.assertFalse((out / "pkg" / "a.txt").exists())
        self.assertEqual((out / "pkg" / "sub" / "b.txt").read_bytes(), b"B")
        self.assertEqual(len(assets.extract_archive(self.t / "a.tgz", self.t / "all")), 3)

    def test_zip_and_wheel(self):
        make_zip(self.t / "w.whl", {"pkg/__init__.py": b"x = 1", "pkg-1.dist-info/LICENSE": b"MIT"})
        out = self.t / "out"
        self.assertEqual(len(assets.extract_archive(self.t / "w.whl", out)), 2)
        self.assertEqual((out / "pkg-1.dist-info" / "LICENSE").read_bytes(), b"MIT")

    def test_unsafe_names_are_refused(self):
        for bad in ("../evil.txt", "/abs.txt", "a/../../evil.txt"):
            make_zip(self.t / "bad.zip", {bad: b"x"})
            with self.assertRaises(RuntimeError, msg=bad):
                assets.extract_archive(self.t / "bad.zip", self.t / "out")

    def test_symlink_member_is_refused(self):
        with tarfile.open(self.t / "l.tgz", "w:gz") as t:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            t.addfile(info)
        with self.assertRaises(RuntimeError):
            assets.extract_archive(self.t / "l.tgz", self.t / "out")

    def test_fetch_unpacked_checks_extracts_and_deletes_the_archive(self):
        want = self.archive("m.tgz", make_tgz, {"p/model.bin": b"weights", "p/note.txt": b"n"})
        cache = self.t / "cache"
        root = assets.fetch_unpacked("m.tgz", want, cache, ["p/model.bin"])
        self.assertEqual((root / "p" / "model.bin").read_bytes(), b"weights")
        self.assertFalse((root / "p" / "note.txt").exists())
        self.assertFalse((cache / want[:16] / "m.tgz").exists(), "the archive is deleted after extraction")
        # a second call with the same members needs no archive (the URL is gone now)
        assets.URLS["m.tgz"] = (self.t / "gone.tgz").as_uri()
        self.assertEqual(assets.fetch_unpacked("m.tgz", want, cache, ["p/model.bin"]), root)
        # a bigger member set needs it again -> download fails -> RuntimeError
        with self.assertRaises(RuntimeError):
            assets.fetch_unpacked("m.tgz", want, cache, ["p/model.bin", "p/note.txt"])

    def test_wrong_checksum_is_refused(self):
        self.archive("m.tgz", make_tgz, {"p/x": b"1"})
        with self.assertRaises(RuntimeError):
            assets.fetch_unpacked("m.tgz", "0" * 64, self.t / "cache")

    def test_check_members(self):
        root = self.t / "root"
        (root / "p").mkdir(parents=True)
        (root / "p" / "f").write_bytes(b"data")
        good = hashlib.sha256(b"data").hexdigest()
        self.assertEqual(assets.check_members(root, {"a.tgz::p/f": good, "a.tgz": "1" * 64, "b.tgz::q": "2" * 64}, "a.tgz"), 1)
        with self.assertRaises(RuntimeError):
            assets.check_members(root, {"a.tgz::p/f": "0" * 64}, "a.tgz")
        with self.assertRaises(RuntimeError):
            assets.check_members(root, {"a.tgz::p/missing": good}, "a.tgz")


VOCAB = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "cat", "sat", "un", "##want", "##ed", "##ing", ".", ",", "cafe", "!", "a"])}


class WordPieceTests(unittest.TestCase):
    def setUp(self):
        self.tok = wordpiece.WordPiece(VOCAB)

    def test_basic_split_lowercase_punctuation(self):
        self.assertEqual(wordpiece.basic_tokens("The Cat, sat."), ["the", "cat", ",", "sat", "."])
        self.assertEqual(wordpiece.basic_tokens("a!b"), ["a", "!", "b"])

    def test_accents_are_stripped_when_lowercasing(self):
        self.assertEqual(wordpiece.basic_tokens("Café"), ["cafe"])
        self.assertEqual(wordpiece.basic_tokens("Café", lower=False), ["Café"])

    def test_cjk_characters_are_split(self):
        self.assertEqual(wordpiece.basic_tokens("a猫b"), ["a", "猫", "b"])

    def test_control_characters_are_dropped_and_whitespace_normalised(self):
        self.assertEqual(wordpiece.basic_tokens("a\u0000b\t c d"), ["ab", "c", "d"])

    def test_greedy_longest_match(self):
        self.assertEqual(self.tok.tokenize("unwanted"), ["un", "##want", "##ed"])
        self.assertEqual(self.tok.tokenize("wanting"), ["[UNK]"])   # 'want' alone is not a word start in this vocabulary
        self.assertEqual(self.tok.tokenize("unwantedx"), ["[UNK]"])  # one unmatched tail makes the whole word unknown

    def test_overlong_word_is_unknown(self):
        self.assertEqual(wordpiece.WordPiece(VOCAB, max_chars=5).tokenize("unwanted"), ["[UNK]"])

    def test_encode_adds_specials_and_truncates(self):
        self.assertEqual(self.tok.encode("The cat sat."), [2, 4, 5, 6, 11, 3])
        self.assertEqual(self.tok.encode("the cat sat the cat sat", max_len=5), [2, 4, 5, 6, 3])

    def test_detokenize_joins_pieces(self):
        self.assertEqual(wordpiece.WordPiece.detokenize(["un", "##want", "##ed", "cat"]), "unwanted cat")
        self.assertEqual(wordpiece.WordPiece.detokenize([]), "")

    def test_from_tokenizer_json(self):
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t) / "tokenizer.json"
            p.write_text(json.dumps({"model": {"vocab": VOCAB}}))
            self.assertEqual(wordpiece.WordPiece.from_tokenizer_json(p).tokenize("the cat"), ["the", "cat"])


class Pure(unittest.TestCase):
    def test_runner_up_score_excludes_the_best_span(self):
        start, end = [0, 5, 4, 0], [0, 1, 4, 3]
        score, s, e = best_span(start, end, 1, 3)
        self.assertEqual((score, s, e), (9.0, 1, 2))
        self.assertEqual(runner_up_score(start, end, 1, 3, (s, e)), 8.0)   # (1, 3) and (2, 2) both score 8
        self.assertIsNone(runner_up_score([1, 2], [1, 2], 1, 1, (1, 1)))

    def test_best_span_respects_order_window_and_length(self):
        start = [0, 5, 1, 9, 0, 0]
        end = [0, 0, 7, 0, 3, 0]
        # (1, 2) and (3, 4) both score 12: the earlier start wins a tie
        self.assertEqual(best_span(start, end, 1, 5), (12.0, 1, 2))
        # a better late start wins, and an end before the start is never taken
        self.assertEqual(best_span([0, 1, 9, 0], [0, 8, 2, 4], 1, 3), (13.0, 2, 3))
        # positions outside the context window do not count
        self.assertEqual(best_span([9, 0, 0], [9, 0, 1], 1, 2)[1:], (1, 2))
        # the answer may not be longer than max_len
        s2, e2 = best_span([5, 0, 0, 0], [0, 0, 0, 5], 0, 3, max_len=2)[1:]
        self.assertLess(e2 - s2, 2)

    def test_squad_features_layout(self):
        tok = wordpiece.WordPiece(VOCAB)
        toks, ids, mask, seg, first, last = text_tasks.squad_features(tok, "the cat sat .", "a cat .")
        self.assertEqual(toks, ["[CLS]", "a", "cat", ".", "[SEP]", "the", "cat", "sat", ".", "[SEP]"])
        self.assertEqual((first, last), (5, 8))
        self.assertEqual(len(ids), 256)
        self.assertEqual(sum(mask), 10)
        self.assertEqual(seg[:10], [0] * 5 + [1] * 5)
        self.assertEqual(set(ids[10:]), {0})

    def test_glove_queries_and_banned_words(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy not installed")
        import numpy as np
        mat = np.eye(4, dtype=np.float32)
        index = {"a": 0, "b": 1, "c": 2, "d": 3}
        rows, banned = text_tasks.glove_query_vectors(index, mat, ["a", "a - b + c"])
        self.assertAlmostEqual(float(np.linalg.norm(rows[1])), 1.0, places=5)
        self.assertEqual(banned, [{"a"}, {"a", "b", "c"}])
        np.testing.assert_allclose(rows[1], np.array([1, -1, 1, 0]) / 3 ** 0.5, rtol=1e-6)

    def test_char_grid_pads_and_cuts(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy not installed")
        g = text_tasks.char_grid(["ab", "x" * 20])
        self.assertEqual(g.shape, (2, 1, 1, 16))
        self.assertEqual(list(g[0, 0, 0, :3]), ["a", "b", ""])
        self.assertEqual(list(g[1, 0, 0]), ["x"] * 16)

    def test_load_glove_parses_and_normalises(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy not installed")
        import gzip
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t) / "v.gz"
            with gzip.open(p, "wt") as f:
                f.write("2 3\nfoo 3.0 0.0 4.0\nbar 0 2 0\n")
            words, mat = text_tasks.load_glove(p)
        self.assertEqual(words, ["foo", "bar"])
        self.assertAlmostEqual(float(mat[0, 0]), 0.6, places=6)
        self.assertAlmostEqual(float(mat[1, 1]), 1.0, places=6)

    def test_load_glove_rejects_a_short_file(self):
        if not HAVE_NUMPY:
            self.skipTest("numpy not installed")
        import gzip
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t) / "v.gz"
            with gzip.open(p, "wt") as f:
                f.write("3 2\nfoo 1 2\nbar 3 4\n")
            with self.assertRaises(Exception):
                text_tasks.load_glove(p)


best_span = text_tasks.best_span
runner_up_score = text_tasks.runner_up_score


class Scripts(unittest.TestCase):
    def test_every_new_workload_has_a_reference_or_is_a_benchmark(self):
        import validate
        for w in ("onnx-zoo-bidaf", "onnx-zoo-bertsquad-int8", "minilm-l6-v2-onnx", "glove-wiki-gigaword-50-knn", "spacy-en-core-web-md",
                  "spacy-multilingual-sm", "py3langid-wheel", "sentencepiece-test-model"):
            self.assertTrue((ROOT / "workloads" / w / "reference.json").exists(), w)
            self.assertEqual(validate.load(ROOT / "workloads" / w / "manifest.yaml")["kind"], "functional")

    def test_cpu_only_workloads_do_not_list_gpu(self):
        import validate
        for w in ("py3langid-wheel", "sentencepiece-test-model"):
            self.assertEqual(validate.load(ROOT / "workloads" / w / "manifest.yaml")["targets"], ["cpu"])

    def test_bench_refuses_cpu(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "text_tasks.py"), "minilm", "--bench"],
                           env=dict(os.environ, PW_TARGET="cpu"), capture_output=True, text=True)
        self.assertEqual(p.returncode, 77)

    def test_cpu_only_task_refuses_gpu(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "text_tasks.py"), "langid"],
                           env=dict(os.environ, PW_TARGET="gpu", PW_WORKLOAD_DIR=str(ROOT / "workloads" / "py3langid-wheel")), capture_output=True, text=True)
        self.assertEqual(p.returncode, 77, p.stdout + p.stderr)

    def test_bad_usage(self):
        p = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "text_tasks.py"), "nope"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)

    def test_no_venv_and_no_install_is_a_skip(self):
        for w in ("onnx-zoo-bidaf", "spacy-multilingual-sm"):
            e = dict(os.environ, PW_TARGET="cpu", PW_WORKLOAD_DIR=str(ROOT / "workloads" / w), PW_NO_INSTALL="1", PW_VENV_ROOT=tempfile.mkdtemp())
            e.pop("PW_ORT_PYTHON", None)
            e.pop("PW_SPACY_PYTHON", None)
            p = subprocess.run(["bash", str(ROOT / "workloads" / w / "run.sh")], env=e, capture_output=True, text=True)
            self.assertEqual(p.returncode, 77, p.stderr)
            self.assertTrue(p.stdout.startswith("SKIP:"))


if __name__ == "__main__":
    unittest.main()
