"""Tests for the synthetic-GGUF generator and the llamacpp-synth-* workloads.

No network, no GPU and no llama.cpp build needed: the generator is pure Python (numpy + the MIT-licensed
`gguf` package; the whole module skips when those are missing), and the workload scripts are driven with stub binaries.

    python3 -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LC = ROOT / "tools" / "llamacpp"
sys.path.insert(0, str(LC))
sys.path.insert(0, str(ROOT / "tools"))

try:
    import numpy as np
    import gguf
    import synth_gguf
    import synth_run
    import synth_workloads
    HAVE = True
except ImportError:
    # synth_workloads imports the generator (synth_gguf), which needs numpy and the `gguf` package, so
    # without them nothing in this module can run: skip it as a whole instead of failing the import.
    raise unittest.SkipTest("numpy and gguf are needed (pip install numpy gguf==0.19.0)")
import validate  # noqa: E402

# sha256 of the F16 file for each architecture at tiny size, default seed, generator version 1.
# If the generator changes on purpose: bump GENERATOR_VERSION, regenerate with
#   python3 tools/llamacpp/synth_gguf.py <arch> -o x.gguf && sha256sum x.gguf
# and re-run `synth_workloads.py tune` (every quantised model changes too).
GOLDEN = {
    "llama": "96f88f66fdcfe57ff6a0fb748c64164acac8cfdd65380e2c71bb978ace3520ba",
    "mistral": "c4131594c07b6acfeb03d8d84539fea9d5af9dc8176baa335ec0829ed0a97852",
    "mixtral": "3344e2491817914dd452b6d78392f42aaa48ab2c08391093a0ca88e3a1312676",
    "qwen2": "db37e6befacfe909b1a0d4c6ae9da796db530f17a54e742f1777f38fdf1aac91",
    "gemma": "ac303b5ebeefb68525358e4cd7c245634e7e9f8728891d7b1c262db5c94375c0",
    "phi3": "f2e03ba5411856f5b387c77ed0d843b193250156d2cc39a83a5a61d3019facdf",
}


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(HAVE, "numpy and gguf are needed")
class Generator(unittest.TestCase):
    def make(self, tmp, arch, seed=synth_gguf.DEFAULT_SEED, size="tiny", name="m.gguf"):
        path = os.path.join(tmp, name)
        synth_gguf.write_model(path, arch, seed, size)
        return path

    def test_same_seed_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arch in synth_gguf.ARCHS:
                a = self.make(tmp, arch, name="a.gguf")
                b = self.make(tmp, arch, name="b.gguf")
                self.assertEqual(sha(a), sha(b), arch)

    def test_different_seed_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, "llama", seed=1, name="a.gguf")
            b = self.make(tmp, "llama", seed=2, name="b.gguf")
            self.assertNotEqual(sha(a), sha(b))

    def test_golden_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arch, want in GOLDEN.items():
                self.assertEqual(sha(self.make(tmp, arch)), want,
                                 f"{arch}: the generator's output changed; see the comment above GOLDEN")

    def test_random_stream_is_platform_independent(self):
        """Integer arithmetic plus one exact division: these values are the same everywhere."""
        z = synth_gguf.normals(synth_gguf.stream_key(7, "t"), 0, 4)
        self.assertEqual(z.dtype, np.float32)
        self.assertEqual([float(x) for x in z], [float(x) for x in synth_gguf.normals(synth_gguf.stream_key(7, "t"), 0, 4)])
        self.assertEqual(float(z[0]) * 65536, round(float(z[0]) * 65536))        # a multiple of 2**-16
        big = synth_gguf.normals(synth_gguf.stream_key(1, "stats"), 0, 200000)
        self.assertAlmostEqual(float(big.mean()), 0.0, delta=0.01)
        self.assertAlmostEqual(float(big.std()), 1.0, delta=0.01)
        # positions are independent of how the stream is cut into chunks
        whole = synth_gguf.normals(synth_gguf.stream_key(1, "x"), 0, 10)
        parts = np.concatenate([synth_gguf.normals(synth_gguf.stream_key(1, "x"), 0, 4),
                                synth_gguf.normals(synth_gguf.stream_key(1, "x"), 4, 6)])
        self.assertTrue(np.array_equal(whole, parts))

    def test_header_and_metadata_sanity(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arch in synth_gguf.ARCHS:
                path = self.make(tmp, arch)
                with open(path, "rb") as f:
                    self.assertEqual(f.read(4), b"GGUF")
                    self.assertEqual(int.from_bytes(f.read(4), "little"), 3)
                r = gguf.GGUFReader(path)
                a = synth_gguf.spec(arch)
                field = lambda k: r.fields[k].contents()
                self.assertEqual(field("general.architecture"), a["gguf_arch"])
                self.assertEqual(field(f"{a['gguf_arch']}.block_count"), 2)
                self.assertEqual(field(f"{a['gguf_arch']}.embedding_length"), 256)
                self.assertIn("RANDOM-WEIGHT", field("general.description"))
                self.assertIn("not trained", field("pantheonworkloads.synth.disclaimer"))
                self.assertEqual(field("pantheonworkloads.synth.seed"), synth_gguf.DEFAULT_SEED)
                self.assertEqual(len(field("tokenizer.ggml.tokens")), a["n_vocab"])
                names = {t.name: t for t in r.tensors}
                self.assertEqual(len(names), len(r.tensors))
                self.assertEqual(set(names), {n for n, *_ in synth_gguf.tensor_plan(a)})
                # GGUF dimensions are numpy's shape reversed
                self.assertEqual(list(names["token_embd.weight"].shape), [256, a["n_vocab"]])
                for n, shape, std, dtype in synth_gguf.tensor_plan(a):
                    self.assertEqual(tuple(reversed(names[n].shape.tolist())), shape, (arch, n))
                    want = gguf.GGMLQuantizationType.F32 if dtype == np.float32 else gguf.GGMLQuantizationType.F16
                    self.assertEqual(names[n].tensor_type, want, (arch, n))
                    self.assertTrue(np.isfinite(np.asarray(names[n].data, dtype=np.float32)).all(), (arch, n))
                # every row length a k-quant needs (multiple of 256) so the quantised variants are not fallbacks
                for n, shape, std, _ in synth_gguf.tensor_plan(a):
                    if len(shape) >= 2:
                        self.assertEqual(shape[-1] % 256, 0, (arch, n))

    def test_architecture_specifics(self):
        plan = lambda arch: {n: s for n, s, *_ in synth_gguf.tensor_plan(synth_gguf.spec(arch))}
        self.assertIn("blk.0.ffn_gate_exps.weight", plan("mixtral"))
        self.assertEqual(plan("mixtral")["blk.0.ffn_gate_exps.weight"][0], 8)
        self.assertIn("blk.0.ffn_gate_inp.weight", plan("mixtral"))
        self.assertIn("blk.0.attn_q.bias", plan("qwen2"))
        self.assertIn("blk.0.attn_qkv.weight", plan("phi3"))
        self.assertEqual(plan("phi3")["blk.0.ffn_up.weight"][0], 2 * 512)
        self.assertNotIn("output.weight", plan("gemma"))
        self.assertIn("output.weight", plan("llama"))
        self.assertEqual(synth_gguf.spec("mistral")["n_head_kv"], 2)
        self.assertEqual(synth_gguf.spec("gemma")["n_head_kv"], 1)
        self.assertEqual(synth_gguf.spec("llama")["n_head_kv"], synth_gguf.spec("llama")["n_head"])
        self.assertEqual(synth_gguf.ARCHS["mixtral"]["gguf_arch"], "llama")     # llama.cpp has no separate mixtral arch

    def test_vocabularies(self):
        for n in (300, 512, 1024):
            tokens, scores, types = synth_gguf.spm_vocab(n)
            self.assertEqual((len(tokens), len(scores), len(types)), (n, n, n))
            self.assertEqual(len(set(tokens)), n, "SPM tokens must be unique")
            self.assertEqual(tokens[:4], ["<unk>", "<s>", "</s>", "<|endoftext|>"])
        tokens, scores, types = synth_gguf.spm_vocab(1024)
        self.assertIn("▁the", tokens)                    # words and every prefix of them are pieces
        self.assertIn("▁th", tokens)
        tokens, types, merges = synth_gguf.bpe_vocab(1024)
        self.assertEqual(len(tokens), 1024)
        self.assertEqual(len(set(tokens)), 1024)
        vocab = set(tokens)
        self.assertTrue(merges)
        for m in merges:
            a, b = m.split(" ")
            self.assertTrue(a in vocab and b in vocab and a + b in vocab, m)

    def test_text_is_deterministic_and_seeded(self):
        self.assertEqual(synth_gguf.synth_text(5, 100), synth_gguf.synth_text(5, 100))
        self.assertNotEqual(synth_gguf.synth_text(5, 100), synth_gguf.synth_text(6, 100))
        self.assertTrue(synth_gguf.synth_text(5, 100).isascii())

    def test_imatrix_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "im.gguf")
            synth_gguf.write_imatrix(path, "mixtral")
            r = gguf.GGUFReader(path)
            names = {t.name: t for t in r.tensors}
            self.assertEqual(r.fields["general.architecture"].contents(), "imatrix")
            self.assertEqual(list(names["blk.0.ffn_up_exps.weight.in_sum2"].shape), [256, 8])
            self.assertEqual(list(names["blk.0.ffn_up_exps.weight.counts"].shape), [8])
            self.assertEqual(list(names["blk.0.attn_q.weight.counts"].shape), [1])
            self.assertTrue((np.asarray(names["output.weight.in_sum2"].data) > 0).all())
            self.assertNotIn("output_norm.weight.in_sum2", names)
            self.assertEqual(sha(path), self.again(path))

    def again(self, path):
        other = path + "2"
        synth_gguf.write_imatrix(other, "mixtral")
        return sha(other)

    def test_larger_sizes_listed_and_padded_vocab(self):
        self.assertEqual(synth_gguf.spec("llama", "m")["n_vocab"], synth_gguf.BENCH_VOCAB)
        self.assertEqual(synth_gguf.spec("llama", "tiny")["n_vocab"], synth_gguf.DEFAULT_VOCAB)
        sizes = [synth_gguf.n_params(synth_gguf.spec("llama", s)) for s in synth_gguf.SIZES]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], 6e9)


@unittest.skipUnless(HAVE, "numpy and gguf are needed")
class Runner(unittest.TestCase):
    def test_sha_mismatch_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            b = tmp / "bin"
            b.mkdir()
            write_stub_tools(b)
            shas = tmp / "m.sha256"
            shas.write_text("llama-tiny-F16 " + "0" * 64 + "\n")
            env = dict(os.environ)
            p = subprocess.run([sys.executable, str(LC / "synth_run.py"), "--arch", "llama", "--quants", "F16",
                                "--bin", str(b), "--cache", str(tmp / "cache"), "--expected-shas", str(shas)],
                               capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 1, p.stderr)
            self.assertIn("expected", p.stderr)

    def test_output_shape_with_stub_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            b = tmp / "bin"
            b.mkdir()
            write_stub_tools(b)
            p = subprocess.run([sys.executable, str(LC / "synth_run.py"), "--arch", "mixtral", "--quants", "F16,Q8_0",
                                "--bin", str(b), "--cache", str(tmp / "cache"), "--prompt", "the water"],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            out = rec["output"]
            self.assertEqual(set(out), {f"{q}.{k}" for q in ("F16", "Q8_0") for k in ("tokens", "top1_logit", "logsumexp", "ppl")})
            self.assertEqual(out["F16.tokens"], "1 2 3")
            self.assertEqual(out["Q8_0.ppl"], 12.5)
            self.assertEqual(rec["metrics"], {})
            self.assertIn("random-weight", rec["detail"])
            # the model files were really generated and "quantised" by the stub
            self.assertTrue((tmp / "cache" / "synth").is_dir())


def write_stub_tools(b):
    (b / "pw-probe").write_text("#!/bin/sh\necho '{\"n_vocab\":8,\"prompt_tokens\":[1],\"steps\":["
                                "{\"top1\":1,\"top2\":2,\"top1_logit\":3.5,\"margin\":1.0,\"mean\":0,\"std\":1,\"logsumexp\":4.0},"
                                "{\"top1\":2,\"top2\":3,\"top1_logit\":3.25,\"margin\":1.0,\"mean\":0,\"std\":1,\"logsumexp\":4.5},"
                                "{\"top1\":3,\"top2\":4,\"top1_logit\":3.0,\"margin\":1.0,\"mean\":0,\"std\":1,\"logsumexp\":5.0}],"
                                "\"generated\":[1,2,3]}'\n")
    (b / "llama-perplexity").write_text("#!/bin/sh\necho 'Final estimate: PPL = 12.5000 +/- 0.1' >&2\n")
    (b / "llama-quantize").write_text("#!/bin/sh\n# stub: copy the input to the output (args: [--imatrix f] in out type threads)\n"
                                      "while [ \"$1\" = --imatrix ]; do shift 2; done\ncp \"$1\" \"$2\"\n")
    for f in b.iterdir():
        f.chmod(0o755)


class Workloads(unittest.TestCase):
    NAMES = [w["name"] for w in synth_workloads.FUNCTIONAL + synth_workloads.BENCH]

    def test_files_match_the_generator(self):
        spreads = synth_workloads.load_spreads()
        for rel, text in synth_workloads.render(spreads).items():
            self.assertEqual((ROOT / rel).read_text(encoding="utf-8"), text,
                             f"{rel} differs from tools/llamacpp/synth_workloads.py (run `generate`)")

    def test_manifests_valid_and_say_what_they_are(self):
        for name in self.NAMES:
            path = ROOT / "workloads" / name / "manifest.yaml"
            errors, warnings = validate.check(path)
            self.assertEqual(errors, [], name)
            m = validate.load(path)
            self.assertEqual(m["runtime"], "llama.cpp")
            self.assertEqual(m["runtime_version"], synth_workloads.RUNTIME_VERSION)
            self.assertIn("RANDOM-WEIGHT", m["notes"], name)
            self.assertIn("NOT the quality or behaviour of any named pretrained model", m["notes"], name)
            self.assertIn("NOT a pretrained model", m["model"]["id"], name)
            self.assertIsNotNone(m["model"]["revision"])
            self.assertFalse(str(m["model"]["licence"]).upper().startswith("UNVERIFIED"))
            self.assertFalse(any("UNVERIFIED" in w for w in warnings), name)

    def test_kinds_and_targets(self):
        for w in synth_workloads.FUNCTIONAL:
            m = validate.load(ROOT / "workloads" / w["name"] / "manifest.yaml")
            self.assertEqual(m["kind"], "functional")
            self.assertEqual(m["targets"], ["cpu", "gpu", "sim:nvidia/*", "sim:amd/*"])
            self.assertEqual(m["compare"], "tolerance")
            self.assertEqual(m["timeout_s"], 3600)
        for w in synth_workloads.BENCH:
            m = validate.load(ROOT / "workloads" / w["name"] / "manifest.yaml")
            self.assertEqual(m["kind"], "benchmark")
            self.assertEqual(m["targets"], ["cpu", "gpu"])        # never sim:
            self.assertNotIn("compare", m)

    def test_tolerances_follow_the_policy(self):
        spreads = synth_workloads.load_spreads()
        for w in synth_workloads.FUNCTIONAL:
            m = validate.load(ROOT / "workloads" / w["name"] / "manifest.yaml")
            fields = m["tolerance"]["fields"]
            self.assertEqual(len(fields), 3 * len(w["quants"]), w["name"])
            for q in w["quants"]:
                rec = spreads[w["name"]]["quants"][q]
                floor_abs, floor_ppl = synth_workloads.FLOORS[synth_workloads.QUANT_CLASS[q]]
                worst = max(rec["top1_logit"], rec["logsumexp"])
                for key in ("top1_logit", "logsumexp"):
                    self.assertGreaterEqual(fields[f"{q}.{key}"]["abs"], max(floor_abs, 5 * worst) - 1e-9, (w["name"], q))
                self.assertGreaterEqual(fields[f"{q}.ppl"]["rel"], max(floor_ppl, 5 * rec["ppl_rel"]) - 1e-12, (w["name"], q))
                self.assertEqual(fields[f"{q}.ppl"]["abs"], 0.0)
            self.assertNotIn(".tokens", " ".join(fields))        # token ids are compared exactly

    def test_references_were_recorded_on_the_cpu_target(self):
        for w in synth_workloads.FUNCTIONAL:
            ref = json.loads((ROOT / "workloads" / w["name"] / "reference.json").read_text())
            self.assertEqual(ref["recorded_on"], "cpu", w["name"])
            self.assertEqual(set(ref["output"]), {f"{q}.{k}" for q in w["quants"] for k in ("tokens", "top1_logit", "logsumexp", "ppl")})
            for q in w["quants"]:
                ids = ref["output"][f"{q}.tokens"].split()
                self.assertEqual(len(ids), 6)
                self.assertTrue(all(i.isdigit() for i in ids))

    def test_every_quant_has_a_class_and_a_hash(self):
        spreads = synth_workloads.load_spreads()
        for w in synth_workloads.FUNCTIONAL:
            lines = (ROOT / "workloads" / w["name"] / "models.sha256").read_text().splitlines()
            hashes = dict(l.split() for l in lines if l and not l.startswith("#"))
            for q in w["quants"]:
                self.assertIn(q, synth_workloads.QUANT_CLASS)
                self.assertRegex(hashes[f"{w['arch']}-tiny-{q}"], r"^[0-9a-f]{64}$")
                self.assertEqual(hashes[f"{w['arch']}-tiny-{q}"], spreads[w["name"]]["quants"][q]["sha256"])

    def test_every_architecture_and_quant_group_is_covered(self):
        archs = {w["arch"] for w in synth_workloads.FUNCTIONAL}
        self.assertEqual(archs, {"llama", "mistral", "mixtral", "qwen2", "gemma", "phi3"})
        quants = {q for w in synth_workloads.FUNCTIONAL for q in w["quants"]}
        for q in ("F16", "BF16", "F32", "Q8_0", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q2_K", "IQ4_XS", "IQ2_XS", "TQ2_0"):
            self.assertIn(q, quants)
        self.assertEqual({w["arch"] for w in synth_workloads.BENCH}, archs)

    def test_scripts_parse_and_use_the_shared_helpers(self):
        for name in self.NAMES:
            f = ROOT / "workloads" / name / "run.sh"
            self.assertTrue(os.access(f, os.X_OK), name)
            self.assertEqual(subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode, 0, name)
            text = f.read_text()
            self.assertIn("lc_synth_prepare", text)
            self.assertIn("NOT a pretrained model", text)
        self.assertEqual(subprocess.run(["bash", "-n", str(LC / "synth.sh")], capture_output=True).returncode, 0)

    def test_skips_without_the_tools(self):
        """With no build and building disabled, every synthetic workload skips (77) instead of failing."""
        for name in (self.NAMES[0], synth_workloads.BENCH[0]["name"]):
            with tempfile.TemporaryDirectory() as tmp:
                e = {"PATH": os.environ["PATH"], "HOME": tmp, "PW_TARGET": "cpu", "PW_OUT": tmp,
                     "PW_WORKLOAD_DIR": str(ROOT / "workloads" / name), "PW_CACHE": tmp + "/cache",
                     "PW_LLAMACPP_NO_BUILD": "1"}
                p = subprocess.run(["bash", str(ROOT / "workloads" / name / "run.sh")], cwd=ROOT, env=e,
                                   capture_output=True, text=True, timeout=60)
                self.assertEqual(p.returncode, 77, (p.stdout, p.stderr))

    @unittest.skipUnless(HAVE, "numpy and gguf are needed")
    def test_whole_contract_with_stub_binaries(self):
        name = "llamacpp-synth-phi3"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            b = tmp / "bin"
            b.mkdir()
            write_stub_tools(b)
            (b / "llama-completion").write_text("#!/bin/sh\nexit 0\n")
            (b / "llama-bench").write_text("#!/bin/sh\nprintf '%s\\n' '[{\"n_prompt\":512,\"n_gen\":0,\"avg_ts\":100.0},{\"n_prompt\":0,\"n_gen\":128,\"avg_ts\":20.0}]'\n")
            for f in b.iterdir():
                f.chmod(0o755)
            e = {"PATH": os.environ["PATH"], "HOME": str(tmp), "PW_TARGET": "cpu", "PW_OUT": str(tmp),
                 "PW_WORKLOAD_DIR": str(ROOT / "workloads" / name), "PW_CACHE": str(tmp / "cache"),
                 "PW_LLAMACPP_BIN_DIR": str(b), "PW_SYNTH_SKIP_SHA": "1"}
            p = subprocess.run(["bash", str(ROOT / "workloads" / name / "run.sh")], cwd=ROOT, env=e,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, (p.stdout, p.stderr))
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertIn("Q4_K_M.tokens", rec["output"])
            self.assertIn("random-weight phi3", rec["detail"])
            self.assertEqual(rec["metrics"], {})
            # the benchmark variant on the same stub tools: the tiny size keeps generation fast
            bench = "llamacpp-synth-bench-phi3"
            e.update(PW_WORKLOAD_DIR=str(ROOT / "workloads" / bench), PW_SYNTH_SIZE="tiny")
            p = subprocess.run(["bash", str(ROOT / "workloads" / bench / "run.sh")], cwd=ROOT, env=e,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, (p.stdout, p.stderr))
            rec = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertEqual(rec["metrics"], {"pp512_tokens_per_s": 100.0, "tg128_tokens_per_s": 20.0})
            self.assertIn("phi3-tiny-Q4_K_M.gguf", rec["detail"])
            # benchmarks refuse simulated targets
            e.update(PW_TARGET="sim:nvidia/h100", PANTHEONSIM_DIR=str(tmp))
            p = subprocess.run(["bash", str(ROOT / "workloads" / bench / "run.sh")], cwd=ROOT, env=e,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 77, (p.stdout, p.stderr))


if __name__ == "__main__":
    unittest.main()
