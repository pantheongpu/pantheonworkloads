"""Tests for bin/pw and tools/validate.py. Needs only python3 and PyYAML.

    python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)


def run_selftest(target="cpu", **env):
    """Run the selftest workload once with the given knobs; return (result, detail, parsed record)."""
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
        result, _, detail, rec = pw.execute("selftest", target, pathlib.Path(tmp))
    return result, detail, rec


class Execute(unittest.TestCase):
    def test_pass(self):
        result, detail, _ = run_selftest()
        self.assertEqual(result, "PASS")
        self.assertIn("target cpu", detail)

    def test_output_differs_from_reference(self):
        result, detail, _ = run_selftest(PW_SELFTEST_OUTPUT="something-else")
        self.assertEqual(result, "FAIL")
        self.assertIn("differs from reference", detail)

    def test_exit_77_is_skip(self):
        self.assertEqual(run_selftest(PW_SELFTEST_SKIP="1")[0], "SKIP")

    def test_other_exit_is_fail_with_stderr(self):
        result, detail, _ = run_selftest(PW_SELFTEST_FAIL="1")
        self.assertEqual(result, "FAIL")
        self.assertIn("boom", detail)

    def test_timeout(self):
        result, detail, _ = run_selftest(PW_SELFTEST_SLEEP="5", PW_TIMEOUT_S="0.5")
        self.assertEqual(result, "TIMEOUT")
        self.assertIn("0.5", detail)

    def test_last_line_must_be_json(self):
        result, detail, _ = run_selftest(PW_SELFTEST_BADJSON="1")
        self.assertEqual(result, "FAIL")
        self.assertIn("JSON", detail)

    def test_metrics_allowed_on_real_targets(self):
        self.assertEqual(run_selftest("gpu", PW_SELFTEST_METRICS="1")[0], "PASS")

    def test_metrics_refused_on_simulated_targets(self):
        result, detail, _ = run_selftest("sim:nvidia/h100", PW_SELFTEST_METRICS="1")
        self.assertEqual(result, "FAIL")
        self.assertIn("simulat", detail)

    def test_simulated_target_gets_profile(self):
        result, detail, _ = run_selftest("sim:amd/mi300x")
        self.assertEqual((result, detail), ("PASS", "target sim:amd/mi300x"))

    def test_unsupported_target_is_skip(self):
        self.assertEqual(run_selftest("tpu")[0], "SKIP")


class PytorchWorkloads(unittest.TestCase):
    """The pytorch-* workloads must SKIP (exit 77), not fail, where there is no PyTorch to use."""
    NAMES = ["pytorch-microsuite", "gpt2-small-pytorch", "bert-base-uncased-pytorch", "resnet18-randinit-pytorch"]

    def test_skip_without_torch(self):
        for name in self.NAMES:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                    os.environ, {"PW_NO_INSTALL": "1", "PW_VENV_ROOT": tmp, "PW_TORCH_PYTHON": ""}):
                result, _, detail, _ = pw.execute(name, "cpu", pathlib.Path(tmp) / "out")
            self.assertEqual(result, "SKIP", (name, detail))
            self.assertIn("PyTorch", detail)

    def test_skip_when_sim_build_missing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"PW_NO_INSTALL": "1", "PW_VENV_ROOT": tmp}):
            os.environ.pop("PANTHEONSIM_DIR", None)
            result, _, _, _ = pw.execute("pytorch-microsuite", "sim:nvidia/h100", pathlib.Path(tmp) / "out")
        self.assertEqual(result, "SKIP")

    def test_unsupported_target_is_skip(self):
        self.assertEqual(pw.execute("pytorch-microsuite", "tpu", pathlib.Path(tempfile.mkdtemp()))[0], "SKIP")


class Compare(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(pw.compare({"compare": "exact"}, "a", "a"))
        self.assertFalse(pw.compare({"compare": "exact"}, "a", "b"))

    def test_tolerance_numbers_and_lists(self):
        m = {"compare": "tolerance", "tolerance": {"abs": 0.01, "rel": 0.0}}
        self.assertTrue(pw.compare(m, [1.0, 2.005], [1.0, 2.0]))
        self.assertFalse(pw.compare(m, [1.0, 2.5], [1.0, 2.0]))
        self.assertFalse(pw.compare(m, [1.0], [1.0, 2.0]))

    def test_relative_tolerance(self):
        m = {"compare": "tolerance", "tolerance": {"abs": 0.0, "rel": 0.1}}
        self.assertTrue(pw.compare(m, 105.0, 100.0))
        self.assertFalse(pw.compare(m, 120.0, 100.0))

    def test_dicts_compare_key_by_key_with_per_field_tolerance(self):
        m = {"compare": "tolerance", "tolerance": {"abs": 1e-3, "rel": 0.0, "fields": {"loose": {"abs": 0.5}}}}
        ref = {"ids": "1 2 3", "tight": [1.0, 2.0], "loose": [10.0]}
        self.assertTrue(pw.compare(m, {"ids": "1 2 3", "tight": [1.0005, 2.0], "loose": [10.4]}, ref))
        self.assertFalse(pw.compare(m, {"ids": "1 2 3", "tight": [1.01, 2.0], "loose": [10.0]}, ref))
        self.assertFalse(pw.compare(m, {"ids": "1 2 4", "tight": [1.0, 2.0], "loose": [10.0]}, ref), "strings are exact")
        self.assertFalse(pw.compare(m, {"ids": "1 2 3", "tight": [1.0, 2.0]}, ref), "a missing key differs")
        self.assertFalse(pw.compare(m, {"ids": "1 2 3", "tight": [1.0, 2.0], "loose": [10.0], "x": 1}, ref))

    def test_wer_field_ignores_punctuation_and_bounds_word_changes(self):
        m = {"compare": "tolerance", "tolerance": {"abs": 0.0, "rel": 0.0, "fields": {"text": {"wer": 0.25}}}}
        ref = {"text": "one two three four"}
        self.assertTrue(pw.compare(m, {"text": "One, two three four."}, ref), "case and punctuation do not count")
        self.assertTrue(pw.compare(m, {"text": "one two three five"}, ref), "one word of four is 0.25")
        self.assertFalse(pw.compare(m, {"text": "one two six five"}, ref))
        self.assertFalse(pw.compare(m, {"text": 3}, ref), "a wer field must be text")

    def test_informational_keys_are_not_compared_but_reported(self):
        m = {"compare": "tolerance", "tolerance": {"abs": 0.0, "rel": 0.0}, "informational": ["gap"]}
        ref = {"ids": "1 2", "gap": 0.5}
        self.assertTrue(pw.compare(m, {"ids": "1 2", "gap": 0.01}, ref))
        self.assertTrue(pw.compare(m, {"ids": "1 2"}, ref), "an informational key may be absent")
        self.assertFalse(pw.compare(m, {"ids": "1 3", "gap": 0.5}, ref))
        self.assertEqual(pw.drop_informational(m, {"ids": "1 2", "gap": 0.01}, ref)[2], ["gap"])
        self.assertEqual(pw.drop_informational(m, {"ids": "1 2", "gap": 0.5}, ref)[2], [])

    def test_ort_int8_workloads_compare_task_output(self):
        """The measured bounds of docs/int8-and-nondeterminism.md, against the real manifests and references."""
        def load(name):
            w = ROOT / "workloads" / name
            return validate.load(w / "manifest.yaml"), json.loads((w / "reference.json").read_text())["output"]
        m, ref = load("moonshine-tiny-en-onnx")
        comma = dict(ref, text=ref["text"].replace("her parent forever", "her parent, forever"), min_logit_gap=0.008)
        self.assertTrue(pw.compare(m, comma, ref))
        self.assertTrue(pw.compare(m, dict(ref, text=ref["text"].replace("dishonoured", "dishonored")), ref))
        self.assertFalse(pw.compare(m, dict(ref, text=ref["text"].replace("dishonoured", "dishonored").replace("lovely", "lonely")), ref))
        m, ref = load("kokoro-tts-int8-onnx")
        short = dict(ref, n_samples=[94200, 70800], n_samples_exact=[94200, 70800])
        self.assertTrue(pw.compare(m, short, ref), "0.84 % fewer samples was measured on AVX2 hosts")
        self.assertFalse(pw.compare(m, dict(ref, n_samples=[94200, 69000]), ref))
        m, ref = load("onnx-zoo-bertsquad-int8")
        self.assertTrue(pw.compare(m, dict(ref, span_scores=[x + 0.5 for x in ref["span_scores"]]), ref))
        self.assertFalse(pw.compare(m, dict(ref, span_scores=[x - 18 for x in ref["span_scores"]]), ref))
        self.assertFalse(pw.compare(m, dict(ref, answers="ardmore | 90 | wick | alder rises"), ref))

    def test_bool_is_not_a_number(self):
        m = {"compare": "tolerance", "tolerance": {"abs": 5.0, "rel": 0.0}}
        self.assertFalse(pw.compare(m, True, 3))


class Cli(unittest.TestCase):
    def test_run_writes_results_files_and_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "r"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pw.main(["run", "selftest", "--target", "cpu", "--out", str(out)]), 0)
            row = (out / "results.tsv").read_text().split("\t")
            self.assertEqual(row[:2], ["selftest", "PASS"])
            self.assertEqual(json.loads((out / "selftest.json").read_text())["result"], "PASS")
            self.assertIn("cpu", (out / "machine.txt").read_text())
            with mock.patch.dict(os.environ, {"PW_SELFTEST_FAIL": "1"}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pw.main(["run", "selftest", "--target", "cpu", "--out", str(out)]), 1)

    def test_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label, result in (("cpu", "PASS"), ("sim:amd/mi300x", "FAIL")):
                d = pathlib.Path(tmp) / label.replace(":", "-").replace("/", "-")
                d.mkdir()
                (d / "results.tsv").write_text(f"selftest\t{result}\t0.1\tx\n")
                (d / "machine.txt").write_text(label + "\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(pw.main(["matrix", tmp]), 0)
            text = buf.getvalue()
            self.assertIn("✅", text)
            self.assertIn("❌", text)
            self.assertIn("sim:amd/mi300x", text)

    def test_record_writes_reference_and_restores_on_failure(self):
        ref = ROOT / "workloads" / "selftest" / "reference.json"
        before = ref.read_text()
        try:
            with mock.patch.dict(os.environ, {"PW_SELFTEST_OUTPUT": "recorded-value"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pw.main(["record", "selftest", "--target", "cpu"]), 0)
            self.assertEqual(json.loads(ref.read_text())["output"], "recorded-value")
            ref.write_text(before)
            with mock.patch.dict(os.environ, {"PW_SELFTEST_FAIL": "1"}), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(pw.main(["record", "selftest", "--target", "cpu"]), 1)
            self.assertEqual(ref.read_text(), before, "a failed record must leave the old reference alone")
        finally:
            ref.write_text(before)


class Bench(unittest.TestCase):
    def run_bench(self, argv, **env):
        with tempfile.TemporaryDirectory() as tmp:
            bench = pathlib.Path(tmp) / "bench"
            env = dict(env, PW_BENCH_DIR=str(bench), PW_SELFTEST_METRICS="1")
            err, out = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = pw.main(["run", "selftest", "--out", str(pathlib.Path(tmp) / "r")] + argv)
            files = sorted(bench.glob("*/*.json"))
            records = [json.loads(f.read_text()) for f in files]
            return code, records, out.getvalue() + err.getvalue(), files

    def test_library_versions_reported_by_the_workload_are_recorded(self):
        code, records, _, _ = self.run_bench(["--target", "gpu", "--bench", "--repeat", "2", "--device", "X"],
                                             PW_SELFTEST_VERSIONS='{"torch": "2.14.1+cu130", "numpy": "2.5.3"}')
        self.assertEqual(code, 0)
        env = records[0]["environment"]
        self.assertEqual(env["runtime_versions"], {"torch": "2.14.1+cu130", "numpy": "2.5.3"})
        self.assertEqual(env["runtime_version"], "numpy 2.5.3, torch 2.14.1+cu130")

    def test_no_reported_versions_leaves_the_manifest_runtime_version(self):
        _, records, _, _ = self.run_bench(["--target", "gpu", "--bench", "--repeat", "1", "--device", "X"])
        self.assertNotIn("runtime_versions", records[0]["environment"])

    def test_records_median_of_repeats_with_environment(self):
        code, records, _, files = self.run_bench(["--target", "gpu", "--bench", "--repeat", "3", "--device", "Test GPU 80GB"])
        self.assertEqual(code, 0)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["schema"], "pw-bench/0")
        self.assertEqual(rec["repeats"], 3)
        self.assertEqual(rec["metrics"]["tokens_per_s"]["median"], 1.0)
        self.assertEqual(len(rec["metrics"]["tokens_per_s"]["values"]), 3)
        self.assertEqual(rec["environment"]["device"], "Test GPU 80GB")
        self.assertIn("repo_commit", rec["environment"])
        self.assertIn("test-gpu-80gb", files[0].name)

    def test_refused_for_simulated_targets(self):
        code, records, text, _ = self.run_bench(["--target", "sim:nvidia/h100", "--bench"])
        self.assertEqual((code, records), (2, []))
        self.assertIn("real targets only", text)

    def test_unknown_device_is_not_recorded(self):
        with mock.patch.object(pw, "detect_device", return_value=(None, None)):
            code, records, text, _ = self.run_bench(["--target", "gpu", "--bench", "--repeat", "1"])
        self.assertEqual((code, records), (1, []))
        self.assertIn("--device", text)

    def test_no_metrics_is_not_recorded(self):
        with mock.patch.dict(os.environ, {"PW_SELFTEST_METRICS": ""}):
            with tempfile.TemporaryDirectory() as tmp:
                env = {"PW_BENCH_DIR": str(pathlib.Path(tmp) / "b"), "PW_SELFTEST_METRICS": ""}
                with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                    code = pw.main(["run", "selftest", "--target", "gpu", "--bench", "--repeat", "1",
                                    "--device", "X", "--out", str(pathlib.Path(tmp) / "r")])
                self.assertEqual(code, 1)
                self.assertEqual(list((pathlib.Path(tmp) / "b").glob("*/*.json")), [])

    def test_a_failing_repeat_stops_and_records_nothing(self):
        code, records, _, _ = self.run_bench(["--target", "gpu", "--bench", "--repeat", "3", "--device", "X"],
                                             PW_SELFTEST_OUTPUT="wrong")
        self.assertEqual((code, records), (1, []))

    def test_bench_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PW_BENCH_DIR": str(pathlib.Path(tmp) / "bench"), "PW_SELFTEST_METRICS": "1"}
            with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                pw.main(["run", "selftest", "--target", "gpu", "--bench", "--repeat", "2",
                         "--device", "Test GPU", "--out", str(pathlib.Path(tmp) / "r")])
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    self.assertEqual(pw.main(["bench-table"]), 0)
            text = buf.getvalue()
            self.assertIn("| selftest | Test GPU | tokens_per_s | 1.0 | 2 |", text)

    def test_bench_table_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pw.main(["bench-table", tmp]), 1)


class Manifests(unittest.TestCase):
    def check(self, text, name="demo", run_sh=True, reference=False):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / name
            d.mkdir()
            (d / "manifest.yaml").write_text(text)
            if run_sh:
                (d / "run.sh").write_text("#!/bin/sh\n")
            if reference:
                (d / "reference.json").write_text("{}")
            return validate.check(d / "manifest.yaml")

    GOOD = """
name: demo
description: d
kind: functional
runtime: ollama
targets: [cpu, "sim:nvidia/*"]
compare: exact
model: {id: m, revision: abc, licence: Apache-2.0, source_url: "https://example.org"}
"""

    def test_good_manifest(self):
        errors, warnings = self.check(self.GOOD, reference=True)
        self.assertEqual((errors, warnings), ([], []))

    def test_unpinned_model_and_missing_reference_warn(self):
        errors, warnings = self.check(self.GOOD.replace("revision: abc", "revision: null"))
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 2)

    def test_errors(self):
        for text, fragment in (
            (self.GOOD.replace("name: demo", "name: Other"), "lower-case"),
            (self.GOOD.replace("name: demo", "name: other"), "directory"),
            (self.GOOD.replace("kind: functional", "kind: fast"), "kind"),
            (self.GOOD.replace("compare: exact\n", ""), "compare"),
            (self.GOOD.replace('"sim:nvidia/*"', "tpu"), "target"),
            (self.GOOD.replace("licence: Apache-2.0, ", ""), "model.licence"),
            (self.GOOD.replace("model: {id: m, revision: abc, licence: Apache-2.0, source_url: \"https://example.org\"}\n", ""), "model"),
            ("- not a mapping", "mapping"),
            ("name: [", "YAML"),
        ):
            errors, _ = self.check(text, reference=True)
            self.assertTrue(any(fragment in e for e in errors), (fragment, errors))

    def test_model_null_means_no_model_but_absent_model_is_an_error(self):
        errors, _ = self.check(self.GOOD.replace('model: {id: m, revision: abc, licence: Apache-2.0, source_url: "https://example.org"}', "model: null"),
                               reference=True)
        self.assertEqual(errors, [])

    def test_missing_run_sh(self):
        errors, _ = self.check(self.GOOD, run_sh=False, reference=True)
        self.assertTrue(any("run.sh" in e for e in errors))

    def test_every_checked_in_manifest_is_valid(self):
        for manifest in sorted((ROOT / "workloads").glob("*/manifest.yaml")):
            errors, _ = validate.check(manifest)
            self.assertEqual(errors, [], manifest.parent.name)


if __name__ == "__main__":
    unittest.main()
