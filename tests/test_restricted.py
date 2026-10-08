"""Tests for the `restricted: true` manifest field: allowed everywhere by name, left out of default selections.

    python3 -m unittest test_restricted -v
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import coverage_table as coverage  # noqa: E402
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw_restricted", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw_restricted", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

BASE = """name: {name}
description: d
kind: functional
runtime: llama.cpp
targets: [gpu]
compare: exact
{extra}model: {{id: m, revision: abc, licence: "Llama 3.2 Community Licence", source_url: "https://example.org"}}
"""


def make(tmp, name, extra=""):
    d = pathlib.Path(tmp) / "workloads" / name
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(BASE.format(name=name, extra=extra))
    (d / "run.sh").write_text("#!/bin/sh\n")
    (d / "reference.json").write_text('{"output": "x", "recorded_on": "gpu"}')
    return d / "manifest.yaml"


def out_of(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(*args)
    return code, buf.getvalue()


class Restricted(unittest.TestCase):
    def test_field_must_be_bool_and_documented(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = make(tmp, "a", 'restricted: "yes"\nnotes: n\n')
            self.assertTrue(any("restricted" in e for e in validate.check(bad)[0]))
            nonotes = make(tmp, "b", "restricted: true\n")
            self.assertTrue(any("notes" in e for e in validate.check(nonotes)[0]))
            good = make(tmp, "c", "restricted: true\nnotes: gated; licence text read\n")
            self.assertEqual(validate.check(good), ([], []))

    def test_restricted_needs_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = make(tmp, "d", "restricted: true\nnotes: n\n")
            m.write_text(m.read_text().split("model:")[0] + "model: null\n")
            self.assertTrue(any("needs a 'model'" in e for e in validate.check(m)[0]))

    def test_is_restricted(self):
        self.assertTrue(validate.is_restricted({"restricted": True}))
        self.assertFalse(validate.is_restricted({"restricted": False}))
        self.assertFalse(validate.is_restricted({}))
        self.assertFalse(validate.is_restricted(None))

    def test_default_selection_skips_restricted_but_names_and_validate_keep_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, "open-one")
            make(tmp, "gated-one", "restricted: true\nnotes: gated\n")
            with mock.patch.object(pw, "WORKLOADS", pathlib.Path(tmp) / "workloads"):
                self.assertEqual(pw.names(), ["gated-one", "open-one"])
                self.assertEqual(pw.default_names(), ["open-one"])
                _, text = out_of(pw.main, ["list"])
                self.assertIn("gated-one", text)
                self.assertIn("[restricted]", text)
                _, text = out_of(pw.main, ["list", "--default"])
                self.assertNotIn("gated-one", text)
                code, text = out_of(pw.main, ["validate"])
                self.assertEqual(code, 0)
                self.assertIn("ok    gated-one", text)

    def test_matrix_hides_restricted_unless_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, "open-one")
            make(tmp, "gated-one", "restricted: true\nnotes: gated\n")
            res = pathlib.Path(tmp) / "res"
            res.mkdir()
            (res / "results.tsv").write_text("open-one\tPASS\t1\tx\ngated-one\tPASS\t1\tx\n")
            with mock.patch.object(pw, "WORKLOADS", pathlib.Path(tmp) / "workloads"):
                _, text = out_of(pw.main, ["matrix", str(res)])
                self.assertIn("open-one", text)
                self.assertNotIn("gated-one", text)
                _, text = out_of(pw.main, ["matrix", str(res), "--include-restricted"])
                self.assertIn("gated-one", text)

    def test_coverage_marks_restricted(self):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, "gated-one", "restricted: true\nnotes: gated\n")
            (pathlib.Path(tmp) / "docs").mkdir()
            self.assertIn("(restricted)", coverage.render(tmp))

    def test_ci_workflows_do_not_expand_all_workloads(self):
        # Workflows must name their workloads; none may feed `pw list` into a run.
        for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
            self.assertNotIn("pw list", wf.read_text(), wf.name)


if __name__ == "__main__":
    unittest.main()
