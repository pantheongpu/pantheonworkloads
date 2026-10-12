"""Tests for heavy workloads (docs/manifest.md): left out of implicit selections, always run when named.

    python3 -m unittest test_heavy -v
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import os
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw_heavy", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw_heavy", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

BASE = """name: {name}
description: d
kind: benchmark
runtime: none
targets: [cpu]
{extra}"""

ALL = ("small", "fits", "big-requires", "big-flag", "both")


def make(tmp, name, extra=""):
    d = pathlib.Path(tmp) / "workloads" / name
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(BASE.format(name=name, extra=extra))
    (d / "run.sh").write_text('echo \'{"output": "x", "metrics": {"m": 1}}\'\n')
    return d / "manifest.yaml"


def req(gpus, mem):
    return f"requires:\n  gpus: {gpus}\n  gpu_memory_gb: {mem}\n"


def out_of(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = fn(*args)
    return code, buf.getvalue()


def ran(out):
    return [line.split("\t")[0] for line in (pathlib.Path(out) / "results.tsv").read_text().splitlines()]


@contextlib.contextmanager
def repo(include_heavy_env=None):
    """A throwaway workloads tree; PW_INCLUDE_HEAVY is set to `include_heavy_env`, or removed when None."""
    with tempfile.TemporaryDirectory() as tmp:
        make(tmp, "small")                                            # no requires
        make(tmp, "fits", req(2, 80))                                 # 160 GiB: not over the line
        make(tmp, "big-requires", req(4, 80))                         # 320 GiB
        make(tmp, "big-flag", "heavy: true\nnotes: 200 GiB of weights\n" + req(1, 80))
        make(tmp, "both", "restricted: true\nheavy: true\nnotes: gated, huge\n" + req(8, 80))
        res = pathlib.Path(tmp) / "res"
        res.mkdir()
        (res / "results.tsv").write_text("".join(f"{n}\tPASS\t1\tx\n" for n in ALL))
        env = {k: v for k, v in os.environ.items() if k != "PW_INCLUDE_HEAVY"}
        if include_heavy_env is not None:
            env["PW_INCLUDE_HEAVY"] = include_heavy_env
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(pw, "WORKLOADS", pathlib.Path(tmp) / "workloads"):
            yield tmp, res


class IsHeavy(unittest.TestCase):
    def test_by_requires_total(self):
        self.assertFalse(validate.is_heavy({"requires": {"gpus": 2, "gpu_memory_gb": 80}}))      # 160, not over
        self.assertTrue(validate.is_heavy({"requires": {"gpus": 2, "gpu_memory_gb": 81}}))
        self.assertTrue(validate.is_heavy({"requires": {"gpus": 8, "gpu_memory_gb": 80}}))
        self.assertTrue(validate.is_heavy({"requires": {"gpu_memory_gb": 192}}))

    def test_by_flag(self):
        self.assertTrue(validate.is_heavy({"heavy": True}))
        self.assertTrue(validate.is_heavy({"heavy": True, "requires": {"gpus": 1, "gpu_memory_gb": 24}}))
        self.assertFalse(validate.is_heavy({"heavy": False, "requires": {"gpus": 1, "gpu_memory_gb": 24}}))

    def test_not_heavy(self):
        for m in ({}, None, {"requires": {"gpus": 8}}, {"requires": {"vendor": "amd"}}, {"heavy": "yes"}):
            self.assertFalse(validate.is_heavy(m), m)


class Validation(unittest.TestCase):
    def test_heavy_must_be_bool_with_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = make(tmp, "a", 'heavy: "yes"\nnotes: n\n')
            self.assertTrue(any("heavy must be" in e for e in validate.check(bad)[0]))
            nonotes = make(tmp, "b", "heavy: true\n")
            self.assertTrue(any("heavy: true needs 'notes'" in e for e in validate.check(nonotes)[0]))
            good = make(tmp, "c", "heavy: true\nnotes: 200 GiB of weights\n")
            self.assertEqual(validate.check(good), ([], []))
            off = make(tmp, "d", "heavy: false\n")
            self.assertEqual(validate.check(off), ([], []))


class Selection(unittest.TestCase):
    def test_default_names_skip_heavy_and_restricted(self):
        with repo():
            self.assertEqual(pw.names(), sorted(ALL))
            self.assertEqual(pw.default_names(), ["fits", "small"])

    def test_include_heavy_argument_and_env(self):
        with repo():
            self.assertEqual(pw.default_names(include_heavy=True), ["big-flag", "big-requires", "fits", "small"])
        with repo("1"):
            self.assertEqual(pw.default_names(), ["big-flag", "big-requires", "fits", "small"])
        with repo("0"):
            self.assertEqual(pw.default_names(), ["fits", "small"])

    def test_list(self):
        with repo():
            _, text = out_of(pw.main, ["list"])
            self.assertEqual(len(text.splitlines()), 5)
            self.assertRegex(text, r"big-requires.*\[heavy\]")
            self.assertRegex(text, r"big-flag.*\[heavy\]")
            self.assertRegex(text, r"both.*\[restricted\] \[heavy\]")
            self.assertNotRegex(text, r"small.*\[heavy\]")
            _, text = out_of(pw.main, ["list", "--default"])
            self.assertEqual([l.split("\t")[0] for l in text.splitlines()], ["fits", "small"])
            _, text = out_of(pw.main, ["list", "--default", "--include-heavy"])
            self.assertEqual([l.split("\t")[0] for l in text.splitlines()], ["big-flag", "big-requires", "fits", "small"])
        with repo("1"):
            _, text = out_of(pw.main, ["list", "--default"])
            self.assertIn("big-flag", text)

    def test_run_all_skips_heavy_but_a_name_runs_it(self):
        with repo() as (tmp, _):
            code, _ = out_of(pw.main, ["run", "--all", "--target", "cpu", "--out", tmp + "/o1"])
            self.assertEqual(code, 0)
            self.assertEqual(ran(tmp + "/o1"), ["fits", "small"])
            out_of(pw.main, ["run", "--all", "--include-heavy", "--target", "cpu", "--out", tmp + "/o2"])
            self.assertEqual(ran(tmp + "/o2"), ["big-flag", "big-requires", "fits", "small"])
            out_of(pw.main, ["run", "big-requires", "both", "--target", "cpu", "--out", tmp + "/o3"])
            self.assertEqual(ran(tmp + "/o3"), ["big-requires", "both"])
        with repo("1") as (tmp, _):
            out_of(pw.main, ["run", "--all", "--target", "cpu", "--out", tmp + "/o4"])
            self.assertEqual(ran(tmp + "/o4"), ["big-flag", "big-requires", "fits", "small"])

    def test_run_needs_names_or_all(self):
        with repo() as (tmp, _):
            self.assertEqual(out_of(pw.main, ["run", "--target", "cpu", "--out", tmp + "/o"])[0], 2)
            self.assertEqual(out_of(pw.main, ["run", "small", "--all", "--target", "cpu", "--out", tmp + "/o"])[0], 2)

    def test_matrix(self):
        with repo() as (_, res):
            _, text = out_of(pw.main, ["matrix", str(res)])
            self.assertIn("| small ", text)
            for n in ("big-flag", "big-requires", "both"):
                self.assertNotIn(f"| {n} ", text)
            _, text = out_of(pw.main, ["matrix", str(res), "--include-heavy"])
            self.assertIn("| big-flag ", text)
            self.assertNotIn("| both ", text)       # still restricted
            _, text = out_of(pw.main, ["matrix", str(res), "--include-heavy", "--include-restricted"])
            self.assertIn("| both ", text)
            _, text = out_of(pw.main, ["matrix", str(res), "--include-restricted"])
            self.assertNotIn("| both ", text)       # still heavy
            self.assertIn("| small ", text)


class Workflows(unittest.TestCase):
    def test_workflows_name_their_workloads(self):
        # No workflow may run `pw run --all` or expand `pw list` into a run: they name what they run, so a heavy
        # workload never enters CI by accident.
        for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
            text = wf.read_text()
            self.assertNotRegex(text, r"pw run[^\n]*--all", wf.name)
            self.assertNotIn("pw list", text, wf.name)
            self.assertNotIn("--include-heavy", text, wf.name)
            self.assertNotIn("PW_INCLUDE_HEAVY", text, wf.name)

    def test_workflow_workloads_are_not_heavy(self):
        names = set(pw.names())
        mentioned = set()
        for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
            mentioned |= {t for t in re.findall(r"[a-z][a-z0-9-]+", wf.read_text()) if t in names}
        self.assertTrue(mentioned)
        for n in mentioned:
            m = validate.load(ROOT / "workloads" / n / "manifest.yaml")
            self.assertFalse(validate.is_heavy(m), f"{n} is heavy but a workflow runs it")


class Repository(unittest.TestCase):
    def test_default_names_exclude_every_heavy_workload(self):
        with mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "PW_INCLUDE_HEAVY"}, clear=True):
            default = set(pw.default_names())
        for n in pw.names():
            if validate.is_heavy(validate.load(ROOT / "workloads" / n / "manifest.yaml")):
                self.assertNotIn(n, default, n)

    def test_weights_over_150_gib_are_heavy(self):
        # The catalog generator states each workload's weights size in its manifest.
        for p in sorted((ROOT / "workloads").glob("*/manifest.yaml")):
            m = re.search(r"weights ([0-9.]+) GiB", p.read_text())
            if m and float(m.group(1)) > 150:
                self.assertTrue(validate.is_heavy(validate.load(p)), p.parent.name)


if __name__ == "__main__":
    unittest.main()
