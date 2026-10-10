"""Tests for the optional `requires` manifest field: validation, the SKIP on a host that cannot meet it,
`list --runnable`, and multi-GPU bench records. A fake nvidia-smi (PW_NVIDIA_SMI) stands in for the host.

    python3 -m unittest test_requires -v
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import coverage_table as coverage  # noqa: E402
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw_requires", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw_requires", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

A10G = "NVIDIA A10G, 23028, 8.6, 550.54"
H100 = "NVIDIA H100 80GB HBM3, 81559, 9.0, 550.54"

MANIFEST = """name: {name}
description: d
kind: benchmark
runtime: none
targets: [{targets}]
{extra}
"""


def make(tmp, name, requires="", targets="gpu"):
    d = pathlib.Path(tmp) / "workloads" / name
    d.mkdir(parents=True)
    extra = "requires:\n" + "".join(f"  {line}\n" for line in requires.strip().splitlines()) if requires else ""
    (d / "manifest.yaml").write_text(MANIFEST.format(name=name, targets=targets, extra=extra))
    (d / "run.sh").write_text('echo \'{"output": "x", "metrics": {"m": 1}}\'\n')
    return d / "manifest.yaml"


def fake_smi(tmp, lines):
    """A fake nvidia-smi printing `lines` (None: the command exits 1, as on a host with no driver)."""
    path = pathlib.Path(tmp) / "nvidia-smi"
    body = "exit 1\n" if lines is None else "cat <<'EOF'\n" + "\n".join(lines) + "\nEOF\n"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def host(tmp, lines):
    """Patch the environment so bin/pw sees exactly these NVIDIA GPUs (and no AMD ones)."""
    empty = pathlib.Path(tmp) / "no-drm"
    empty.mkdir(exist_ok=True)
    return mock.patch.dict(os.environ, {"PW_NVIDIA_SMI": fake_smi(tmp, lines), "PW_ROCM_SMI": "/nonexistent/rocm-smi",
                                        "PW_DRM_ROOT": str(empty), "PW_IGNORE_REQUIRES": ""})


def run(tmp, name, target="gpu"):
    with mock.patch.object(pw, "WORKLOADS", pathlib.Path(tmp) / "workloads"):
        return pw.execute(name, target, pathlib.Path(tmp) / "scratch")


def out_of(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(*args)
    return code, buf.getvalue()


class Validation(unittest.TestCase):
    def errors(self, requires):
        with tempfile.TemporaryDirectory() as tmp:
            return validate.check(make(tmp, "w", requires))[0]

    def test_full_schema_is_valid(self):
        req = 'gpus: 8\ngpu_memory_gb: 80\nmin_compute_capability: "9.0"\nvendor: nvidia\ninterconnect: nvlink\nnotes: "70B"'
        self.assertEqual(self.errors(req), [])

    def test_errors(self):
        for bad, word in [("gpus: 0", "gpus"), ("gpus: 1.5", "gpus"), ("gpus: true", "gpus"),
                          ("gpu_memory_gb: -1", "gpu_memory_gb"), ("gpu_memory_gb: lots", "gpu_memory_gb"),
                          ("min_compute_capability: 9.0", "min_compute_capability"),
                          ('min_compute_capability: "nine"', "min_compute_capability"),
                          ("vendor: intel", "vendor"), ("bogus: 1", "unknown key"),
                          ('vendor: amd\nmin_compute_capability: "9.0"', "NVIDIA only"),
                          ("notes: ''", "notes")]:
            with self.subTest(bad=bad):
                self.assertTrue(any(word in e for e in self.errors(bad)), self.errors(bad))

    def test_requires_must_be_a_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = make(tmp, "w")
            m.write_text(m.read_text() + "requires: 8\n")
            self.assertTrue(any("mapping" in e for e in validate.check(m)[0]))

    def test_repository_manifests_stay_valid(self):
        for p in (ROOT / "workloads").glob("*/manifest.yaml"):
            self.assertEqual(validate.check(p)[0], [], p)


class Gate(unittest.TestCase):
    def result(self, lines, requires, target="gpu", **env):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, "w", requires)
            with host(tmp, lines), mock.patch.dict(os.environ, env):
                r = run(tmp, "w", target)
            return r[0], r[2]

    def test_met(self):
        self.assertEqual(self.result([A10G], "gpus: 1\ngpu_memory_gb: 22")[0], "PASS")

    def test_no_requires_runs_anywhere(self):
        self.assertEqual(self.result(None, "")[0], "PASS")

    def test_memory_short(self):
        result, detail = self.result([A10G], "gpu_memory_gb: 80")
        self.assertEqual(result, "SKIP")
        self.assertEqual(detail, "needs 1 GPU with >= 80 GiB each; this host has 1 x NVIDIA A10G 22.49 GiB")

    def test_marketing_size_matches_driver_size(self):
        # An 80 GB H100 reports 79.6 GiB; an A10G (22.5 GiB) is not a 24 GiB card.
        self.assertEqual(self.result([H100], "gpu_memory_gb: 80")[0], "PASS")
        self.assertEqual(self.result([A10G], "gpu_memory_gb: 24")[0], "SKIP")

    def test_gpu_count_short(self):
        result, detail = self.result([H100] * 4, 'gpus: 8\ngpu_memory_gb: 80\nmin_compute_capability: "9.0"')
        self.assertEqual(result, "SKIP")
        self.assertEqual(detail, "needs 8 GPUs with >= 80 GiB each (compute capability >= 9.0); "
                                 "this host has 4 x NVIDIA H100 80GB HBM3 79.65 GiB")

    def test_count_met_by_eight(self):
        self.assertEqual(self.result([H100] * 8, "gpus: 8\ngpu_memory_gb: 80")[0], "PASS")

    def test_capability_too_low(self):
        result, detail = self.result([A10G], 'min_compute_capability: "9.0"')
        self.assertEqual(result, "SKIP")
        self.assertIn("compute capability >= 9.0", detail)
        self.assertEqual(self.result([H100], 'min_compute_capability: "9.0"')[0], "PASS")
        self.assertEqual(self.result([H100], 'min_compute_capability: "10.0"')[0], "SKIP")

    def test_only_qualifying_cards_count(self):
        self.assertEqual(self.result([H100, A10G], "gpus: 2\ngpu_memory_gb: 80")[0], "SKIP")

    def test_vendor_mismatch(self):
        result, detail = self.result([H100], "vendor: amd")
        self.assertEqual(result, "SKIP")
        self.assertIn("(amd)", detail)

    def test_unknown_host_skips_not_fails(self):
        result, detail = self.result(None, "gpu_memory_gb: 16")
        self.assertEqual(result, "SKIP")
        self.assertTrue(detail.endswith("this host has no GPU detected"), detail)

    def test_override_env(self):
        self.assertEqual(self.result(None, "gpus: 8", PW_IGNORE_REQUIRES="1")[0], "PASS")

    def test_cpu_and_sim_targets_ignore_requires(self):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, "w", "gpus: 8\ngpu_memory_gb: 80", targets="cpu, sim:nvidia/*")
            with host(tmp, None):
                self.assertEqual(run(tmp, "w", "cpu")[0], "PASS")
                self.assertNotIn("needs", run(tmp, "w", "sim:nvidia/h100")[2])

    def test_amd_host_through_rocm_smi(self):
        with tempfile.TemporaryDirectory() as tmp:
            rocm = pathlib.Path(tmp) / "rocm-smi"
            card = {"Card Series": "Instinct MI300X", "VRAM Total Memory (B)": str(192 * 2**30)}
            rocm.write_text("#!/bin/sh\ncat <<'EOF'\n" + json.dumps({"card0": card, "card1": card}) + "\nEOF\n")
            rocm.chmod(0o755)
            make(tmp, "w", "vendor: amd\ngpus: 2\ngpu_memory_gb: 128")
            make(tmp, "w2", "vendor: amd\ngpus: 4")
            with host(tmp, None), mock.patch.dict(os.environ, {"PW_ROCM_SMI": str(rocm)}):
                self.assertEqual(run(tmp, "w")[0], "PASS")
                self.assertEqual(run(tmp, "w2")[0], "SKIP")


class Listing(unittest.TestCase):
    def test_list_shows_requires_and_runnable_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, "any-one")
            make(tmp, "big-one", "gpus: 8\ngpu_memory_gb: 80")
            make(tmp, "cpu-one", "gpus: 8", targets="cpu, gpu")
            with host(tmp, [A10G]), mock.patch.object(pw, "WORKLOADS", pathlib.Path(tmp) / "workloads"):
                _, text = out_of(pw.main, ["list"])
                self.assertIn("[needs 8 x 80 GiB]", text)
                _, text = out_of(pw.main, ["list", "--runnable"])
                self.assertIn("any-one", text)
                self.assertIn("cpu-one", text)
                self.assertNotIn("big-one", text)
            with host(tmp, [H100] * 8), mock.patch.object(pw, "WORKLOADS", pathlib.Path(tmp) / "workloads"):
                _, text = out_of(pw.main, ["list", "--runnable"])
                self.assertIn("big-one", text)


class Records(unittest.TestCase):
    def env(self, lines, requires, device=None):
        with tempfile.TemporaryDirectory() as tmp:
            m = validate.load(make(tmp, "w", requires))
            with host(tmp, lines):
                return pw.collect_env(m, "gpu", device)

    def test_eight_gpu_device_string(self):
        env = self.env([H100] * 8, "gpus: 8\ngpu_memory_gb: 80")
        self.assertEqual(env["device"], "8x NVIDIA H100 80GB HBM3")
        self.assertEqual(env["gpu_count"], 8)
        self.assertEqual(len(env["devices"]), 8)
        self.assertEqual(env["devices"][0], {"name": "NVIDIA H100 80GB HBM3", "memory_gib": 79.65})

    def test_single_gpu_record_unchanged(self):
        for requires in ("", "gpu_memory_gb: 20"):
            env = self.env([A10G], requires)
            self.assertEqual(env["device"], "NVIDIA A10G")
            self.assertNotIn("gpu_count", env)
            self.assertNotIn("devices", env)

    def test_requirement_of_few_gpus_on_a_big_host_records_those_used(self):
        env = self.env([H100] * 8, "gpus: 2")
        self.assertEqual((env["device"], env["gpu_count"]), ("2x NVIDIA H100 80GB HBM3", 2))

    def test_device_flag_still_wins(self):
        self.assertEqual(self.env([H100] * 8, "gpus: 8", device="my box")["device"], "my box")


class Coverage(unittest.TestCase):
    def test_needs_column(self):
        self.assertEqual(validate.needs_text(None), "any")
        self.assertEqual(validate.needs_text(validate.requirements({"requires": {"gpus": 8, "gpu_memory_gb": 80}})), "8 x 80 GiB")
        self.assertEqual(validate.needs_text(validate.requirements({"requires": {"gpu_memory_gb": 24}})), "1 x 24 GiB")
        with tempfile.TemporaryDirectory() as tmp:
            make(tmp, "big", "gpus: 8\ngpu_memory_gb: 80")
            make(tmp, "plain")
            (pathlib.Path(tmp) / "docs").mkdir()
            text = coverage.render(tmp)
            self.assertIn("| Needs |", text)
            self.assertIn("| 8 x 80 GiB |", text)
            self.assertIn("| any |", text)


if __name__ == "__main__":
    unittest.main()
