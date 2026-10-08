"""Tests for the lib-* library-coverage workloads and tools/pw-conda-torch.sh.

Tests that need PyTorch run it from PW_PYTHON (or python3 when that has torch) and are skipped otherwise:

    python3 -m unittest discover -s tests -v
"""
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import shutil
import stat
import subprocess
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

FUNCTIONAL = ["lib-fft-linalg", "lib-sparse-embedding", "lib-rnn-conv", "lib-attention-precision", "lib-composite-blocks"]
LIB = ROOT / "workloads" / "_lib"
CONDA = ROOT / "tools" / "pw-conda-torch.sh"


def torch_python():
    for cand in (os.environ.get("PW_PYTHON"), sys.executable, shutil.which("python3")):
        if cand and subprocess.run([cand, "-c", "import torch"], capture_output=True).returncode == 0:
            return cand
    return None


PY = torch_python()
needs_torch = unittest.skipUnless(PY, "no Python with PyTorch (set PW_PYTHON)")


def in_torch(code, **env):
    """Run `code` in the torch interpreter with _lib on the path; return stdout (raises on failure)."""
    full = {**os.environ, "PW_TARGET": "cpu", "PW_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", **env}
    p = subprocess.run([PY, "-c", f"import sys; sys.path.insert(0, {str(LIB)!r})\n{code}"], capture_output=True, text=True, env=full)
    if p.returncode:
        raise AssertionError(p.stderr[-2000:])
    return p.stdout


def run(name, target="cpu", **env):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
        return pw.execute(name, target, pathlib.Path(tmp))


class Manifests(unittest.TestCase):
    def test_manifests_are_valid_and_functional_ones_have_references(self):
        for name in FUNCTIONAL + ["lib-kernels-bench"]:
            errors, _ = validate.check(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(errors, [], name)
        for name in FUNCTIONAL:
            self.assertTrue((ROOT / "workloads" / name / "reference.json").exists(), name)

    def test_tolerance_fields_are_numbers(self):
        # PyYAML reads 2e-3 (no dot) as a string; bin/pw would then crash on it
        for name in FUNCTIONAL:
            m = validate.load(ROOT / "workloads" / name / "manifest.yaml")
            for op, f in m["tolerance"]["fields"].items():
                self.assertIsInstance(f["abs"], float, (name, op))
                self.assertIsInstance(f["rel"], float, (name, op))

    def test_benchmark_is_gpu_only_and_never_on_simulators(self):
        m = validate.load(ROOT / "workloads" / "lib-kernels-bench" / "manifest.yaml")
        self.assertEqual((m["kind"], m["targets"]), ("benchmark", ["gpu"]))
        for target in ("cpu", "sim:nvidia/h100", "sim:amd/mi300x"):
            self.assertEqual(run("lib-kernels-bench", target)[0], "SKIP", target)

    def test_references_record_their_recorder(self):
        for name in FUNCTIONAL:
            ref = json.loads((ROOT / "workloads" / name / "reference.json").read_text())
            self.assertEqual(ref["recorded_on"], "cpu", name)

    def test_no_torch_is_skip(self):
        for name in FUNCTIONAL:
            result, _, detail, _ = run(name, PW_PYTHON="/nonexistent/python", PW_CONDA_PREFIX="/nonexistent")
            self.assertEqual(result, "SKIP", name)
            self.assertIn("not found", detail)

    def test_sim_targets_need_pantheonsim(self):
        for target in ("sim:nvidia/h100", "sim:amd/mi300x"):
            result, _, detail, _ = run("lib-fft-linalg", target, PANTHEONSIM_DIR="")
            self.assertEqual(result, "SKIP", target)


@needs_torch
class Framework(unittest.TestCase):
    def test_checksums_and_classification(self):
        out = in_torch('''
import torch, libkernels as lk, json
y = torch.arange(1., 10.)
r = {"stats": lk.stats(y), "icheck": lk.icheck(torch.tensor([3, 1, 2])),
     "unsupported": [lk.is_unsupported(e) for e in (NotImplementedError("x"), RuntimeError("No available kernel."),
                     RuntimeError("Unsupported dtype Half"), RuntimeError("out of memory"), ValueError("shapes differ"))]}
print(json.dumps(r))''')
        r = json.loads(out)
        self.assertEqual(r["stats"], [5.0, 5.62731434, 1.0, 4.0, 9.0])      # mean |y|, rms = sqrt(285/9), y[0], y[n//3], y[-1]
        self.assertEqual(r["icheck"], "n=3 sum=6 chk=11 head=3 1 2")    # 3*1 + 1*2 + 2*3
        self.assertEqual(r["unsupported"], [True, True, True, False, False])

    def test_unsupported_wrong_and_crashing_ops_are_told_apart(self):
        out = in_torch('''
import torch, libkernels as lk, json
S = lk.Suite("t")
@S.op("ok", bound=1e-6)
def _(c): return c(torch.arange(6.)) * 2
@S.op("no_kernel")
def _(c): raise RuntimeError("No available kernel. Aborting execution.")
@S.op("wrong", bound=1e-3)
def _(c): return c(torch.arange(6.)) * (2.1 if not c.is_ref else 2.0)
@S.op("crash")
def _(c): raise RuntimeError("something else broke")
@S.op("ints", kind="int")
def _(c): return c.raw(torch.tensor([3, 1, 2])).sort().values
@S.op("ints_wrong", kind="int", ref=lambda c: torch.tensor([9, 9, 9]))
def _(c): return c.raw(torch.tensor([3, 1, 2]))
@S.op("check_fails")
def _(c): raise lk.Fail("dtype was not what autocast promises")
output, detail, failures = lk.run_suite(S)
print(json.dumps({"output": output, "failures": failures}))''')
        r = json.loads(out)
        self.assertEqual(r["output"]["no_kernel"], "unsupported")
        self.assertIsInstance(r["output"]["ok"], list)
        self.assertEqual(r["output"]["wrong"], "wrong")
        self.assertEqual(r["output"]["crash"], "error")
        self.assertTrue(r["output"]["ints"].startswith("n=3 sum=6"))
        names = sorted(f.split(":")[0] for f in r["failures"])
        self.assertEqual(names, ["check_fails", "crash", "ints_wrong", "wrong"])    # unsupported is not a failure

    def test_informational_ops_never_decide_pass_or_fail(self):
        out = in_torch('''
import torch, libkernels as lk, json
S = lk.Suite("t")
@S.op("differs", kind="info")
def _(c): return torch.tensor([0, 1]) if not c.is_ref else torch.tensor([0, 0])      # device and CPU disagree
_.labels = ("saturates", "nan"); _.inputs = [480.0, 500.0]
@S.op("crashes", kind="info")
def _(c): raise RuntimeError("No available kernel")                                  # even "unsupported" is not reported as such
@S.op("ok", bound=1e-6)
def _(c): return c(torch.arange(6.)) * 2
output, detail, failures = lk.run_suite(S)
print(json.dumps({"output": output, "failures": failures, "info": S.info, "detail": detail}))''')
        r = json.loads(out)
        self.assertEqual(r["failures"], [])
        self.assertEqual(r["output"]["differs"], "informational")
        self.assertEqual(r["output"]["crashes"], "informational")
        self.assertEqual(r["info"]["differs"], {"inputs": ["480.0", "500.0"], "device": ["saturates", "nan"], "cpu": ["saturates", "saturates"]})
        self.assertIn("error", r["info"]["crashes"])
        self.assertIn("2 informational", r["detail"])
        self.assertIn("1 checked", r["detail"])

    def test_fp8_cast_ops_are_stable_across_pytorch_versions(self):
        # PyTorch 2.10 casts out-of-range values to e4m3fn as NaN, 2.13 and newer saturate (docs/fp8-cast-semantics.md):
        # the numerics op must only see in-range values, and the overflow ops must be informational.
        out = in_torch('''
import torch, json, math, libkernels as lk
g = lk.load_group("lib-attention-precision")
ops = g.SUITE.ops
cls = g.overflow_classes(torch.tensor([480.0, -480.0, 5e4, math.inf, math.nan, 1.0]),
                         torch.tensor([448.0, -448.0, math.nan, math.inf, math.nan, 2.0]), 448.0)
x = g.CASTV.clamp(-448.0, 448.0)
print(json.dumps({"kinds": {k: o.kind for k, o in ops.items() if k.startswith("cast_overflow")}, "cls": cls, "max": float(x.abs().max()),
                  "e5m2_has_57344": bool((g.CASTV == 57344.0).any())}))''')
        r = json.loads(out)
        self.assertEqual(r["kinds"], {"cast_overflow_class_fp8_e4m3fn": "info", "cast_overflow_class_fp8_e5m2": "info"})
        self.assertEqual(r["cls"], ["saturates", "saturates", "nan", "inf", "nan", "other"])
        self.assertEqual(r["max"], 448.0)
        self.assertTrue(r["e5m2_has_57344"])
        ref = json.loads((ROOT / "workloads" / "lib-attention-precision" / "reference.json").read_text())["output"]
        self.assertEqual(ref["cast_overflow_class_fp8_e4m3fn"], "informational")
        self.assertEqual(ref["cast_overflow_class_fp8_e5m2"], "informational")

    def test_reference_context_rounds_inputs_like_the_run(self):
        out = in_torch('''
import torch, libkernels as lk
x = torch.tensor([1 + 2.0 ** -11, 1.0])         # a tie in fp16: rounds to 1.0
ref = lk.Ctx("cpu", torch.float64, round_to=torch.float16)(x)
run = lk.Ctx("cpu", torch.float16)(x)
print(ref.dtype, ref[0].item(), run.dtype, run[0].item())''')
        self.assertEqual(out.split(), ["torch.float64", "1.0", "torch.float16", "1.0"])

    def test_groups_register_unique_ops_matching_the_references(self):
        for name in FUNCTIONAL:
            out = in_torch(f'import libkernels as lk, json; print(json.dumps(list(lk.load_group({name!r}).SUITE.ops)))')
            ops = json.loads(out)
            self.assertEqual(len(ops), len(set(ops)))
            ref = json.loads((ROOT / "workloads" / name / "reference.json").read_text())["output"]
            self.assertEqual(sorted(ref), sorted(ops), name)

    def test_calibration_suggests_fields_for_half_precision_only(self):
        out = in_torch('''
import torch, libkernels as lk
S = lk.Suite("t")
@S.op("fp32_op")
def _(c): return c(torch.arange(1., 50.)) * 1.5
@S.op("fp16_op", dtype=torch.float16, bound=1e-2)
def _(c): return c(torch.arange(1., 50.)) / 3
print(lk.suggest(lk.calibrate(S)))''')
        fields = [l for l in out.splitlines() if l.startswith("    ")]
        self.assertEqual(len(fields), 1)
        self.assertIn("fp16_op: {abs:", fields[0])
        self.assertIn("rel: 5.0e-03", fields[0])


@needs_torch
class RunsOnCpu(unittest.TestCase):
    def test_functional_workloads_pass_against_their_references(self):
        for name in FUNCTIONAL:
            result, _, detail, rec = run(name, PW_PYTHON=PY, OMP_NUM_THREADS="2")
            self.assertEqual(result, "PASS", (name, detail))
            self.assertEqual(rec["metrics"], {})

    def test_a_changed_checksum_is_caught(self):
        name = "lib-sparse-embedding"
        ref = json.loads((ROOT / "workloads" / name / "reference.json").read_text())["output"]
        bad = json.loads(json.dumps(ref))
        bad["spmm_csr_dense"][1] *= 1.01              # rms off by 1%: far outside rel 1e-3
        manifest = validate.load(ROOT / "workloads" / name / "manifest.yaml")
        self.assertTrue(pw.compare(manifest, ref, ref))
        self.assertFalse(pw.compare(manifest, bad, ref))
        gone = dict(ref, spmm_csr_dense="unsupported")
        self.assertFalse(pw.compare(manifest, *pw.split_unsupported(manifest, gone, ref)[:2]))     # a limited backend fails on exactly that key

    def test_benchmark_code_path_runs_at_tiny_size(self):
        p = subprocess.run([PY, str(LIB / "libkernels.py"), "lib-composite-blocks", "--bench"], capture_output=True, text=True,
                           env=dict(os.environ, PW_TARGET="cpu", PW_DEVICE="cpu", CUDA_VISIBLE_DEVICES="", PW_LIB_SIZE="tiny", OMP_NUM_THREADS="2"))
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        rec = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(set(rec), {"output", "detail", "metrics"})
        self.assertTrue(rec["metrics"])
        self.assertTrue(all(v > 0 for v in rec["metrics"].values()))

    def test_metrics_are_not_reported_for_simulated_targets(self):
        out = in_torch('import libkernels as lk; lk.finish("x", "d", {"a": 1.0})', PW_TARGET="sim:amd/mi300x")
        self.assertEqual(json.loads(out)["metrics"], {})


class UncheckedOps(unittest.TestCase):
    def test_ops_the_reference_lacks_are_unchecked_not_failed(self):
        manifest = {"compare": "tolerance", "tolerance": {"abs": 1e-4, "rel": 1e-3}}
        ref = {"a": [1.0, 2.0], "b": "unsupported", "c": "unsupported", "d": [3.0]}
        out = {"a": [1.0, 2.0], "b": [9.0, 9.0], "c": "unsupported", "d": [3.0]}   # b: the target has a kernel, c: neither has
        o, r, unchecked = pw.split_unsupported(manifest, out, ref)
        self.assertEqual(unchecked, ["b"])
        self.assertTrue(pw.compare(manifest, o, r))
        o, r, _ = pw.split_unsupported(manifest, dict(out, a=[1.0, 2.5]), ref)      # a wrong checked op still fails
        self.assertFalse(pw.compare(manifest, o, r))
        # the reverse: a number in the reference, "unsupported" on the target, fails unless the manifest lists it
        lacking = dict(out, a="unsupported")
        self.assertFalse(pw.compare(manifest, *pw.split_unsupported(manifest, lacking, ref)[:2]))
        listed = dict(manifest, unsupported_ok=["a"])
        o, r, unchecked = pw.split_unsupported(listed, lacking, ref)
        self.assertTrue(pw.compare(listed, o, r))
        self.assertEqual(unchecked, ["b"])
        # a mapping lists the keys per target: a GPU generation that lacks the op, not every GPU
        per = dict(manifest, unsupported_ok={"sim:nvidia/t4": ["a"], "sim:nvidia/a100": ["d"]})
        o, r, _ = pw.split_unsupported(per, lacking, ref, "sim:nvidia/t4")
        self.assertTrue(pw.compare(per, o, r))
        o, r, _ = pw.split_unsupported(per, lacking, ref, "sim:nvidia/h100")
        self.assertFalse(pw.compare(per, o, r))
        o, r, _ = pw.split_unsupported(per, lacking, ref)
        self.assertFalse(pw.compare(per, o, r))


class CondaHelper(unittest.TestCase):
    def sh(self, **env):
        return subprocess.run(["bash", str(CONDA)], capture_output=True, text=True, env=dict(os.environ, **env))

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(CONDA)]).returncode, 0)

    def test_existing_environment_is_reused_without_a_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            py = pathlib.Path(tmp) / "envs" / "torch" / "bin" / "python"
            py.parent.mkdir(parents=True)
            py.write_text("#!/bin/sh\nexit 0\n")
            py.chmod(py.stat().st_mode | stat.S_IEXEC)
            p = self.sh(PW_CONDA_PREFIX=tmp, PATH="/nonexistent:/usr/bin:/bin")
            self.assertEqual((p.returncode, p.stdout.strip()), (0, str(py)))

    def test_creates_the_environment_from_conda_forge_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "args"
            mm = pathlib.Path(tmp) / "micromamba"
            mm.write_text(f'#!/bin/sh\necho "$@" > {log}\nmkdir -p "{tmp}/envs/torch/bin"\n'
                          f'printf "#!/bin/sh\\nexit 0\\n" > "{tmp}/envs/torch/bin/python"; chmod +x "{tmp}/envs/torch/bin/python"\n')
            mm.chmod(mm.stat().st_mode | stat.S_IEXEC)
            p = self.sh(PW_CONDA_PREFIX=tmp, PW_MICROMAMBA=str(mm))
            self.assertEqual(p.returncode, 0, p.stderr)
            args = log.read_text()
            for want in ("create -y -p", "-c conda-forge --override-channels", "pytorch=2.13.0=cpu*", "python=3.12", "numpy"):
                self.assertIn(want, args)
            self.assertNotIn("--dry-run", args)
            self.assertTrue(p.stdout.strip().endswith("envs/torch/bin/python"))

    def test_dry_run_installs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "args"
            mm = pathlib.Path(tmp) / "micromamba"
            mm.write_text(f'#!/bin/sh\necho "$@" > {log}\n')
            mm.chmod(mm.stat().st_mode | stat.S_IEXEC)
            p = self.sh(PW_CONDA_PREFIX=tmp, PW_MICROMAMBA=str(mm), PW_CONDA_DRY_RUN="1")
            self.assertEqual((p.returncode, p.stdout), (0, ""))
            self.assertIn("--dry-run", log.read_text())


if __name__ == "__main__":
    unittest.main()
