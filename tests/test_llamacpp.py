"""Tests for the llama.cpp workloads' helpers. No network, no GPU, no llama.cpp build.

    python3 -m unittest discover -s tests -v
"""
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
LC = ROOT / "tools" / "llamacpp"
sys.path.insert(0, str(LC))
sys.path.insert(0, str(ROOT / "tools"))
import bench_parse  # noqa: E402
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

WORKLOADS = ["llamacpp-smollm2-135m", "llamacpp-bench-smollm2-135m", "llamacpp-bench-mistral-7b-v03"]

# The shape llama-bench -o json prints (field names from tools/llama-bench/llama-bench.cpp).
SAMPLE = [
    {"model_type": "x", "n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ns": 1, "avg_ts": 4000.123, "stddev_ts": 1.0},
    {"model_type": "x", "n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ns": 1, "avg_ts": 250.5, "stddev_ts": 0.5},
]


class BenchParse(unittest.TestCase):
    def test_pp_and_tg(self):
        self.assertEqual(bench_parse.metrics(SAMPLE),
                         {"pp512_tokens_per_s": 4000.12, "tg128_tokens_per_s": 250.5})

    def test_log_lines_before_json_are_ignored(self):
        text = "load_backend: loaded CPU\n" + json.dumps(SAMPLE)
        self.assertEqual(set(bench_parse.parse(text)), {"pp512_tokens_per_s", "tg128_tokens_per_s"})

    def test_depth_and_combined_tests_get_distinct_keys(self):
        rows = [{"n_prompt": 0, "n_gen": 32, "n_depth": 1024, "avg_ts": 1.0},
                {"n_prompt": 64, "n_gen": 32, "avg_ts": 2.0}]
        self.assertEqual(bench_parse.metrics(rows), {"tg32_d1024_tokens_per_s": 1.0, "pp64_tg32_tokens_per_s": 2.0})

    def test_empty_or_missing_json_is_an_error(self):
        with self.assertRaises(ValueError):
            bench_parse.metrics([])
        with self.assertRaises(ValueError):
            bench_parse.parse("no json here")


PINNED = {"llamacpp-qwen25-0p5b": True, "llamacpp-bench-qwen25-0p5b": False,
          "llamacpp-mistral-7b-v03": True, "llamacpp-bench-mistral-7b-v03": False}   # name -> functional
# The 2026-10-09 additions (docs/models.md, "More llama.cpp architectures"): same checks as PINNED.
for _slug in ("qwen3-30b-a3b", "qwen3-8b", "phi4-14b", "granite40-h-small", "mistral-small-32-24b",
              "olmo2-7b-instruct", "qwen3-embedding-4b"):
    PINNED["llamacpp-" + _slug] = True
    PINNED["llamacpp-bench-" + _slug] = False


class PinnedModels(unittest.TestCase):
    """Real-model workloads: licence read, revision pinned, sha256 consistent between manifest, model.sha256 and run.sh."""

    def test_pins_are_consistent(self):
        import re
        for name, functional in PINNED.items():
            d = ROOT / "workloads" / name
            errors, warnings = validate.check(d / "manifest.yaml")
            self.assertEqual(errors, [], name)
            self.assertFalse(any("UNVERIFIED" in w for w in warnings), name)
            m = validate.load(d / "manifest.yaml")
            sha = (d / "model.sha256").read_text().strip()
            self.assertRegex(sha, r"^[0-9a-f]{64}$", name)
            self.assertIn(sha, m["model"]["revision"], name)
            rev = re.match(r"([0-9a-f]{40}) ", m["model"]["revision"])
            self.assertTrue(rev, name)
            self.assertIn("/resolve/" + rev.group(1) + "/", (d / "run.sh").read_text(), name)
            self.assertNotIn("UNVERIFIED", str(m["model"]["licence"]), name)
            if functional:
                ref = json.loads((d / "reference.json").read_text())
                self.assertEqual(ref["recorded_on"], "gpu", name)
                self.assertTrue(ref["output"], name)


class Manifests(unittest.TestCase):
    def test_valid_and_licence_never_silently_asserted(self):
        for name in WORKLOADS:
            errors, warnings = validate.check(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(errors, [], name)
            m = validate.load(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(m["runtime"], "llama.cpp")
            # A licence is either read from the model's card (not the case while huggingface.co is
            # unreachable) or flagged: either way validate must say so when it is unverified.
            if str(m["model"]["licence"]).startswith("UNVERIFIED"):
                self.assertTrue(any("UNVERIFIED" in w for w in warnings), name)

    def test_benchmarks_are_real_targets_only(self):
        for name in WORKLOADS[1:]:
            m = validate.load(ROOT / "workloads" / name / "manifest.yaml")
            self.assertEqual(m["kind"], "benchmark")
            self.assertFalse(any(t.startswith("sim:") for t in m["targets"]), name)

    def test_unverified_licence_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "demo"
            d.mkdir()
            (d / "run.sh").write_text("#!/bin/sh\n")
            (d / "manifest.yaml").write_text(
                'name: demo\ndescription: d\nkind: benchmark\nruntime: llama.cpp\ntargets: [gpu]\n'
                'model: {id: m, revision: abc, licence: "UNVERIFIED (x)", source_url: "https://e.org"}\n')
            errors, warnings = validate.check(d / "manifest.yaml")
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("UNVERIFIED", warnings[0])

    def test_pinned_runtime_version_is_recorded(self):
        m = validate.load(ROOT / "workloads" / WORKLOADS[1] / "manifest.yaml")
        with mock.patch.object(pw, "detect_device", return_value=("X", None)):
            env = pw.collect_env(m, "gpu", None)
        self.assertIn("b11447", env["runtime_version"])


class Scripts(unittest.TestCase):
    def test_shell_syntax(self):
        files = [LC / "build.sh", LC / "common.sh", LC / "bench.sh"] + [ROOT / "workloads" / n / "run.sh" for n in WORKLOADS + ["llamacpp-qwen3-embedding-4b"]]
        for f in files:
            p = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, f"{f}: {p.stderr}")

    def test_pin_is_a_full_commit(self):
        text = (LC / "pin.env").read_text()
        commit = [l.split("=", 1)[1] for l in text.splitlines() if l.startswith("LLAMACPP_PINNED_COMMIT=")][0]
        self.assertRegex(commit, r"^[0-9a-f]{40}$")

    def run_workload(self, name, target, **env):
        """Run a workload's run.sh in an empty environment: it must skip (77), never fail."""
        with tempfile.TemporaryDirectory() as tmp:
            e = {"PATH": os.environ["PATH"], "HOME": tmp, "PW_TARGET": target, "PW_OUT": tmp,
                 "PW_WORKLOAD_DIR": str(ROOT / "workloads" / name), "PW_CACHE": tmp + "/cache",
                 "PW_LLAMACPP_NO_BUILD": "1"}
            e.update(env)
            return subprocess.run(["bash", str(ROOT / "workloads" / name / "run.sh")], cwd=ROOT, env=e,
                                  capture_output=True, text=True, timeout=60)

    def test_skips_without_a_binary(self):
        for name in WORKLOADS:
            p = self.run_workload(name, "cpu")
            self.assertEqual(p.returncode, 77, (name, p.stdout, p.stderr))
            self.assertIn("SKIP", p.stdout)

    def test_gpu_target_without_a_gpu_skips(self):
        p = self.run_workload(WORKLOADS[1], "gpu", PATH="/usr/bin:/bin")
        self.assertEqual(p.returncode, 77, (p.stdout, p.stderr))

    def test_embedding_workload_with_a_fake_llama_embedding(self):
        """The embedding check: four unit vectors in, cosines and the passage ranking out; a build without the tool skips."""
        name = "llamacpp-qwen3-embedding-4b"
        vecs = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]    # query 0: passage 1 is nearest, then 3, then 2
        with tempfile.TemporaryDirectory() as tmp:
            bindir = pathlib.Path(tmp) / "prefix" / "bin"
            bindir.mkdir(parents=True)
            (bindir / "llama-completion").write_text("#!/bin/sh\n")
            (bindir / "llama-bench").write_text("#!/bin/sh\n")
            for f in bindir.iterdir():
                f.chmod(0o755)
            model = pathlib.Path(tmp) / "m.gguf"
            model.write_bytes(b"not a model")
            env = {"PW_LLAMACPP_BIN_DIR": str(bindir), "PW_MODEL_FILE": str(model)}
            p = self.run_workload(name, "cpu", **env)
            self.assertEqual(p.returncode, 77, (p.stdout, p.stderr))
            self.assertIn("llama-embedding", p.stdout)
            (bindir / "llama-embedding").write_text("#!/bin/sh\nprintf 'log line\\n%s\\n' '" + json.dumps(vecs) + "'\n")
            (bindir / "llama-embedding").chmod(0o755)
            p = self.run_workload(name, "cpu", **env)
            self.assertEqual(p.returncode, 0, (p.stdout, p.stderr))
            out = json.loads(p.stdout.splitlines()[-1])["output"]
            self.assertEqual(out["dim"], 2)
            self.assertEqual(out["query_ranking"], "1 3 2")
            self.assertEqual(out["cosine"]["0-1"], 0.8)
            self.assertTrue(out["finite"])

    def test_unsupported_target_skips(self):
        self.assertEqual(self.run_workload(WORKLOADS[0], "tpu").returncode, 77)

    def test_fake_binaries_drive_the_whole_contract(self):
        """A stub llama-completion and llama-bench stand in for the real ones."""
        with tempfile.TemporaryDirectory() as tmp:
            bindir = pathlib.Path(tmp) / "prefix" / "bin"
            bindir.mkdir(parents=True)
            (bindir / "llama-completion").write_text("#!/bin/sh\nprintf ' Paris'\n")
            (bindir / "llama-bench").write_text("#!/bin/sh\nprintf 'log line\\n%s\\n' '" + json.dumps(SAMPLE) + "'\n")
            for f in bindir.iterdir():
                f.chmod(0o755)
            model = pathlib.Path(tmp) / "m.gguf"
            model.write_bytes(b"not a model")
            env = {"PW_LLAMACPP_BIN_DIR": str(bindir), "PW_MODEL_FILE": str(model)}
            p = self.run_workload(WORKLOADS[0], "cpu", **env)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout.splitlines()[-1])["output"], " Paris")
            p = self.run_workload(WORKLOADS[1], "cpu", **env)
            self.assertEqual(p.returncode, 0, p.stderr)
            rec = json.loads(p.stdout.splitlines()[-1])
            self.assertEqual(rec["metrics"]["tg128_tokens_per_s"], 250.5)
            # a wrong checksum is a failure, not a skip
            p = self.run_workload(WORKLOADS[1], "cpu", PW_MODEL_SHA256="0" * 64, **env)
            self.assertEqual(p.returncode, 1)
            # benchmarks refuse simulated targets
            p = self.run_workload(WORKLOADS[1], "sim:nvidia/h100", PANTHEONSIM_DIR=tmp, **env)
            self.assertEqual(p.returncode, 77)


FAKE_HIPCC = """#!/bin/sh
echo "HIP version: %(ver)s"
echo "%(vendor)s clang version %(clang)s.0.6"
echo "Target: x86_64-pc-linux-gnu"
echo "InstalledDir: %(inst)s"
"""


def fake_rocm(root, ver, clang, layout):
    """A ROCm tree that prints what `hipcc --version` prints. layout: 'ubuntu' or 'amd'."""
    root = pathlib.Path(root)
    if layout == "ubuntu":
        llvm = root / "usr" / "lib" / f"llvm-{clang}" / "bin"
        hipcc = root / "usr" / "bin" / "hipcc"
        cmake = root / "usr" / "lib" / "x86_64-linux-gnu" / "cmake" / "hip-lang"
        inst = root / "usr" / "bin"
        top = root / "usr"
    else:
        top = root / f"rocm-{ver}"
        llvm = top / "lib" / "llvm" / "bin"
        hipcc = top / "bin" / "hipcc"
        cmake = top / "lib" / "cmake" / "hip-lang"
        inst = llvm
    for d in (llvm, hipcc.parent, cmake, inst):
        d.mkdir(parents=True, exist_ok=True)
    for n in ("clang", "clang++"):
        (llvm / n).write_text("#!/bin/sh\necho 'clang version %s.0.6'\n" % clang)
        (llvm / n).chmod(0o755)
    if layout == "ubuntu":
        # Debian: /usr/bin/clang++ is another LLVM than hipcc's, and clang++-N names the right one
        other = root / "usr" / "lib" / "llvm-99" / "bin"
        other.mkdir(parents=True)
        for n in ("clang", "clang++"):
            (other / n).write_text("#!/bin/sh\necho 'clang version 99.0.0'\n")
            (other / n).chmod(0o755)
        (inst / "clang++").symlink_to(other / "clang++")
        (inst / f"clang++-{clang}").symlink_to(llvm / "clang++")
    (cmake / "hip-lang-config.cmake").write_text("")
    hipcc.write_text(FAKE_HIPCC % {"ver": ver + ".31921-0", "clang": clang, "inst": inst,
                                   "vendor": "Ubuntu" if layout == "ubuntu" else "AMD"})
    hipcc.chmod(0o755)
    return top, hipcc


def hip_detect(hipcc):
    p = subprocess.run(["bash", "-c", '. "$1"; hip_detect; echo "$? $HIP_ROCM $HIP_CLANGXX $HIP_CMAKE_LIB $HIP_MAJOR.$HIP_MINOR '
                        '$HIP_COMPAT $HIP_WHY"', "x", str(LC / "hip-config.sh")],
                       env={"PATH": "/usr/bin:/bin", "HIPCC": str(hipcc)}, capture_output=True, text=True)
    return p.stdout.split()


class HipBuild(unittest.TestCase):
    """The hip backend of tools/llamacpp/build.sh, found by hip-config.sh, and what it needs from ROCm 5.x."""

    def test_shell_syntax(self):
        for f in ("build.sh", "hip-config.sh", "hip-apt-prefix.sh", "common.sh"):
            p = subprocess.run(["bash", "-n", str(LC / f)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, f"{f}: {p.stderr}")

    def test_ubuntu_rocm57_layout_needs_the_compat_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            top, hipcc = fake_rocm(tmp, "5.7", "17", "ubuntu")
            status, rocm, cxx, cmakelib, ver, compat = hip_detect(hipcc)[:6]
            self.assertEqual((status, ver, compat), ("0", "5.7", "1"))
            self.assertEqual(rocm, str(top))
            self.assertEqual(cxx, str(top / "lib" / "llvm-17" / "bin" / "clang++"))   # not Debian's default LLVM
            self.assertEqual(cmakelib, str(top / "lib" / "x86_64-linux-gnu"))

    def test_amd_layout_and_rocm_61_or_later_need_no_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ver, compat in (("6.0", "1"), ("6.1", "0"), ("6.4", "0"), ("7.0", "0")):
                top, hipcc = fake_rocm(pathlib.Path(tmp) / ver, ver, "19", "amd")
                out = hip_detect(hipcc)
                self.assertEqual((out[0], out[4], out[5]), ("0", ver, compat), out)
                self.assertEqual(out[2], str(top / "lib" / "llvm" / "bin" / "clang++"))

    def test_no_hipcc_is_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:     # a PATH with nothing on it: hipcc is not found
            p = subprocess.run(["/bin/bash", "-c", '. "$1"; hip_detect; echo "$?: $HIP_WHY"', "x", str(LC / "hip-config.sh")],
                               env={"PATH": tmp}, capture_output=True, text=True)
            self.assertEqual(p.stdout.strip(), "1: hipcc (ROCm) is not installed")
            fake = pathlib.Path(tmp) / "bin"
            fake.mkdir()
            for t in ("git", "cmake", "c++", "dirname", "mktemp", "rm", "bash"):   # build.sh needs these before it looks for hipcc
                exe = shutil.which(t)
                if exe:
                    (fake / t).symlink_to(exe)
            p = subprocess.run(["/bin/bash", str(LC / "build.sh"), "hip", tmp + "/prefix"], capture_output=True, text=True,
                               env={"PATH": str(fake), "HOME": tmp})
            self.assertEqual(p.returncode, 77, (p.stdout, p.stderr))
            self.assertIn("SKIP: hipcc", p.stdout)

    def test_compat_patch_covers_the_four_known_breaks(self):
        patch = (LC / "patches" / "hip-rocm5.patch").read_text()
        p = subprocess.run(["git", "apply", "--stat", "-"], input=patch, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        for name in ("ggml-cuda/mma.cuh", "ggml-cuda/vendors/hip.h", "ggml-hip/CMakeLists.txt"):
            self.assertIn(name, p.stdout)
        for what in ("pw_hipStreamWaitEvent", "pw_hipblasStrsmBatched", "pw_hipGetLastError", "x[ne] = {};"):
            self.assertIn(what, patch)

    def test_mmq_padding_patch(self):
        patch = (LC / "patches" / "mmq-y-overread.patch").read_text()
        p = subprocess.run(["git", "apply", "--stat", "-"], input=patch, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("ggml-cuda/mmq.cu", p.stdout)
        self.assertEqual(patch.count("+            2048;") + patch.count("+        2048;"), 2)   # both allocations of the activation buffer

    def test_sim_amd_target_names_the_shim_after_the_hip_soname_the_build_asks_for(self):
        """A build made with ROCm 5 asks for libamdhip64.so.5, and the simulator's library is preloaded under that name."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "dep").mkdir()
            (tmp / "dep" / "a.c").write_text("int a(void){return 1;}\n")
            subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,-soname,libamdhip64.so.5", "-o", str(tmp / "dep" / "libamdhip64.so.5"),
                            str(tmp / "dep" / "a.c")], check=True)
            lib, bindir = tmp / "prefix" / "lib", tmp / "prefix" / "bin"
            lib.mkdir(parents=True); bindir.mkdir()
            (tmp / "b.c").write_text("int b(void){return 2;}\n")
            subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(lib / "libggml-hip.so"), str(tmp / "b.c"),
                            "-L" + str(tmp / "dep"), "-Wl,--no-as-needed", "-l:libamdhip64.so.5"], check=True)
            shim = tmp / "sim" / "build" / "shim"
            shim.mkdir(parents=True)
            for n in ("libamdhip64.so.7", "libhsa-runtime64.so.1"):
                (shim / n).write_text("x")
            (bindir / "llama-completion").write_text("#!/bin/sh\nprintf '%s' \"$LD_PRELOAD\"\n")
            (bindir / "llama-bench").write_text("#!/bin/sh\n")
            for f in bindir.iterdir():
                f.chmod(0o755)
            model = tmp / "m.gguf"
            model.write_bytes(b"x")
            out = tmp / "out"; out.mkdir()
            p = subprocess.run(["bash", str(ROOT / "workloads" / WORKLOADS[0] / "run.sh")], cwd=ROOT, capture_output=True, text=True,
                               env={"PATH": os.environ["PATH"], "HOME": str(tmp), "PW_TARGET": "sim:amd/mi250x",
                                    "PW_SIM_PROFILE": "amd/mi250x", "PW_OUT": str(out), "PANTHEONSIM_DIR": str(tmp / "sim"),
                                    "PW_WORKLOAD_DIR": str(ROOT / "workloads" / WORKLOADS[0]), "PW_CACHE": str(tmp / "cache"),
                                    "PW_LLAMACPP_BIN_DIR": str(bindir), "PW_MODEL_FILE": str(model)})
            self.assertEqual(p.returncode, 0, (p.stdout, p.stderr))
            preload = json.loads(p.stdout.splitlines()[-1])["output"]
            self.assertEqual(preload, f"{out}/shim/libamdhip64.so.5:{out}/shim/libhsa-runtime64.so.1")
            self.assertTrue((out / "shim" / "libamdhip64.so.5").exists())


if __name__ == "__main__":
    unittest.main()
