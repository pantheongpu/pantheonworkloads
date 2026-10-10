"""Tests for the model catalog (tools/catalog/, docs/model-registry.md) and the workloads it writes.

Everything is checked offline from the files in the repository: hub.json (the Hub facts read on 2026-10-09), the
generated manifests, model.sha256 files and run.sh scripts. Nothing is downloaded, no GPU is needed; the host-check
tests run the real run.sh scripts against a fake nvidia-smi.

    python3 -m unittest test_model_catalog -v
"""
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "catalog"))
import generate  # noqa: E402
import model_catalog as mc  # noqa: E402
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw_catalog", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw_catalog", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

WL = ROOT / "workloads"
RESOLVED, FILES = generate.build_all()
WRITTEN = [r for r in RESOLVED if r["workloads"]]
BLOCKED = [r for r in RESOLVED if not r["workloads"]]
SUPPORT = mc.load_support()
HUB = mc.load_hub()
PERMISSIVE_LABELS = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "CC-BY-4.0", "CC0-1.0", "ISC", "Unlicense"}


def sums(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            s, n = line.split(None, 1)
            out[n.strip()] = s
    return out


def pairs(r):
    return generate.names(r)


def fake_smi(tmp, gpus):
    """A fake nvidia-smi printing `name, MiB, compute_cap` lines for the given (name, MiB) list."""
    p = pathlib.Path(tmp) / "nvidia-smi"
    lines = "\\n".join(f"{n}, {m}, 9.0, 550.1" for n, m in gpus)
    p.write_text(f'#!/bin/sh\nprintf "{lines}\\n"\n')
    p.chmod(0o755)
    return str(p)


class Generated(unittest.TestCase):
    def test_workloads_on_disk_are_what_the_catalog_generates(self):
        stale = [rel for rel, (content, _) in FILES.items() if not (ROOT / rel).exists() or (ROOT / rel).read_text(encoding="utf-8") != content]
        self.assertEqual(stale, [], "run tools/catalog/generate.py")

    def test_no_stray_catalog_workloads(self):
        expected = {rel.split("/")[1] for rel in FILES}
        found = {p.parent.name for p in WL.glob("*/manifest.yaml") if "WRITTEN, NEVER RUN" in p.read_text(encoding="utf-8")}
        self.assertEqual(found - expected, set(), "catalog workloads no longer in the catalog")

    def test_registry_lists_every_entry_and_its_blocked_reason(self):
        doc = (ROOT / "docs" / "model-registry.md").read_text(encoding="utf-8")
        for r in RESOLVED:
            self.assertIn(f"| {r['title'].replace('|', '/')} |", doc, r["key"])
        for r in BLOCKED:
            self.assertIn("blocked", r["status"], r["key"])
            self.assertIn(r["status"].replace("|", "/"), doc, r["key"])
            if r.get("access"):
                self.assertIn("HF_TOKEN", doc)

    def test_keys_are_unique_and_names_fit_the_manifest_rules(self):
        keys = [r["key"] for r in RESOLVED]
        self.assertEqual(len(keys), len(set(keys)))
        names = [n for r in WRITTEN for n in pairs(r)]
        self.assertEqual(len(names), len(set(names)))
        for n in names:
            self.assertRegex(n, validate.NAME.pattern)


class PinnedModels(unittest.TestCase):
    """Every written manifest is pinned, has a licence that is not UNVERIFIED, a requires block and matching model.sha256 entries."""

    def test_each_manifest_validates_with_only_the_no_reference_warning(self):
        for r in WRITTEN:
            for name in pairs(r):
                errors, warnings = validate.check(WL / name / "manifest.yaml")
                self.assertEqual(errors, [], name)
                self.assertTrue(all("no reference.json" in w for w in warnings), (name, warnings))

    def test_pinned_revision_licence_requires_and_never_run_note(self):
        for r in WRITTEN:
            for name in pairs(r):
                m = validate.load(WL / name / "manifest.yaml")
                model = m["model"]
                self.assertTrue(model["revision"].startswith(r["repo_sha"]) or r["repo_sha"] in model["revision"], name)
                self.assertRegex(r["repo_sha"], r"^[0-9a-f]{40}$", name)
                self.assertNotIn("UNVERIFIED", str(model["licence"]).upper(), name)
                self.assertEqual(model["licence"], r["licence"]["label"], name)
                self.assertNotEqual(model["licence"], "unreadable", name)
                self.assertEqual(model["source_url"], "https://huggingface.co/" + r["repo"], name)
                req = validate.requirements(m)
                self.assertIsNotNone(req, name)
                self.assertEqual(req["gpus"], r["hw"]["gpus"], name)
                self.assertEqual(req["gpu_memory_gb"], r["hw"]["gpu_memory_gb"], name)
                self.assertEqual(m["targets"], ["gpu"], name)
                self.assertIn("WRITTEN, NEVER RUN", m["notes"], name)
                self.assertIn("Not verified", m["notes"], name)
                self.assertIn(r["licence"]["label"], m["notes"], name)

    def test_model_sha256_matches_the_hub_listing(self):
        for r in WRITTEN:
            expected = {f["path"]: f["sha256"] for f in r["files"] if f["sha256"]}
            self.assertTrue(expected, r["key"])
            for name in pairs(r):
                got = sums(WL / name / "model.sha256")
                self.assertEqual(got, expected, name)
                for s in got.values():
                    self.assertRegex(s, r"^[0-9a-f]{64}$", name)
            self.assertEqual((WL / pairs(r)[0] / "model.sha256").read_text().splitlines()[1:], (WL / pairs(r)[1] / "model.sha256").read_text().splitlines()[1:])

    def test_pins_are_the_hub_data(self):
        for r in WRITTEN:
            rec = HUB["repos"][r["repo"]]
            self.assertEqual(r["repo_sha"], rec["sha"], r["key"])
            by = {f["path"]: f for f in rec["files"]}
            for f in r["files"]:
                self.assertEqual(f["sha256"], by[f["path"]]["sha256"], (r["key"], f["path"]))
        self.assertEqual(HUB["fetched"], mc.DATE)

    def test_no_reference_and_no_bench_record(self):
        for r in WRITTEN:
            for name in pairs(r):
                self.assertFalse((WL / name / "reference.json").exists(), name)
                self.assertFalse((ROOT / "bench" / name).exists(), name)

    def test_restricted_follows_the_licence(self):
        for r in RESOLVED:
            lic = r["licence"]
            derived = any("Llama community licence terms apply to derivatives" in n for n in lic["notes"])   # e.g. Orpheus: Apache-2.0 card, Llama 3.2 base
            self.assertEqual(lic["restricted"], derived or lic["id"] not in mc.PERMISSIVE, r["key"])
            if not r["workloads"]:
                continue
            for name in pairs(r):
                m = validate.load(WL / name / "manifest.yaml")
                self.assertEqual(validate.is_restricted(m), lic["restricted"], name)
                if lic["restricted"]:
                    self.assertIn("RESTRICTED", m["notes"], name)
            if r["licence"]["label"] not in PERMISSIVE_LABELS:
                self.assertTrue(lic["restricted"], r["key"])

    def test_llama_gemma_and_noncommercial_licences_are_restricted(self):
        for r in RESOLVED:
            label = r["licence"]["label"]
            if re.search(r"llama|gemma|non-?commercial|\bNC\b|research|community|falcon|qwen license", label, re.I):
                self.assertTrue(r["licence"]["restricted"], r["key"])

    def test_blocked_entries_have_no_workload_dir_and_say_why(self):
        for r in BLOCKED:
            self.assertTrue(r["status"].startswith("blocked: "), r["key"])
            for stem in (f"llamacpp-{r['key']}", f"llamacpp-bench-{r['key']}", f"{r['key']}-pytorch", f"{r['key']}-diffusers", f"vllm-{r['key']}"):
                self.assertFalse((WL / stem).exists(), stem)
            if r["status"].startswith("blocked: gated"):
                self.assertIn("HF_TOKEN", r.get("access", ""), r["key"])


class HardwareArithmetic(unittest.TestCase):
    """requires.gpus / requires.gpu_memory_gb agree with the registry arithmetic, recomputed here from the documented formulas."""

    @staticmethod
    def per_gpu(plan, w, n, kind=None, largest=None):
        if plan in ("llamacpp", "llamacpp-embed"):
            return w / n * 1.05 + 3.0
        if plan == "whispercpp":
            return w / n * 1.05 + 2.0
        if plan == "vllm":
            return (w / n + 6.0) / 0.90
        if plan == "diffusers":
            return max(largest or 0, w / n) + (16.0 if kind == "video" else 6.0)
        return w / n * 1.10 + 4.0

    def test_requires_holds_the_weights_plus_margin_and_is_the_smallest_class(self):
        for r in WRITTEN:
            m = validate.load(WL / pairs(r)[0] / "manifest.yaml")
            n, mem = m["requires"]["gpus"], m["requires"]["gpu_memory_gb"]
            w = r["weights_gib"]
            largest = None
            if r["plan"] == "diffusers":
                comps = {}
                for f in r["files"]:
                    if f["path"].endswith(".safetensors"):
                        comps[f["path"].split("/")[0]] = comps.get(f["path"].split("/")[0], 0) + f["size"]
                half = set(HUB["repos"][r["repo"]].get("param_dtypes") or {}) == {"F32"} and not r.get("variant")
                largest = max(comps.values()) / mc.GIB * (0.5 if half else 1.0)
            need = self.per_gpu(r["plan"], w, n, r["spec"].get("kind"), largest)
            self.assertLessEqual(need, 0.98 * mem + 1e-9, (r["key"], need, mem))
            self.assertLessEqual(w, n * mem * 0.98, r["key"])   # the weights alone fit the GPUs' memory
            smaller = [t for t in mc.TIERS if t < mem and (mc.COMMON_TIERS if mem in mc.COMMON_TIERS else mc.TIERS)]
            if n == 1 or mem in mc.COMMON_TIERS:
                # no smaller class holds it on the same number of GPUs
                self.assertFalse([t for t in smaller if need <= 0.98 * t], (r["key"], need, mem))
            self.assertIn(mem, mc.TIERS)
            self.assertEqual(r["hw"]["gpus"], n)

    def test_big_workloads_declare_more_than_one_gpu_and_fit_their_total_memory(self):
        for r in WRITTEN:
            m = validate.load(WL / pairs(r)[0] / "manifest.yaml")
            req = validate.requirements(m)
            if r["weights_gib"] > 80 * 0.98 - 6 and r["plan"] != "diffusers":
                self.assertGreater(req["gpus"] * req["gpu_memory_gb"], r["weights_gib"], r["key"])
            if r["weights_gib"] > 600:
                self.assertGreaterEqual(req["gpus"], 4, r["key"])

    def test_every_manifest_of_a_pair_requires_the_same(self):
        for r in WRITTEN:
            a, b = (validate.requirements(validate.load(WL / n / "manifest.yaml")) for n in pairs(r))
            self.assertEqual({k: v for k, v in a.items() if k != "notes"}, {k: v for k, v in b.items() if k != "notes"}, r["key"])

    def test_the_biggest_known_cases(self):
        by = {r["key"]: r for r in WRITTEN}
        self.assertEqual((by["kimi-k3"]["hw"]["gpus"], by["kimi-k3"]["hw"]["gpu_memory_gb"]), (8, 192))
        self.assertEqual((by["gpt-oss-120b-vllm"]["hw"]["gpus"], by["gpt-oss-120b-vllm"]["hw"]["gpu_memory_gb"]), (1, 80))
        self.assertEqual((by["deepseek-r1-0528"]["hw"]["gpus"], by["deepseek-r1-0528"]["hw"]["gpu_memory_gb"]), (8, 80))
        self.assertEqual(by["qwen3-reranker-0p6b"]["hw"]["gpus"], 1)

    def test_hardware_function_picks_few_gpus_in_common_classes_first(self):
        self.assertEqual(mc.hardware("llamacpp", 4.0)["gpu_memory_gb"], 10)
        self.assertEqual((mc.hardware("llamacpp", 39.6)["gpus"], mc.hardware("llamacpp", 39.6)["gpu_memory_gb"]), (1, 48))
        self.assertEqual(mc.hardware("llamacpp", 300.0)["gpus"], 8)
        self.assertLessEqual(mc.hardware("llamacpp", 300.0)["gpu_memory_gb"], 80)
        self.assertGreater(mc.hardware("llamacpp", 1000.0)["gpu_memory_gb"], 80)
        self.assertIn("nodes", mc.hardware("llamacpp", 3000.0)["scope"])


class RuntimeSupport(unittest.TestCase):
    """A workload is written only when the pinned runtime lists the architecture (support.json, read from the runtimes' source)."""

    def test_llamacpp_architectures_are_in_the_pinned_source(self):
        archs = set(SUPPORT["llama.cpp"]["architectures"])
        for r in WRITTEN:
            if r["plan"] in ("llamacpp", "llamacpp-embed"):
                self.assertIn(r["gguf_arch"], archs, r["key"])

    def test_vllm_architectures_are_in_the_pinned_registry(self):
        archs = set(SUPPORT["vllm"]["architectures"])
        for r in WRITTEN:
            if r["plan"] == "vllm" and not r["spec"].get("mistral_format"):
                self.assertIn(r["architecture"], archs, r["key"])

    def test_diffusers_pipelines_exist_in_the_pinned_version(self):
        pipes = set(SUPPORT["diffusers"]["pipelines"])
        for r in WRITTEN:
            if r["plan"] == "diffusers":
                self.assertIn(r["architecture"], pipes, r["key"])

    def test_vlm_model_types_are_image_text_to_text_in_the_pinned_transformers(self):
        types = set(SUPPORT["transformers"]["image_text_to_text_types"])
        for r in WRITTEN:
            if r["plan"] == "vlm":
                self.assertIn(HUB["repos"][r["repo"]]["config"]["model_type"], types, r["key"])

    def test_gguf_files_are_complete_shard_sets(self):
        for r in WRITTEN:
            if r["plan"] not in ("llamacpp", "llamacpp-embed"):
                continue
            names = [f["path"].rsplit("/", 1)[-1] for f in r["files"]]
            shards = [re.search(r"-(\d{5})-of-(\d{5})\.gguf$", n) for n in names]
            if any(shards):
                total = int(shards[0].group(2))
                self.assertEqual(sorted(int(s.group(1)) for s in shards), list(range(1, total + 1)), r["key"])
            else:
                self.assertEqual(len(names), 1, r["key"])

    def test_speech_requirements_are_exact_pins(self):
        lines = [x.strip() for x in (WL / "_pytorch" / "requirements-speech.txt").read_text().splitlines() if x.strip() and not x.startswith("#")]
        self.assertTrue(lines)
        for x in lines:
            self.assertRegex(x, r"^[A-Za-z0-9_.\-]+(\[[a-z]+\])?==[0-9][0-9.]*$", x)


class RunScripts(unittest.TestCase):
    """Each run.sh exits 77 with a clear message on a host without the GPUs (checked by running it), is valid bash, and its numbers match its manifest."""

    def test_bash_syntax_and_matching_gpu_numbers(self):
        for r in WRITTEN:
            for name in pairs(r):
                path = WL / name / "run.sh"
                self.assertTrue(os.access(path, os.X_OK), name)
                p = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, (name, p.stderr))
                text = path.read_text()
                req = validate.requirements(validate.load(WL / name / "manifest.yaml"))
                self.assertRegex(text, rf"(pw_require_gpus|lc_require_gpus) {req['gpus']} {req['gpu_memory_gb']}\b", name)
                self.assertNotRegex(text, r"curl [^\n]*\bhf\b|from_pretrained\(", name)   # nothing fetches a model at validate time

    def test_python_mains_compile_and_pin_the_commit(self):
        for r in WRITTEN:
            f = WL / pairs(r)[0] / "main.py"
            if f.exists():
                src = f.read_text()
                compile(src, str(f), "exec")
                self.assertIn(f'"{r["repo_sha"]}"', src, r["key"])
        for p in (WL / "_pytorch").glob("*.py"):
            compile(p.read_text(), str(p), "exec")
        compile((WL / "_shared" / "vllm_catalog.py").read_text(), "vllm_catalog.py", "exec")

    def run_script(self, name, gpus, target="gpu"):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PW_TARGET=target, PW_WORKLOAD_DIR=str(WL / name), PW_OUT=tmp, PW_NVIDIA_SMI=fake_smi(tmp, gpus),
                       PW_ROCM_SMI="/nonexistent", PW_DRM_ROOT=tmp, PW_CACHE=str(pathlib.Path(tmp) / "cache"))
            env.pop("PW_IGNORE_REQUIRES", None)
            return subprocess.run(["bash", str(WL / name / "run.sh")], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120)

    def test_each_plan_exits_77_with_the_needs_on_a_small_host(self):
        by_plan = {}
        for r in WRITTEN:
            by_plan.setdefault((r["plan"], r["spec"].get("engine") or r["spec"].get("head")), r)
        self.assertGreaterEqual(len(by_plan), 10)
        for plan, r in by_plan.items():
            for name in pairs(r):
                p = self.run_script(name, [])   # no GPU at all (and the wanted memory on none)
                self.assertEqual(p.returncode, 77, (name, p.stdout, p.stderr))
                self.assertIn("SKIP", p.stdout + p.stderr, name)
                self.assertRegex(p.stdout + p.stderr, r"needs \d+ GPU|GiB", name)

    def test_a_non_gpu_target_is_a_skip(self):
        r = next(r for r in WRITTEN if r["plan"] == "llamacpp")
        for target in ("cpu", "sim:nvidia/h100"):
            p = self.run_script(pairs(r)[0], [("NVIDIA H100", 81559)] * 8, target)
            self.assertEqual(p.returncode, 77, (target, p.stdout, p.stderr))

    def test_enough_gpus_pass_the_check_and_stop_at_the_next_missing_thing(self):
        r = next(r for r in WRITTEN if r["key"] == "gpt-oss-120b-vllm")
        p = self.run_script(pairs(r)[0], [("NVIDIA H100 80GB HBM3", 81559)])
        self.assertNotRegex(p.stdout + p.stderr, r"needs \d+ GPU", (p.stdout, p.stderr))   # the GPU check passed (it stops at vLLM or the disk)

    def test_runner_skips_before_starting_the_workload(self):
        r = next(r for r in WRITTEN if r["key"] == "kimi-k3")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"PW_NVIDIA_SMI": fake_smi(tmp, [("NVIDIA A10G", 23028)])}):
            result, _, detail, _ = pw.execute(pairs(r)[0], "gpu", pathlib.Path(tmp))
        self.assertEqual(result, "SKIP")
        self.assertIn("needs 8 GPUs with >= 192 GiB each", detail)

    def test_restricted_workloads_are_left_out_of_default_selections(self):
        default = set(pw.default_names())
        for r in WRITTEN:
            for name in pairs(r):
                self.assertEqual(name in default, not r["licence"]["restricted"], name)


class HwCheck(unittest.TestCase):
    def count(self, gib, gpus, vendor=None):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PW_NVIDIA_SMI=fake_smi(tmp, gpus), PW_ROCM_SMI="/nonexistent")
            args = [sys.executable, str(ROOT / "tools" / "hwcheck.py"), str(gib)] + ([vendor] if vendor else [])
            return subprocess.run(args, capture_output=True, text=True, env=env).stdout.strip()

    def test_counts_cards_with_the_98_percent_allowance(self):
        self.assertTrue(self.count(80, [("H100", 81559)] * 8).startswith("8 "))   # 79.6 GiB counts as 80
        self.assertTrue(self.count(80, [("A100-40", 40960)] * 8).startswith("0 "))
        self.assertTrue(self.count(22, [("A10G", 23028)]).startswith("1 "))
        self.assertTrue(self.count(24, [("A10G", 23028)]).startswith("0 "))   # 22.5 GiB is not a 24 GiB card
        self.assertTrue(self.count(44, [("L40S", 46068)]).startswith("1 "))

    def test_vendor_and_no_gpu(self):
        self.assertTrue(self.count(10, [("H100", 81559)], "amd").startswith("0 "))
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PW_NVIDIA_SMI="/nonexistent", PW_ROCM_SMI="/nonexistent", PW_DRM_ROOT=tmp)
            out = subprocess.run([sys.executable, str(ROOT / "tools" / "hwcheck.py"), "10"], capture_output=True, text=True, env=env).stdout
        self.assertEqual(out.strip(), "0 no GPU detected")

    def test_amd_sysfs_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "card0" / "device"
            d.mkdir(parents=True)
            (d / "vendor").write_text("0x1002\n")
            (d / "mem_info_vram_total").write_text(str(192 * 2**30))
            env = dict(os.environ, PW_NVIDIA_SMI="/nonexistent", PW_ROCM_SMI="/nonexistent", PW_DRM_ROOT=tmp)
            out = subprocess.run([sys.executable, str(ROOT / "tools" / "hwcheck.py"), "180", "amd"], capture_output=True, text=True, env=env).stdout
        self.assertTrue(out.startswith("1 "), out)


class LlamaCppShards(unittest.TestCase):
    def test_lc_fetch_shards_verifies_every_file_and_sets_the_first_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            web = tmp / "web" / "sub"
            web.mkdir(parents=True)
            import hashlib
            names = ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
            lines = []
            for i, n in enumerate(names):
                (web / n).write_bytes(bytes([i]) * 100)
                lines.append(f"{hashlib.sha256((web / n).read_bytes()).hexdigest()}  sub/{n}")
            wdir = tmp / "wl"
            wdir.mkdir()
            (wdir / "model.sha256").write_text("# header\n" + "\n".join(lines) + "\n")
            script = (f'set -u; PW_TARGET=gpu; PW_CACHE={tmp}/cache; . {ROOT}/tools/llamacpp/common.sh; '
                      f'lc_fetch_shards file://{tmp}/web {wdir}/model.sha256 0; echo "$LC_MODEL"')
            p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(p.stdout.strip(), f"{tmp}/cache/models/wl/{names[0]}")
            self.assertTrue((tmp / "cache" / "models" / "wl" / names[1]).exists())
            (wdir / "model.sha256").write_text("0" * 64 + f"  sub/{names[0]}\n")
            bad = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(bad.returncode, 1, bad.stdout)
            self.assertIn("expected", bad.stderr)


if __name__ == "__main__":
    unittest.main()
