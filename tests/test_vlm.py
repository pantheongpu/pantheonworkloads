"""Tests for the vision-language-model workloads (qwen25-vl-7b, qwen3-vl-4b, smolvlm2-2p2b). No network, no GPU, no torch.

    python3 -m unittest discover -s tests -v
"""
import importlib.machinery
import importlib.util
import json
import pathlib
import re
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("pw_vlm", str(ROOT / "bin" / "pw"))
_spec = importlib.util.spec_from_loader("pw_vlm", _loader)
pw = importlib.util.module_from_spec(_spec)
_loader.exec_module(pw)

MODELS = {   # key -> (Hub repo, HF commit)
    "qwen25-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "qwen3-vl-4b": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "smolvlm2-2p2b": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4"),
}
IMAGE_SHA256 = "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"
WL = ROOT / "workloads"


def sums(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            s, n = line.split(None, 1)
            out[n.strip()] = s
    return out


class PinnedVlm(unittest.TestCase):
    def test_pins_are_consistent(self):
        for key, (repo, rev) in MODELS.items():
            f, b = WL / f"{key}-pytorch", WL / f"{key}-bench"
            for d in (f, b):
                errors, warnings = validate.check(d / "manifest.yaml")
                self.assertEqual(errors, [], d.name)
                self.assertFalse(any("UNVERIFIED" in w for w in warnings), d.name)
                m = validate.load(d / "manifest.yaml")
                self.assertTrue(m["model"]["revision"].startswith(rev + " "), d.name)
                self.assertEqual(m["model"]["source_url"], "https://huggingface.co/" + repo, d.name)
                self.assertRegex(m["model"]["licence"], r"^[A-Za-z0-9.\- ]+$", d.name)
                self.assertNotIn("UNVERIFIED", str(m["model"]["licence"]), d.name)
                self.assertEqual(m["targets"], ["gpu"], d.name)   # the CPU path is a documented SKIP, not a run
                s = sums(d / "model.sha256")
                self.assertEqual(s["astronaut.png"], IMAGE_SHA256, d.name)
                shards = [n for n in s if n.endswith(".safetensors")]
                self.assertTrue(shards, d.name)
                for n in shards:
                    self.assertRegex(s[n], r"^[0-9a-f]{64}$", d.name)
                self.assertIn(rev, (d / "model.sha256").read_text().splitlines()[0], d.name)
            self.assertEqual((f / "model.sha256").read_text(), (b / "model.sha256").read_text(), key)
            main = (f / "main.py").read_text()
            self.assertIn(f'"{repo}", "{rev}"', main, key)
            self.assertIn("requirements-vlm.txt", (f / "run.sh").read_text(), key)
            self.assertIn("requirements-vlm.txt", (b / "run.sh").read_text(), key)
            self.assertIn("PW_VLM_MODE=bench", (b / "run.sh").read_text(), key)
            self.assertIn(f"../{key}-pytorch/main.py", (b / "run.sh").read_text(), key)

    def test_references_come_from_a_gpu(self):
        for key in MODELS:
            ref = WL / f"{key}-pytorch" / "reference.json"
            if not ref.exists():
                self.skipTest("references are recorded on the rig")
            r = json.loads(ref.read_text())
            self.assertEqual(r["recorded_on"], "gpu", key)
            self.assertTrue(r["output"]["text"].strip(), key)
            ids = r["output"]["token_ids"].split()
            self.assertTrue(ids and all(i.isdigit() for i in ids), key)
            self.assertFalse((WL / f"{key}-bench" / "reference.json").exists(), key)

    def test_none_is_restricted_and_none_runs_on_cpu_or_sim(self):
        for key in MODELS:
            for d in (WL / f"{key}-pytorch", WL / f"{key}-bench"):
                self.assertFalse(validate.is_restricted(validate.load(d / "manifest.yaml")), d.name)
                for target in ("cpu", "sim:nvidia/h100"):
                    with tempfile.TemporaryDirectory() as tmp:
                        result, _, detail, _ = pw.execute(d.name, target, pathlib.Path(tmp))
                    self.assertEqual(result, "SKIP", (d.name, target))


class Requirements(unittest.TestCase):
    def test_vlm_requirements_are_exact_pins_and_the_shared_pins_did_not_move(self):
        lines = [x.strip() for x in (WL / "_pytorch" / "requirements-vlm.txt").read_text().splitlines()
                 if x.strip() and not x.startswith("#")]
        self.assertTrue(lines)
        for x in lines:
            self.assertRegex(x, r"^[A-Za-z0-9_.\-]+==[0-9][0-9.]*$", x)
        self.assertIn("num2words", " ".join(lines))   # SmolVLM's processor raises ImportError without it
        self.assertIn("transformers==5.19.0", (WL / "_pytorch" / "requirements-common.txt").read_text())
        torch = (WL / "_pytorch" / "requirements-torch.txt").read_text()
        self.assertIn("torch==2.14.1", torch)
        self.assertIn("torchvision==0.29.1", torch)

    def test_vlm_module_fixes_the_image_and_a_greedy_bench_shape(self):
        src = (WL / "_pytorch" / "vlm.py").read_text()
        self.assertIn(IMAGE_SHA256, src)
        self.assertIn("do_sample=False", src)
        self.assertRegex(src, r"BENCH_REPS = 5")
        self.assertEqual(len(re.findall(r"repetition_penalty=1\.0", src)), 1)


if __name__ == "__main__":
    unittest.main()
