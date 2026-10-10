#!/usr/bin/env python3
"""Record which architectures the pinned runtimes know, into tools/catalog/support.json.

The catalog claims a model can run on llama.cpp / vLLM / transformers / diffusers only when the pinned
version of that runtime lists the model's architecture. The lists are read from the runtimes' own source
at their pinned tags (no install, no download of weights):

  llama.cpp   src/llama-arch.cpp at the commit in tools/llamacpp/pin.env   (architecture names as GGUF writes them)
  vLLM        vllm/model_executor/models/registry.py at tag v0.30.0         (architecture class names of config.json)
  transformers  models/auto/auto_mappings.py and modeling_auto.py at v5.19.0 (config model_type names; image-text-to-text)
  diffusers   src/diffusers/__init__.py at v0.41.0                           (pipeline class names)

Run `tools/catalog/fetch_support.py` after changing a pin.
"""
import json
import pathlib
import re
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
VLLM_TAG, TRANSFORMERS_TAG, DIFFUSERS_TAG = "v0.30.0", "v5.19.0", "v0.41.0"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pantheonworkloads-catalog"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")


def block(text, name):
    """The text of `NAME = OrderedDict( ... )` up to its closing line."""
    m = re.search(rf"^{name}\s*=\s*OrderedDict\(\s*\[?(.*?)^\)", text, re.S | re.M)
    return m.group(1) if m else ""


def main():
    pin = dict(line.split("=", 1) for line in (ROOT / "tools/llamacpp/pin.env").read_text().splitlines() if "=" in line and not line.startswith("#"))
    commit = pin["LLAMACPP_PINNED_COMMIT"].strip()
    arch = get(f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{commit}/src/llama-arch.cpp")
    llama = sorted(set(re.findall(r'\{\s*LLM_ARCH_[A-Z0-9_]+,\s*"([a-z0-9_.-]+)"', arch)))
    reg = get(f"https://raw.githubusercontent.com/vllm-project/vllm/{VLLM_TAG}/vllm/model_executor/models/registry.py")
    vllm = sorted(set(re.findall(r'"([A-Za-z0-9_]+)"\s*:\s*\(', reg)))
    base = f"https://raw.githubusercontent.com/huggingface/transformers/{TRANSFORMERS_TAG}/src/transformers/models/auto/"
    maps = get(base + "auto_mappings.py")
    cfg = sorted(set(re.findall(r'\(\s*"([a-z0-9_-]+)",\s*"[A-Za-z0-9_]+Config"\s*\)', maps)) | {"parakeet_tdt"})
    model = get(base + "modeling_auto.py")
    itt = block(model, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES")
    itt_types = sorted(set(re.findall(r'\(\s*"([a-z0-9_-]+)",\s*"[A-Za-z0-9_]+"\s*\)', itt)))
    dif = get(f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_TAG}/src/diffusers/__init__.py")
    diffusers = sorted(set(re.findall(r'"([A-Za-z0-9]+Pipeline)"', dif)))
    out = {
        "llama.cpp": {"source": f"ggml-org/llama.cpp@{commit} src/llama-arch.cpp", "architectures": llama},
        "vllm": {"source": f"vllm-project/vllm@{VLLM_TAG} vllm/model_executor/models/registry.py", "architectures": vllm},
        "transformers": {"source": f"huggingface/transformers@{TRANSFORMERS_TAG} models/auto/auto_mappings.py, modeling_auto.py",
                         "model_types": cfg, "image_text_to_text_types": itt_types},
        "diffusers": {"source": f"huggingface/diffusers@{DIFFUSERS_TAG} src/diffusers/__init__.py", "pipelines": diffusers},
    }
    (HERE / "support.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print({k: len(next(v for kk, v in d.items() if kk != "source")) for k, d in out.items()})


if __name__ == "__main__":
    main()
