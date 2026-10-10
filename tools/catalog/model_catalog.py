"""Resolve tools/catalog/catalog.yaml against the Hub facts in hub.json (no network, nothing downloaded).

`resolve_all()` returns one dict per catalog entry with: the pinned repositories and files, the licence as read,
whether the entry is `restricted`, the runtime plan, the weights size, the hardware need with its arithmetic,
and a status (`manifest written` or `blocked: <reason>`). tools/catalog/generate.py turns the records into
workloads/ directories and docs/model-registry.md; tests/test_model_catalog.py checks the written manifests
against them.
"""
import json
import pathlib
import re
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_hub  # noqa: E402

GIB = 1 << 30
DATE = "2026-10-09"

# GPU memory classes a `requires.gpu_memory_gb` may name, in GiB as the driver reports them (docs/manifest.md: a card
# counts when it reports at least 98% of the value). 22 is the 24 GB-marketing cards that report 22.5 GiB (A10G, L4),
# 24 the cards that report a full 24 (RTX 3090/4090, A5000), 44 the 48 GB L40S (45 GiB), 48 a full 48 (A6000, RTX 6000 Ada),
# 140 an H200 (140.4 GiB), 180 a B200, 192 an MI300X, 256 an MI325X, 288 an MI355X / B300.
COMMON_TIERS = [10, 15, 22, 24, 40, 44, 48, 80]   # the classes of cards that are widely rented
BIG_TIERS = [140, 180, 192, 256, 288]
TIERS = COMMON_TIERS + BIG_TIERS
USABLE = 0.98
GPU_COUNTS = [1, 2, 4, 8]
NODE_GPUS = 8   # more GPUs than this means more than one node

# Margins on top of the weights. Each is a stated assumption, not a measurement (nothing has been run).
LLAMACPP = dict(split=1.05, fixed=3.0)       # layer split imbalance; CUDA context + compute buffers + KV cache for a 4096-token context
VLLM = dict(fixed=6.0, util=0.90)            # activations + CUDA context + KV for 4096 tokens; gpu_memory_utilization 0.90
TRANSFORMERS = dict(split=1.10, fixed=4.0)   # device_map="auto" imbalance; activations of one image/clip and a short KV cache
DIFFUSERS = dict(image=6.0, video=16.0)      # latents, attention activations and the VAE decode at the workload's resolution
WHISPERCPP = dict(split=1.05, fixed=2.0)

PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0", "cc-by-4.0"}
SPDX = {"apache-2.0": "Apache-2.0", "mit": "MIT", "bsd-3-clause": "BSD-3-Clause", "bsd-2-clause": "BSD-2-Clause",
        "cc-by-4.0": "CC-BY-4.0", "cc-by-nc-4.0": "CC-BY-NC-4.0", "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0", "cc0-1.0": "CC0-1.0",
        "isc": "ISC", "unlicense": "Unlicense"}
LLAMA_NAMES = {"llama3": "Llama 3 Community Licence", "llama3.1": "Llama 3.1 Community Licence",
               "llama3.2": "Llama 3.2 Community Licence", "llama3.3": "Llama 3.3 Community Licence",
               "gemma": "Gemma Terms of Use"}
# card `license_name` slugs seen in the catalog, spelled out. The slug itself is always kept in the evidence.
SLUG_NAMES = {"mnpl": "Mistral AI Non-Production License", "mrl": "Mistral Research License", "qwen": "Qwen License",
              "qwen-research": "Qwen Research License", "falcon-llm-license": "Falcon LLM License", "llama4": "Llama 4 Community License",
              "flux-1-dev-non-commercial-license": "FLUX.1 [dev] Non-Commercial License",
              "flux-non-commercial-license": "FLUX Non-Commercial License", "stabilityai-ai-community": "Stability AI Community License",
              "openmdw-1.1": "OpenMDW-1.1", "modified-mit": "Modified MIT", "dinov3-license": "DINOv3 License",
              "nvidia-nemotron-open-model-license": "NVIDIA Nemotron Open Model License",
              "nvidia-open-model-license": "NVIDIA Open Model License", "tencent-hunyuan-community": "Tencent Hunyuan Community License",
              "qwen-community-1.0": "Qwen Community License 1.0", "glm-5.3": "GLM-5.3 License", "kimi-k3": "Kimi K3 License",
              "exaone": "EXAONE AI Model License Agreement"}

QUANT_RE = re.compile(r"(?<![A-Za-z0-9])((?:UD-)?(?:IQ[1-4]_(?:XXS|XS|S|M|NL)|Q[2-8]_K_(?:XL|XXL|L|M|S)|Q[2-8]_K|Q[4-8]_[01]|MXFP4(?:_MOE)?|NVFP4|BF16|F16|F32|TQ[12]_0))(?![A-Za-z0-9])", re.I)
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})(?=\.gguf$)")
GGUF_SKIP = re.compile(r"(mmproj|(^|[-_/.])mtp[-_.]|dflash|dspark|draft|imatrix|eagle|lora|vocab|assistant|Q4_0_\d_\d)", re.I)
BIG_PREF = ["Q4_K_M", "UD-Q4_K_M", "MXFP4_MOE", "MXFP4", "UD-Q4_K_XL", "Q4_0", "UD-IQ4_XS", "IQ4_XS", "Q8_0"]
SMALL_PREF = ["Q8_0", "Q4_K_M", "UD-Q4_K_M", "UD-Q4_K_XL", "Q4_0", "MXFP4", "IQ4_XS"]
LAST_RESORT = ["Q3_K_M", "Q3_K_S", "Q3_K_L", "Q5_K_M", "Q6_K", "Q2_K"]
SMALL_PARAMS = 14.5e9
WEIGHT_FILE = re.compile(r"\.(safetensors|bin|pt|pth|nemo|ckpt)$")
SMALL_FILE = re.compile(r"(^|/)([^/]*\.json|[^/]*\.txt|[^/]*\.model|merges\.txt|tokenizer[^/]*|[^/]*\.jinja)$")


def load_catalog():
    entries = yaml.safe_load((HERE / "catalog.yaml").read_text(encoding="utf-8"))["entries"]
    for e in entries:
        e.setdefault("spec", {})
    return entries


def load_hub():
    return fetch_hub.load(HERE / "hub.json")


def load_support():
    return json.loads((HERE / "support.json").read_text(encoding="utf-8"))


def gguf_candidates(entry):
    return fetch_hub.gguf_candidates(entry)


def human_params(n):
    if not n:
        return "?"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if n >= div:
            v = n / div
            return f"{v:.3g}{unit}" if v < 100 else f"{v:.0f}{unit}"
    return str(n)


def tier_for(per_gpu, tiers=TIERS):
    for t in tiers:
        if per_gpu <= USABLE * t:
            return t
    return None


def per_gpu_need(plan, weights_gib, n, kind=None, largest_gib=None):
    """GiB needed on each of n GPUs, and the formula text with its numbers."""
    w = weights_gib
    if plan in ("llamacpp", "llamacpp-embed"):
        m = LLAMACPP
        return w / n * m["split"] + m["fixed"], f"{w:.1f}/{n}*{m['split']} + {m['fixed']:g}"
    if plan == "whispercpp":
        m = WHISPERCPP
        return w / n * m["split"] + m["fixed"], f"{w:.1f}/{n}*{m['split']} + {m['fixed']:g}"
    if plan == "vllm":
        m = VLLM
        return (w / n + m["fixed"]) / m["util"], f"({w:.1f}/{n} + {m['fixed']:g})/{m['util']}"
    if plan == "diffusers":
        margin = DIFFUSERS["video" if kind == "video" else "image"]
        base = max(largest_gib or 0.0, w / n)
        return base + margin, f"max({largest_gib or 0:.1f} largest component, {w:.1f}/{n}) + {margin:g}"
    m = TRANSFORMERS
    return w / n * m["split"] + m["fixed"], f"{w:.1f}/{n}*{m['split']} + {m['fixed']:g}"


def hardware(plan, weights_gib, kind=None, largest_gib=None):
    """The fewest GPUs (1, 2, 4, 8) that hold the weights in a widely rented class (<= 80 GiB), each of the smallest class
    that fits. If even 8 x 80 GiB is not enough: the cheapest n x class among 140..288 GiB cards (n x class minimal, ties
    to fewer GPUs), and above 8 GPUs a multi-node count (16, 32)."""
    def attempt(n, tiers):
        need, text = per_gpu_need(plan, weights_gib, n, kind, largest_gib)
        t = tier_for(need, tiers)
        return (n, t, need, text) if t else None

    pick, scope = None, "single node"
    for n in GPU_COUNTS:
        pick = attempt(n, COMMON_TIERS)
        if pick:
            break
    if not pick:
        options = [o for o in (attempt(n, TIERS) for n in GPU_COUNTS) if o]
        if options:
            pick = min(options, key=lambda o: (o[0] * o[1], o[0]))
    if not pick:
        for n in (16, 32):
            pick = attempt(n, TIERS)
            if pick:
                scope = f"{n // NODE_GPUS} nodes of {NODE_GPUS} GPUs"
                break
    if not pick:
        return None
    n, t, need, text = pick
    return {"gpus": n, "gpu_memory_gb": t, "per_gpu_gib": round(need, 1), "formula": text, "scope": scope,
            "arithmetic": f"{text} = {need:.1f} GiB per GPU; smallest class holding it at {USABLE:.0%} usable: {t} GiB; {n} x {t}"}


def licence_label(rec):
    """(label, id, permissive?) from a repo record's card data; (None, None, False) when the card names no licence."""
    lic = (rec.get("license") or "").strip()
    name = (rec.get("license_name") or "").strip()
    if name == "other":
        name = ""
    if not lic and rec.get("tags_license"):
        lic = rec["tags_license"][0].split(":", 1)[1]
    if lic in SPDX:
        return SPDX[lic], lic, lic in PERMISSIVE
    if lic in LLAMA_NAMES:
        return LLAMA_NAMES[lic], lic, False
    if lic == "other":
        if name:
            return SLUG_NAMES.get(name, name), f"other/{name}", False
        heads = list((rec.get("licence_files") or {}).values())
        if heads:
            return f"custom licence (LICENSE file begins '{heads[0][:60]}')", "other", False
        return "custom licence (card: license other, no name given)", "other", False
    if lic:
        return lic, lic, False
    return None, None, False


def licence_evidence(repo, rec):
    parts = [f"{repo}@{rec['sha'][:8]}"]
    lic = rec.get("license") or (rec.get("tags_license") or [""])[0].replace("license:", "") or None
    if lic:
        s = f"card license: {lic}"
        if rec.get("license_name"):
            s += f", license_name: {rec['license_name']}"
        if rec.get("license_link"):
            s += f", license_link: {rec['license_link']}"
        parts.append(s)
    for fn, head in (rec.get("licence_files") or {}).items():
        parts.append(f"{fn} begins '{head[:90]}'")
    if rec.get("gated"):
        parts.append(f"GATED ({rec['gated']}): only the API tags were readable")
    elif rec.get("card_licence_section"):
        parts.append(f"card licence section: '{rec['card_licence_section'][:100]}'")
    return "; ".join(parts)


def gguf_groups(rec):
    """{quant: [(shard count, files)]} of the complete GGUF models in a repository record."""
    groups = {}
    for f in rec["files"]:
        p = f["path"]
        base = p.rsplit("/", 1)[-1]
        if not p.endswith(".gguf") or GGUF_SKIP.search(p):
            continue
        m = QUANT_RE.search(base)
        if not m:
            continue
        s = SHARD_RE.search(base)
        key = (p.rsplit("/", 1)[0] if "/" in p else "", SHARD_RE.sub("", base), m.group(1).upper(), int(s.group(2)) if s else 0)
        groups.setdefault(key, []).append((int(s.group(1)) if s else 0, f))
    out = {}
    for (_, _, q, total), items in groups.items():
        have = sorted(i for i, _ in items)
        if total and have != list(range(1, total + 1)):
            continue
        files = [f for _, f in sorted(items, key=lambda x: x[1]["path"])]
        if not all(f["sha256"] and f["size"] for f in files):
            continue
        out.setdefault(q, []).append((total or 1, files))
    for q in out:
        out[q].sort(key=lambda t: t[0])   # fewest files first
    return out


def pick_gguf(entry, hub, params):
    """Choose (repo, quant, files) among the candidate GGUF repositories: best quantisation first, then repository order."""
    prefs = [entry["quant"]] if entry.get("quant") else (SMALL_PREF if params and params <= SMALL_PARAMS else BIG_PREF)
    tried = []
    avail = {}
    for repo in gguf_candidates(entry):
        rec = hub["repos"].get(repo)
        if not rec or "error" in rec:
            continue
        if rec.get("gated"):
            tried.append(f"{repo} (gated)")
            continue
        tried.append(repo)
        for q, groups in gguf_groups(rec).items():
            avail.setdefault(q, []).append((repo, groups[0][1]))
    for q in prefs + ([] if entry.get("quant") else LAST_RESORT):
        for key, cands in avail.items():
            if key.upper() == q.upper():
                return cands[0][0], key, cands[0][1], tried, sorted(avail)
    return None, None, None, tried, sorted(avail)


def resolve_entry(entry, hub, support):
    key, plan = entry["key"], entry["plan"]
    out = {"key": key, "family": entry["family"], "title": entry["title"], "plan": plan, "base": entry["base"],
           "spec": entry.get("spec") or {}, "note": entry.get("note"), "entry": entry}
    base = hub["repos"].get(entry["base"])
    repo = entry.get("repo") or entry["base"]
    if not base or "error" in base:
        out.update(status=f"blocked: base repository unreadable ({(base or {}).get('error')})", workloads=False,
                   licence={"label": "unreadable", "id": None, "restricted": True, "evidence": [], "notes": [], "unreadable": True})
        return out
    out["base_sha"] = base["sha"]
    params = base.get("params")
    out["params"] = params
    out["base_gated"] = base.get("gated")
    reason = entry.get("block")

    if plan in ("llamacpp", "llamacpp-embed"):
        grepo, quant, files, tried, avail = pick_gguf(entry, hub, params)
        out["gguf_tried"], out["gguf_quants_seen"] = tried, avail
        if grepo is None:
            gated = [t for t in tried if t.endswith("(gated)")]
            if not reason:
                reason = "gated" if base.get("gated") and not any(not t.endswith("(gated)") for t in tried) else "no GGUF"
            out.update(status=f"blocked: {reason}", workloads=False)
            out.update(_licence(entry, hub, base, None))
            if base.get("gated"):
                out["access"] = _access(entry["base"], base)
            return out
        rec = hub["repos"][grepo]
        out.update(repo=grepo, repo_sha=rec["sha"], quant=quant, files=files, params=params or (rec.get("gguf") or {}).get("total"))
        arch = (rec.get("gguf") or {}).get("architecture")
        out["gguf_arch"] = arch
        w = sum(f["size"] for f in files) / GIB
        out["weights_gib"] = w
        out["dtype"] = quant
        if arch and arch not in support["llama.cpp"]["architectures"] and not reason:
            reason = f"llama.cpp b11447 has no '{arch}' architecture (as the GGUF header declares it)"
        out["hw"] = hardware(plan, w)
        out.update(_licence(entry, hub, base, rec, grepo))
        if base.get("gated"):
            out["community"] = True
    elif plan == "whispercpp":
        rec = hub["repos"][entry["repo"]]
        pat = [re.compile(p) for p in entry["allow"]]
        files = [f for f in rec["files"] if any(p.search(f["path"]) for p in pat)]
        out.update(repo=entry["repo"], repo_sha=rec["sha"], files=files, quant="ggml (fp16)")
        w = sum(f["size"] for f in files) / GIB
        out["weights_gib"] = w
        out["hw"] = hardware("whispercpp", w)
        out.update(_licence(entry, hub, base, rec, entry["repo"]))
    else:
        rec = hub["repos"].get(repo)
        if not rec or "error" in rec:
            out.update(status="blocked: repository unreadable", workloads=False,
                       licence={"label": "unreadable", "id": None, "restricted": True, "evidence": [], "notes": [], "unreadable": True})
            return out
        out.update(repo=repo, repo_sha=rec["sha"])
        out.update(_licence(entry, hub, base, rec if repo != entry["base"] else None, repo))
        files, dtype, variant = _select_files(entry, rec)
        out["files"], out["dtype"], out["variant"] = files, dtype, variant
        wfiles = [f for f in files if WEIGHT_FILE.search(f["path"])]
        half = plan == "diffusers" and set(rec.get("param_dtypes") or {}) == {"F32"} and not variant
        disk = sum(f["size"] for f in wfiles)
        cfg = rec.get("config") or {}
        quantised = bool(cfg.get("quantization")) or any(k not in ("BF16", "F16", "F32") for k in (rec.get("param_dtypes") or {}))
        load = disk
        if plan in ("vlm", "retrieval", "vision", "asr", "tts", "timeseries") and params and not quantised:
            load = min(disk, params * 2)
        stored_fp32 = set(rec.get("param_dtypes") or {}) == {"F32"}
        if plan == "diffusers" and stored_fp32 and not out.get("variant"):
            load = disk / 2   # stored fp32, loaded bf16 (diffusion.py passes torch_dtype=bfloat16)
            out["dtype"] = "stored fp32, loaded bf16"
        out["disk_gib"] = disk / GIB
        w = load / GIB
        out["weights_gib"] = w
        kind = out["spec"].get("kind")
        comps = {}
        for f in wfiles:
            comps[f["path"].split("/")[0]] = comps.get(f["path"].split("/")[0], 0) + f["size"]
        largest = (max(comps.values()) / GIB) * (0.5 if half else 1.0) if comps else None
        out["hw"] = hardware("vllm" if plan == "vllm" else ("diffusers" if plan == "diffusers" else "transformers"), w, kind, largest)
        arch = (cfg.get("architectures") or [None])[0]
        out["architecture"] = arch or (rec.get("model_index") or {}).get("class")
        if rec.get("gated") and not reason:
            reason = "gated"
        if not reason:
            reason = _runtime_gap(plan, entry, cfg, rec, support, arch)
        if not reason and not wfiles:
            reason = "no weights files found"
    if out.get("hw") is None and not reason:
        reason = "needs more than 32 GPUs of 288 GiB"
    out["status"] = f"blocked: {reason}" if reason else "manifest written"
    out["workloads"] = reason is None
    if out["licence"]["unreadable"] and not reason:
        out["status"] = "blocked: licence unreadable"
        out["workloads"] = False
    if not out["workloads"] and (base.get("gated") or (hub["repos"].get(repo) or {}).get("gated")):
        out["access"] = _access(repo if (hub["repos"].get(repo) or {}).get("gated") else entry["base"], hub["repos"].get(repo) or base)
    return out


def _access(repo, rec):
    return f"accept the terms at https://huggingface.co/{repo} (gated: {rec.get('gated')}), then set HF_TOKEN"


def _runtime_gap(plan, entry, cfg, rec, support, arch):
    mtype = cfg.get("model_type")
    if plan == "vllm":
        if entry["spec"].get("mistral_format"):
            return None
        if arch and arch not in support["vllm"]["architectures"]:
            return f"vLLM v0.30.0 does not list architecture {arch}"
    if plan in ("vlm", "retrieval", "vision", "asr", "tts", "timeseries"):
        native = mtype in set(support["transformers"]["model_types"])
        if cfg.get("auto_map") and not native:
            return f"custom model code (auto_map {cfg['auto_map']}); model_type '{mtype}' is not in transformers 5.19.0 and trust_remote_code is not used"
        if plan == "vlm" and mtype and mtype not in set(support["transformers"]["image_text_to_text_types"]):
            return f"transformers 5.19.0 lists no image-text-to-text model_type '{mtype}'"
        if plan in ("retrieval", "vision", "asr") and mtype and not native and entry["spec"].get("engine") != "nemo":
            return f"transformers 5.19.0 lists no model_type '{mtype}'"
    if plan == "diffusers":
        mi = rec.get("model_index")
        if not mi:
            return "no diffusers layout (no model_index.json)"
        if mi["class"] not in set(support["diffusers"]["pipelines"]):
            return f"diffusers 0.41.0 has no {mi['class']}"
    return None


def _select_files(entry, rec):
    """(files, dtype label, variant) to download for a non-GGUF plan: the weights the model's own index (or single file)
    names, in the variant the pipeline loads, plus nothing else (small configs are downloaded by the workload by pattern)."""
    files = rec["files"]
    by_path = {f["path"]: f for f in files}
    plan = entry["plan"]
    if entry.get("allow"):
        pats = [re.compile(p) for p in entry["allow"]]
        return [f for f in files if any(p.search(f["path"]) for p in pats) and WEIGHT_FILE.search(f["path"])] + \
               [f for f in files if any(p.search(f["path"]) for p in pats) and not WEIGHT_FILE.search(f["path"])], "as stored", None
    idx = rec.get("index_shards") or {}

    def component_weights(d, variant):
        """The weight files of directory d (path prefix with trailing slash, '' for the top level) for a variant."""
        suffix = f".{variant}" if variant else ""
        for name in ([f"{d}diffusion_pytorch_model.safetensors.index{suffix}.json", f"{d}model.safetensors.index{suffix}.json"]):
            if name in idx:
                return [by_path[p] for p in idx[name] if p in by_path]
        for name in (f"{d}diffusion_pytorch_model{suffix}.safetensors", f"{d}model{suffix}.safetensors"):
            if name in by_path:
                return [by_path[name]]
        if not idx and not suffix:   # a repository whose contents are not readable (gated): shard names by pattern
            pat = re.compile(rf"^{re.escape(d)}(diffusion_pytorch_model|model)(-\d{{5}}-of-\d{{5}})?\.safetensors$")
            hit = [f for p, f in by_path.items() if pat.match(p)]
            if hit:
                return hit
        return None

    if plan == "diffusers":
        dirs = sorted({p.rsplit("/", 1)[0] + "/" for p in by_path if "/" in p and p.endswith(".safetensors")})
        variant = None
        sizes = {d: sum(f["size"] or 0 for f in (component_weights(d, None) or [])) for d in dirs}
        biggest = max(sizes, key=sizes.get) if sizes else None
        for v in ("bf16", "fp16"):   # a variant is used when the largest component has one; the others fall back to their plain files
            if biggest and component_weights(biggest, v):
                variant = v
                break
        keep = [by_path["model_index.json"]] if "model_index.json" in by_path else []
        for d in dirs:
            w = (component_weights(d, variant) if variant else None) or component_weights(d, None)
            if w:
                keep += w
        return keep, (f"{variant} variant" if variant else "as stored"), variant
    # transformers / vLLM / NeMo: the top-level weights
    if entry["spec"].get("mistral_format"):
        keep = [f for f in files if "/" not in f["path"] and (f["path"].startswith("consolidated") and f["path"].endswith(".safetensors"))]
        return keep, "as stored (Mistral format)", None
    if entry["spec"].get("engine") == "nemo":
        return [f for f in files if "/" not in f["path"] and f["path"].endswith(".nemo")], "as stored (.nemo)", None
    w = component_weights("", None)
    if w:
        return w, "as stored", None
    bins = [f for f in files if "/" not in f["path"] and re.match(r"pytorch_model.*\.bin$", f["path"])]
    return bins, "as stored (pytorch .bin)", None


def _licence(entry, hub, base, grec, grepo=None):
    """The licence record: label, restricted, evidence, readable. `base` is the official repo, `grec` the downloaded one."""
    label, spdx, permissive = licence_label(base)
    ev = [licence_evidence(entry["base"], base)]
    notes = []
    extra = hub["repos"].get(entry.get("licence_from")) if entry.get("licence_from") else None
    if extra and "error" not in extra:
        ev.append("licence also read from " + licence_evidence(entry["licence_from"], extra))
        elabel, espdx, eperm = licence_label(extra)
        if label is None or (spdx == "other" and espdx and espdx != "other"):   # an unnamed 'other' is replaced by a named licence
            label, spdx, permissive = elabel, espdx, eperm
        elif elabel and espdx != spdx and espdx != "other":
            notes.append(f"{entry['licence_from']} says {espdx}, {entry['base']} says {spdx}; the more restrictive is recorded")
            if permissive and not eperm:
                label, spdx, permissive = elabel, espdx, eperm
    url = entry.get("licence_text")
    if url and (hub.get("texts") or {}).get(url):
        ev.append(f"public licence text {url} begins '{hub['texts'][url][:100]}'")
    if grec is not None and grepo != entry["base"]:
        glabel, gspdx, gperm = licence_label(grec)
        txt = grec.get("card_has_licence_text")
        ev.append("downloaded repository " + licence_evidence(grepo, grec) + (f"; its card carries a licence agreement text: {txt}" if txt is not None else ""))
        if label is None and glabel:
            label, spdx, permissive = glabel, gspdx, gperm
            notes.append("the official repository names no licence; the label is the downloaded repository's own card")
        elif glabel and gspdx != spdx:
            if gspdx == "other":
                notes.append(f"the downloaded repository's card says 'license: other' without a name; the official repository's {spdx} is recorded")
            else:
                notes.append(f"the two cards disagree ({spdx} vs {gspdx}); the more restrictive is recorded")
                if permissive and not gperm:
                    label, spdx, permissive = glabel, gspdx, gperm
    bases = []
    for rec in (base, grec):
        bm = (rec or {}).get("base_model")
        bases += [bm] if isinstance(bm, str) else list(bm or [])
    derived = sorted({b for b in bases if b.startswith("meta-llama/")} if not entry["base"].startswith("meta-llama/") else set())
    if derived and permissive:
        notes.append(f"the card's base_model lists {', '.join(derived)}: the Llama community licence terms apply to derivatives of Llama models, "
                     f"so the model is treated as restricted although the card declares {label}")
        permissive = False
    readable = label is not None
    return {"licence": {"label": label or "unreadable", "id": spdx, "restricted": not permissive, "evidence": ev,
                        "notes": notes, "unreadable": not readable}}


def resolve_all():
    hub, support = load_hub(), load_support()
    return [resolve_entry(e, hub, support) for e in load_catalog()]
