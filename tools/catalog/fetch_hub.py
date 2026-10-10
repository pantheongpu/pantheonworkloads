#!/usr/bin/env python3
"""Read the public Hugging Face metadata of every repository the catalog names into tools/catalog/hub.json.

    tools/catalog/fetch_hub.py            fetch what is missing from hub.json
    tools/catalog/fetch_hub.py --refresh  fetch everything again

No credentials are used and nothing is downloaded except small text files (model card, LICENSE files,
config.json, model_index.json) of repositories that answer without a login. For each repository:

  * `GET https://huggingface.co/api/models/<id>?blobs=true`: commit sha, gated flag, tags, card data,
    parameter counts, GGUF header summary, and every file with its size and, for LFS files, its sha256;
  * for repositories that are not gated: the first bytes of README.md and of each LICENSE-like file at that
    commit, whether the card carries a licence agreement text, and config.json / model_index.json facts.

Gated repositories list their files but not their contents, so only the API part is recorded for them.
"""
import argparse
import concurrent.futures
import datetime
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

import yaml

HERE = pathlib.Path(__file__).resolve().parent
API = "https://huggingface.co/api/models/"
RAW = "https://huggingface.co/{repo}/raw/{sha}/{path}"
LICENCE_FILE = re.compile(r"(^|/)(licen[cs]e|notice|use_policy|acceptable)[^/]*$|licen[cs]e.*\.(txt|md)$", re.I)
PROVIDERS = ["ggml-org", "unsloth", "bartowski", "lmstudio-community"]


def http(url, limit=None):
    """GET with retries: the anonymous API rate limit answers 429, which is waited out, never recorded."""
    for attempt in range(12):
        req = urllib.request.Request(url, headers={"User-Agent": "pantheonworkloads-catalog"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(limit) if limit else r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 11:
                raise
            time.sleep(min(60, float(e.headers.get("Retry-After") or 0) or 5 * (attempt + 1)))
        except (urllib.error.URLError, TimeoutError):
            if attempt == 11:
                raise
            time.sleep(5)


def gguf_candidates(entry):
    """Candidate GGUF repositories for a llamacpp entry: the listed ones, else names derived from the base."""
    if entry.get("gguf"):
        return list(entry["gguf"])
    org, name = entry["base"].split("/")
    name = re.sub(r"-bf16$", "", name, flags=re.I)
    names = [name]
    if name.endswith("-Instruct"):
        names.append(name[: -len("-Instruct")])
    out = [f"{org}/{n}-GGUF" for n in names]
    for n in names:
        out += [f"ggml-org/{n}-GGUF", f"unsloth/{n}-GGUF", f"bartowski/{org}_{n}-GGUF", f"bartowski/{n}-GGUF",
                f"lmstudio-community/{n}-GGUF"]
    seen = []
    for c in out:
        if c not in seen:
            seen.append(c)
    return seen


def repos_of(entries):
    out = []
    for e in entries:
        for r in [e["base"], e.get("repo"), e.get("licence_from")] + (gguf_candidates(e) if e["plan"] in ("llamacpp", "llamacpp-embed") else []):
            if r and r not in out:
                out.append(r)
    return out


def _raw(repo, sha, path, limit):
    try:
        return http(RAW.format(repo=repo, sha=sha, path=path), limit).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return None
        raise


def index_shards(files, raw):
    """{path of each *.index.json (or index.<variant>.json): the shard file names its weight_map points at}."""
    out = {}
    for f in files:
        p = f["path"]
        if re.search(r"\.index(\.[a-z0-9]+)?\.json$", p) and (f["size"] or 0) < 20_000_000:
            try:
                wm = json.loads(raw(p, 20_000_000) or "{}").get("weight_map") or {}
            except ValueError:
                wm = {}
            d = p.rsplit("/", 1)[0] + "/" if "/" in p else ""
            out[p] = sorted({d + v for v in wm.values()})
    return out


def pack(hub):
    """Compact form for hub.json: each file as [path, size, sha256]."""
    out = {"fetched": hub["fetched"], "texts": hub.get("texts", {}), "repos": {}}
    for k, v in hub["repos"].items():
        v = dict(v)
        if "files" in v:
            v["files"] = [[f["path"], f["size"], f["sha256"]] for f in v["files"]]
        out["repos"][k] = v
    return out


def unpack(raw):
    for v in raw["repos"].values():
        if v.get("files") and isinstance(v["files"][0], list):
            v["files"] = [{"path": p, "size": s, "sha256": h} for p, s, h in v["files"]]
    return raw


def load(path=None):
    path = path or HERE / "hub.json"
    return unpack(json.loads(path.read_text(encoding="utf-8")))


def save(hub, path=None):
    path = path or HERE / "hub.json"
    packed = pack(hub)
    items = [f"{json.dumps(k)}: {json.dumps(v, sort_keys=True, separators=(',', ':'))}" for k, v in sorted(packed["repos"].items())]
    path.write_text('{"fetched": %s, "texts": %s, "repos": {\n%s\n}}\n' % (json.dumps(packed["fetched"]), json.dumps(packed["texts"], sort_keys=True), ",\n".join(items)), encoding="utf-8")


def front_matter(text):
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    return m.group(1) if m else ""


def fetch_repo(repo):
    try:
        d = json.loads(http(API + repo + "?blobs=true"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return {"error": e.code}
        raise
    card = d.get("cardData") or {}
    files = [{"path": s["rfilename"], "size": s.get("size"), "sha256": (s.get("lfs") or {}).get("sha256")}
             for s in d.get("siblings", [])]
    rec = {
        "sha": d["sha"], "gated": d.get("gated"), "private": d.get("private"),
        "last_modified": d.get("lastModified"), "pipeline_tag": d.get("pipeline_tag"), "library": d.get("library_name"),
        "tags_license": [t for t in d.get("tags", []) if t.startswith("license")],
        "license": card.get("license"), "license_name": card.get("license_name"), "license_link": card.get("license_link"),
        "base_model": card.get("base_model"),
        "params": (d.get("safetensors") or {}).get("total"),
        "param_dtypes": (d.get("safetensors") or {}).get("parameters"),
        "gguf": {k: v for k, v in (d.get("gguf") or {}).items() if k in ("total", "architecture", "context_length")} or None,
        "files": files,
    }
    if d.get("gated"):
        return rec
    sha = d["sha"]

    def raw(path, limit):
        try:
            return http(RAW.format(repo=repo, sha=sha, path=path), limit).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):   # not readable without a login, or not there: recorded as absent
                return None
            raise

    names = {f["path"] for f in files}
    if "README.md" in names:
        readme = raw("README.md", 200000)
        if readme is not None:
            rec["card_has_licence_text"] = bool(re.search(r"(?i)licen[cs]e agreement|terms of use|permission is hereby granted|apache license", readme))
            rec["card_front_matter_lines"] = len(front_matter(readme).splitlines())
            m = re.search(r"(?im)^#+\s*licen[cs]e\s*\n+(.{0,200})", readme)
            rec["card_licence_section"] = " ".join(m.group(1).split()) if m else None
    heads = {}
    for f in files:
        if LICENCE_FILE.search(f["path"]) and "/" not in f["path"] and (f["size"] or 0) < 400000:
            t = raw(f["path"], 400)
            if t is not None:
                heads[f["path"]] = " ".join(t.split())[:140]
    rec["licence_files"] = heads
    if "config.json" in names:
        try:
            cfg = json.loads(raw("config.json", 400000) or "{}")
            rec["config"] = {"architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"),
                             "auto_map": sorted(cfg["auto_map"]) if isinstance(cfg.get("auto_map"), dict) else None,
                             "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
                             "quantization": (cfg.get("quantization_config") or {}).get("quant_method")}
        except ValueError:
            pass
    rec["index_shards"] = index_shards(files, raw)
    if "model_index.json" in names:
        try:
            mi = json.loads(raw("model_index.json", 100000) or "{}")
            rec["model_index"] = {"class": mi.get("_class_name"), "diffusers_version": mi.get("_diffusers_version"),
                                  "components": sorted(k for k in mi if not k.startswith("_"))}
        except ValueError:
            pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--date", help="date to record (default: today)")
    a = ap.parse_args()
    entries = yaml.safe_load((HERE / "catalog.yaml").read_text(encoding="utf-8"))["entries"]
    path = HERE / "hub.json"
    hub = load(path) if path.exists() and not a.refresh else {"fetched": None, "repos": {}}
    for repo, rec in hub["repos"].items():   # records written before index_shards existed
        if "error" not in rec and not rec.get("gated") and "index_shards" not in rec:
            rec["index_shards"] = index_shards(rec["files"], lambda p, n, repo=repo, sha=rec["sha"]: _raw(repo, sha, p, n))
    hub.setdefault("texts", {})
    for url in sorted({e["licence_text"] for e in entries if e.get("licence_text")} - set(hub["texts"])):
        hub["texts"][url] = " ".join(http(url, 600).decode("utf-8", "replace").split())[:160]
    todo = [r for r in repos_of(entries) if r not in hub["repos"]]
    print(f"{len(todo)} repositories to fetch", file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        for repo, rec in zip(todo, pool.map(fetch_repo, todo)):
            hub["repos"][repo] = rec
    hub["fetched"] = (a.date or datetime.date.today().isoformat()) if not hub["fetched"] or todo else hub["fetched"]
    hub["repos"] = dict(sorted(hub["repos"].items()))
    save(hub, path)
    missing = [r for r in repos_of(entries) if "error" in hub["repos"][r]]
    print(f"wrote {path} ({len(hub['repos'])} repositories; {len(missing)} answered an error)", file=sys.stderr)


if __name__ == "__main__":
    main()
