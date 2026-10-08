"""Coverage table: one row per workload, derived from the repository (manifests, reference.json,
model.sha256 files, bench/ records and the results tables in docs/*validation*.md).

    bin/pw coverage            print the markdown
    bin/pw coverage --write    rewrite the block between the coverage markers in README.md
    bin/pw coverage --check    exit 1 if README.md's block differs from the generated output

Nothing here is typed by hand: a cell changes only when the data it is read from changes.
"""
import fnmatch
import json
import pathlib
import re

import validate

START = "<!-- coverage:start -->"
END = "<!-- coverage:end -->"

FAMILIES = [
    ("llamacpp", "llama.cpp (synthetic GGUF, pretrained GGUF and llama-bench)"),
    ("pretrained", "Pretrained models on onnxruntime, spaCy, numpy and sentencepiece"),
    ("arch", "Architecture coverage (random-weight PyTorch)"),
    ("lib", "GPU-library coverage (random inputs, PyTorch)"),
    ("train", "Training"),
    ("other", "Other (Hugging Face PyTorch, whisper.cpp, vLLM, Ollama, MLPerf LoadGen, runner self-check)"),
]
PRETRAINED_RUNTIMES = {"onnxruntime", "spacy", "numpy", "sentencepiece"}
NO_WEIGHTS = re.compile(r"random|synthetic|no pretrained|weights=None", re.I)
SIM = re.compile(r"sim:(nvidia|amd)/(\{[^}]*\}|[\w*-]+)")
COLUMNS = ("name", "kind", "runtime", "targets", "reference", "licence", "pinned", "sim", "gpu")


def family(name, runtime):
    if name.startswith("llamacpp-") or runtime == "llama.cpp":
        return "llamacpp"
    if name.startswith("arch-"):
        return "arch"
    if name.startswith("lib-"):
        return "lib"
    if name.startswith("gpt-train-"):
        return "train"
    if runtime in PRETRAINED_RUNTIMES:
        return "pretrained"
    return "other"


def _cells(line):
    return [c.strip().strip("`") for c in line.strip().strip("|").split("|")]


def validation_rows(root):
    """Yield (workload patterns, target cell, result) for every results-table row in docs/*validation*.md."""
    for path in sorted((root / "docs").glob("*validation*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|"):
                continue
            c = _cells(line)
            if len(c) >= 3 and c[2] in ("PASS", "FAIL", "SKIP"):
                yield [w.strip().strip("`") for w in c[0].split(",")], c[1], c[2]


def passes(root, names):
    """name -> {'sim': {vendor: set(profiles)}, 'gpu': bool} from the PASS rows."""
    out = {n: {"sim": {}, "gpu": False} for n in names}
    for patterns, target, result in validation_rows(root):
        if result != "PASS":
            continue
        hit = [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
        for vendor, prof in SIM.findall(target):
            profiles = [p.strip() for p in prof.strip("{}").split(",")]
            for n in hit:
                out[n]["sim"].setdefault(vendor, set()).update(profiles)
        if re.match(r"gpu\b", target):
            for n in hit:
                out[n]["gpu"] = True
    return out


def short_targets(targets):
    out = []
    for t in targets:
        if t == "sim:*/*":
            out.append("sim:*")
        elif t.startswith("sim:") and t.endswith("/*"):
            out.append(t[:-2])
        else:
            out.append(t)
    return ", ".join(out)


def pinned(root, name, model):
    d = root / "workloads" / name
    if (d / "model.sha256").exists() or (d / "models.sha256").exists() \
            or (model and "sha256" in str(model.get("revision") or "")):
        return "yes"
    if model is None or NO_WEIGHTS.search(str(model.get("id", ""))):
        return "n/a"
    return "revision only" if model.get("revision") else "no"


def licence(model, raw=""):
    if model is None or NO_WEIGHTS.search(str(model.get("id", ""))):
        return "n/a (no weights)"
    text = str(model.get("licence") or "")
    short = re.split(r" \(|:", text, maxsplit=1)[0].strip() or "?"
    if short.upper() == "UNVERIFIED":
        return "UNVERIFIED"
    if re.search(r"\bnpm\b", str(model.get("id", ""))):
        short += " (second-hand)"
    elif re.search(r"model card", (re.search(r"^\s+licence:.*$", raw, re.M) or re.match(r"", "")).group(0)):
        short += " (per card)"
    return short


def rows(root):
    root = pathlib.Path(root)
    manifests = {}
    for p in sorted((root / "workloads").glob("*/manifest.yaml")):
        manifests[p.parent.name] = validate.load(p)
    seen = passes(root, list(manifests))
    result = []
    for n, m in manifests.items():
        ref = root / "workloads" / n / "reference.json"
        if ref.exists():
            where = json.loads(ref.read_text(encoding="utf-8")).get("recorded_on")
            reference = f"yes ({where})" if where else "yes (target not recorded)"
        else:
            reference = "n/a" if m.get("kind") == "benchmark" else "no"
        sim = seen[n]["sim"]
        has_bench = any((root / "bench" / n).glob("*.json"))
        gpu = [x for x, ok in (("bench", has_bench), ("functional", seen[n]["gpu"])) if ok]
        result.append({
            "name": n, "family": family(n, m.get("runtime")), "kind": m.get("kind"),
            "runtime": m.get("runtime"), "targets": short_targets(m.get("targets", [])),
            "reference": reference, "licence": licence(m.get("model"), (root / "workloads" / n / "manifest.yaml").read_text(encoding="utf-8")) + (" (restricted)" if validate.is_restricted(m) else ""), "pinned": pinned(root, n, m.get("model")),
            "sim": ", ".join(f"{v} x{len(p)}" for v, p in sorted(sim.items())) or "no", "sim_ok": bool(sim),
            "gpu": " + ".join(gpu) or "no", "gpu_ok": bool(gpu),
        })
    return result


def render(root):
    data = rows(pathlib.Path(root))
    out = []
    for key, title in FAMILIES:
        group = [r for r in data if r["family"] == key]
        if not group:
            continue
        out += [f"#### {title}: {len(group)}", "",
                "| Workload | Kind | Runtime | Targets | Reference | Model licence | Model pinned | Sim-validated | GPU-validated |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in group:
            out.append("| " + " | ".join(str(r[k]).replace("|", "/") for k in COLUMNS) + " |")
        out.append("")
    out += ["#### Totals", "",
            "| Family | Workloads | Functional | Benchmark | Reference recorded | Model sha256-pinned | Sim-validated | GPU-validated |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def tally(label, g):
        return (f"| {label} | {len(g)} | {sum(r['kind'] == 'functional' for r in g)} | "
                f"{sum(r['kind'] == 'benchmark' for r in g)} | {sum(r['reference'].startswith('yes') for r in g)} | "
                f"{sum(r['pinned'] == 'yes' for r in g)} | {sum(r['sim_ok'] for r in g)} | {sum(r['gpu_ok'] for r in g)} |")
    for key, _ in FAMILIES:
        g = [r for r in data if r["family"] == key]
        if g:
            out.append(tally(key, g))
    out.append(tally("**all**", data))
    return "\n".join(out) + "\n"


def block(root):
    return f"{START}\n\n{render(root)}\n{END}"


def split_readme(text):
    i, j = text.find(START), text.find(END)
    if i < 0 or j < i:
        return None
    return text[:i], text[i:j + len(END)], text[j + len(END):]


def check(root, readme=None):
    """Return a list of problems (empty when README.md's block equals the generated one)."""
    root = pathlib.Path(root)
    path = pathlib.Path(readme) if readme else root / "README.md"
    parts = split_readme(path.read_text(encoding="utf-8"))
    if parts is None:
        return [f"{path.name}: the {START} / {END} markers are missing"]
    if parts[1] != block(root):
        return [f"{path.name}: the coverage block is stale; run `bin/pw coverage --write`"]
    return []


def write(root, readme=None):
    root = pathlib.Path(root)
    path = pathlib.Path(readme) if readme else root / "README.md"
    before, _, after = split_readme(path.read_text(encoding="utf-8"))
    path.write_text(before + block(root) + after, encoding="utf-8")
