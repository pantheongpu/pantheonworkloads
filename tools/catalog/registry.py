"""The generated tables of docs/model-registry.md (between its registry markers)."""
import collections

import generate  # noqa: F401  (circular import is avoided: generate imports this module lazily)
import model_catalog as mc

FAMILY_ORDER = ["DeepSeek", "Meta Llama", "Qwen", "Google Gemma", "Mistral", "OpenAI gpt-oss", "GLM", "Kimi", "Cohere", "Falcon", "Microsoft Phi",
                "NVIDIA Nemotron", "MiniMax", "Other LLMs", "Image generation", "Video generation", "Speech", "Retrieval", "Vision", "OCR and documents",
                "Time series"]
PLAN_TEXT = {"llamacpp": "llama.cpp (GGUF)", "llamacpp-embed": "llama.cpp embedding (GGUF)", "vllm": "vLLM", "vlm": "transformers", "diffusers": "diffusers",
             "whispercpp": "whisper.cpp", "asr": "transformers / NeMo", "tts": "package", "retrieval": "transformers", "vision": "transformers", "timeseries": "chronos"}


def cell(s):
    return str(s).replace("|", "/").replace("\n", " ")


def row(r):
    gated = r.get("base_gated")
    if r.get("community"):
        gate = f"official: {gated}; downloaded copy: no"
    else:
        rec_gated = r.get("repo_gated", gated)
        gate = str(rec_gated) if rec_gated else "no"
    hw = r.get("hw")
    need = f"{hw['gpus']} x {hw['gpu_memory_gb']} GiB ({hw['formula']} = {hw['per_gpu_gib']})" if hw else "-"
    if hw and hw["scope"] != "single node":
        need += f"; {hw['scope']}"
    status = r["status"]
    if r.get("community"):
        status += f" (community GGUF; the official repo is gated, blocked-for-download: accept the terms at https://huggingface.co/{r['base']}, then set HF_TOKEN)"
    if r.get("access"):
        status += f". Access: {r['access']}"
    lic = r["licence"]["label"] + (" (restricted)" if r["licence"]["restricted"] else "")
    repo = r.get("repo", r["base"])
    ref = f"`{repo}` `{r['repo_sha']}`" if r.get("repo_sha") else f"`{r['base']}` `{r.get('base_sha', '?')}`"
    if repo != r["base"]:
        ref += f" (base `{r['base']}` `{r['base_sha']}`)"
    q = r.get("quant") or r.get("dtype") or "-"
    w = f"{r['weights_gib']:.1f}" if r.get("weights_gib") else "-"
    work = "-"
    if r["workloads"]:
        from generate import names
        f, b = names(r)
        work = f"`{f}`, `{b}`"
    return (f"| {cell(r['title'])} | {cell(ref)} | {mc.human_params(r.get('params'))} | {cell(lic)} | {cell(gate)} | {cell(PLAN_TEXT.get(r['plan'], r['plan']))} "
            f"| {cell(q)} | {w} | {cell(need)} | {cell(status)} | {work} |")


def tables(resolved):
    by = collections.OrderedDict((f, []) for f in FAMILY_ORDER)
    for r in resolved:
        by.setdefault(r["family"], []).append(r)
    out = []
    ok = [r for r in resolved if r["workloads"]]
    out += [f"**{len(resolved)} entries: {len(ok)} with workloads written (never run), {len(resolved) - len(ok)} blocked.** "
            f"Hub data read {mc.DATE}; runtimes: llama.cpp b11447, vLLM v0.30.0, transformers 5.19.0, diffusers 0.41.0.", "",
            "| Family | Entries | Workloads written | Blocked |", "| --- | ---: | ---: | ---: |"]
    for fam, rs in by.items():
        if rs:
            n_ok = sum(1 for r in rs if r["workloads"])
            out.append(f"| {fam} | {len(rs)} | {n_ok} | {len(rs) - n_ok} |")
    out += ["", "Blocked, by reason:", ""]
    reasons = collections.Counter(r["status"].split(":", 1)[1].strip().split(" (")[0].split(";")[0][:70] for r in resolved if not r["workloads"])
    for reason, n in reasons.most_common():
        out.append(f"- {n} x {reason}")
    out += ["", "Largest hardware needs among the written workloads (weights plus margin, per the formulas above):", ""]
    big = sorted((r for r in ok), key=lambda r: -r["weights_gib"])[:15]
    out += ["| Workload | Weights GiB | Needs | Total GPU memory |", "| --- | ---: | --- | ---: |"]
    from generate import names
    for r in big:
        hw = r["hw"]
        out.append(f"| `{names(r)[0]}` | {r['weights_gib']:.1f} | {hw['gpus']} x {hw['gpu_memory_gb']} GiB ({hw['scope']}) | {hw['gpus'] * hw['gpu_memory_gb']} GiB |")
    for fam, rs in by.items():
        if not rs:
            continue
        out += ["", f"### {fam}", "",
                "| Model | Repository and commit (weights) | Params | Licence as read | Gated | Plan | Quant / dtype | Weights GiB | Hardware need (arithmetic: GiB per GPU) | Status | Workloads |",
                "| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | --- | --- |"]
        out += [row(r) for r in rs]
    return "\n".join(out) + "\n"


def skeleton():
    return "# Model registry\n\n" + generate.MARK_START + "\n" + generate.MARK_END + "\n"
