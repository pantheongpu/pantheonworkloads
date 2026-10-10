#!/usr/bin/env python3
"""Generate the model-catalog workloads and the registry tables from catalog.yaml + hub.json + support.json.

    tools/catalog/generate.py            write workloads/<name>/ for every unblocked entry and the tables of docs/model-registry.md
    tools/catalog/generate.py --check    exit 1 when anything on disk differs from what would be written

Every written workload starts as WRITTEN, NEVER RUN: no weights were downloaded, no GPU was used, there is no reference.json
and no bench record (the coverage table says so). A workload directory not listed by the catalog is never touched.

A workload that has had its first real run *graduates*: its manifest notes then start with `VERIFIED ON` (a reference and bench
record exist, from a real GPU) or `BLOCKED(run)` (the first run showed it cannot work as pinned; the notes carry the evidence).
From then on its files are maintained by hand (run-script and pin fixes live in the workload directory) and this generator
neither writes nor checks them; the registry tables show the state read from the manifest. Never put those prefixes on a
workload that has not been run.
"""
import argparse
import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import model_catalog as mc  # noqa: E402

RUNTIME_VERSION = "llama.cpp b11447 (da263e7275dfbaeefcd61504eaa4fd5247540e11)"
VLLM_VERSION = "vLLM v0.30.0"
MARK_START, MARK_END = "<!-- registry:start -->", "<!-- registry:end -->"
EXAMPLES = "https://huggingface.co/"
GRADUATED = ("VERIFIED ON", "BLOCKED(run)")   # prefixes of the notes of a workload that has been run (see the module docstring)


def run_state(name):
    """'written' (never run), 'verified' or 'blocked' (a first run happened), read from the workload's manifest notes; 'written' when there is no manifest."""
    path = ROOT / "workloads" / name / "manifest.yaml"
    if not path.exists():
        return "written"
    import yaml
    notes = str((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("notes") or "")
    if notes.startswith("VERIFIED ON"):
        return "verified"
    if notes.startswith("BLOCKED(run)"):
        return "blocked"
    return "written"


def q(s):
    """A YAML double-quoted scalar (JSON strings are valid YAML)."""
    return json.dumps(s, ensure_ascii=False)


def sha_short(s):
    return s[:8]


def timeout_s(r, bench):
    base = min(172800, max(3600, int(r["weights_gib"] * 12) + 3600))
    return base * 2 if bench else base


def names(r):
    """(functional name, bench name) of the workloads of a resolved entry."""
    k, plan = r["key"], r["plan"]
    if plan in ("llamacpp", "llamacpp-embed"):
        return f"llamacpp-{k}", f"llamacpp-bench-{k}"
    if plan == "vllm":
        return f"vllm-{k}", f"vllm-bench-{k}"
    if plan == "whispercpp":
        stem = k.removeprefix("whisper-")
        return f"whisper-cpp-{stem}", f"whisper-cpp-bench-{stem}"
    if plan == "diffusers":
        return f"{k}-diffusers", f"{k}-bench"
    return f"{k}-pytorch", f"{k}-bench"


def sums_text(r, header):
    lines = [header]
    for f in sorted(r["files"], key=lambda f: f["path"]):
        if f["sha256"]:
            lines.append(f"{f['sha256']}  {f['path']}")
    return "\n".join(lines) + "\n"


def evidence_text(r):
    lic = r["licence"]
    text = f"Licence: {lic['label']}. Read {mc.DATE} from the Hub without credentials: " + " | ".join(lic["evidence"])
    if lic["notes"]:
        text += ". Note: " + "; ".join(lic["notes"])
    return text


def common_notes(r, bench):
    hw = r["hw"]
    parts = [
        "WRITTEN, NEVER RUN: nothing in this workload has been executed; no weights were downloaded and no GPU was used.",
        f"What was read ({mc.DATE}, public Hugging Face API and raw files, no credentials): commit, file names, sizes and LFS sha256 of the "
        f"{len([f for f in r['files'] if f['sha256']])} pinned file(s) (model.sha256, Hub oids, verified on every run).",
        evidence_text(r) + ".",
    ]
    if r["licence"]["restricted"]:
        parts.append(f"RESTRICTED: {r['licence']['label']} is a custom, non-OSI or use-based licence (rule in docs/model-registry.md), so the "
                     "workload is excluded from default selections (docs/manifest.md); name it to run it, and accept the licence yourself.")
    parts.append(f"Hardware: requires {hw['gpus']} x {hw['gpu_memory_gb']} GiB ({hw['scope']}); weights {r['weights_gib']:.1f} GiB; {hw['arithmetic']}. "
                 "These margins are assumptions, not measurements.")
    if r.get("community"):
        parts.append(f"Third-party conversion: the official {r['base']} is gated ({r['base_gated']}); the files come from {r['repo']}, "
                     "which is not gated, and whose card was read for the licence text.")
    if r.get("note"):
        parts.append(r["note"] + ".")
    return " ".join(parts)


def requires_block(r):
    hw = r["hw"]
    note = f"weights {r['weights_gib']:.1f} GiB; {hw['arithmetic']}"
    if hw["scope"] != "single node":
        note += f"; {hw['scope']}: this repository's run.sh does not span nodes"
    return f"requires:\n  gpus: {hw['gpus']}\n  gpu_memory_gb: {hw['gpu_memory_gb']}\n  notes: {q(note)}\n"


def manifest(r, bench, name, extra):
    """extra: dict with description, runtime, compare lines (list of str), runtime_version, model_id, revision, notes."""
    out = [f"name: {name}", f"description: {q(extra['description'])}", f"kind: {'benchmark' if bench else 'functional'}",
           f"runtime: {extra['runtime']}"]
    if extra.get("runtime_version"):
        out.append(f"runtime_version: {q(extra['runtime_version'])}")
    out.append("targets: [gpu]")
    if not bench:
        out += extra["compare"]
    out.append(f"timeout_s: {timeout_s(r, bench)}")
    if r["licence"]["restricted"]:
        out.append("restricted: true")
    out.append(requires_block(r).rstrip("\n"))
    out += ["model:", f"  id: {q(extra['model_id'])}", f"  revision: {q(extra['revision'])}", f"  licence: {q(r['licence']['label'])}",
            f"  source_url: {EXAMPLES}{r['repo']}", f"notes: {q(extra['notes'])}"]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ llama.cpp

def llamacpp(r):
    f_name, b_name = names(r)
    hw = r["hw"]
    files = r["files"]
    n = len(files)
    shas = f"; {n} file(s), model.sha256 lists them" if n > 1 else f"; file sha256 {files[0]['sha256']}"
    model_id = (f"{r['repo']} ({', '.join(f['path'].rsplit('/', 1)[-1] for f in files[:1])}{' + %d more shard(s)' % (n - 1) if n > 1 else ''}; "
                f"{r['quant']} GGUF of {r['base']})")
    revision = f"{r['repo_sha']} (HF commit of the GGUF repo){shas}; base model {r['base']} @ {r['base_sha']}"
    runtime = "llama.cpp"
    out = {}
    for bench, name in ((False, f_name), (True, b_name)):
        notes = common_notes(r, bench)
        if r["plan"] == "llamacpp-embed":
            task = ("Task: llama-embedding embeds the four fixed texts of llamacpp-qwen3-embedding-4b (instruction-format query + three passages), "
                    "last-token pooling, L2 normalisation; output = cosine similarities and the passage ranking (abs tolerance 0.005).")
        else:
            task = ('Task: prompt "The capital of France is", -n 3, --temp 0, seed 1, no chat template, -c 256, -ngl 99 (layers split over the '
                    f"GPUs llama.cpp sees, --split-mode layer by default); output compared exactly once a reference is recorded.")
        if bench:
            task = ("Task: llama-bench pp512 and tg128 (tools/llamacpp/bench.sh), -ngl 99, layers split over the GPUs llama.cpp sees; metrics "
                    "pp512_tokens_per_s and tg128_tokens_per_s. Real GPUs only.")
        notes += (f" {task} Build and knobs: docs/llamacpp.md. Not verified: that llama.cpp b11447 loads these files (the GGUF header declares architecture "
                  f"'{r['gguf_arch']}', which the pinned source lists), the output, speed, real memory use. No reference.json" + ("" if not bench else ", no bench record") + ".")
        desc = (f"llama-bench prompt-processing and token-generation tokens/s for {r['title']} ({r['quant']} GGUF, {hw['gpus']} x {hw['gpu_memory_gb']} GiB)." if bench else
                (f"llama.cpp (llama-embedding) embeds four fixed texts with {r['title']} ({r['quant']} GGUF)." if r["plan"] == "llamacpp-embed"
                 else f"llama.cpp (llama-completion) greedy-decodes 3 tokens from {r['title']} ({r['quant']} GGUF)."))
        compare = (["compare: tolerance", "tolerance:", "  abs: 0.005", "  rel: 0.0"] if r["plan"] == "llamacpp-embed" else ["compare: exact"])
        out[name] = {"manifest": manifest(r, bench, name, {"description": desc, "runtime": runtime, "runtime_version": RUNTIME_VERSION,
                                                           "compare": compare, "model_id": model_id, "revision": revision, "notes": notes}),
                     "run.sh": llamacpp_run(r, bench, name),
                     "model.sha256": sums_text(r, f"# {r['repo']} @ {r['repo_sha']} (LFS oids of the GGUF file(s) from the Hub API, read {mc.DATE}; never downloaded when this was written)")}
    return out


def llamacpp_run(r, bench, name):
    hw = r["hw"]
    disk = math.ceil(r["weights_gib"])
    head = [
        "#!/usr/bin/env bash",
        f"# {name}: " + ("llama-bench tokens/s" if bench else ("llama-embedding on four fixed texts" if r["plan"] == "llamacpp-embed" else "llama-completion greedy-decodes 3 tokens"))
        + f" with {r['title']} ({r['quant']} GGUF, {r['weights_gib']:.1f} GiB, {hw['gpus']} x {hw['gpu_memory_gb']} GiB GPU(s)).",
        "# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md.",
        "# Backends, build and knobs: docs/llamacpp.md and tools/llamacpp/common.sh.",
        "#",
        "#   PW_MODEL_FILE         your own first GGUF shard instead of downloading (not checked against model.sha256)",
        "#   PW_CACHE              where models and builds are kept; the weights need about %d GiB there" % disk,
        "#   PW_IGNORE_REQUIRES=1  skip the GPU count / memory check",
        "set -uo pipefail",
        'here="$(cd "$(dirname "$0")" && pwd)"',
        '. "$here/../../tools/llamacpp/common.sh"',
        f'[[ "$PW_TARGET" == gpu ]] || lc_skip "target $PW_TARGET: {r["title"]} needs real GPUs ({hw["gpus"]} x {hw["gpu_memory_gb"]} GiB)"',
        f"lc_require_gpus {hw['gpus']} {hw['gpu_memory_gb']}",
    ]
    if r["plan"] == "llamacpp-embed":
        head.append('export LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS="${LLAMACPP_EXTRA_TARGETS:-llama-embedding}"')
    head.append("lc_setup")
    if r["plan"] == "llamacpp-embed" and not bench:
        head.append('[[ -x "$LC_BIN/llama-embedding" ]] || lc_skip "no llama-embedding in $LC_BIN (build with LLAMACPP_BUILD_EXAMPLES=1 LLAMACPP_EXTRA_TARGETS=llama-embedding, see tools/llamacpp/build.sh)"')
    head.append(f'lc_fetch_shards "https://huggingface.co/{r["repo"]}/resolve/{r["repo_sha"]}" "$here/model.sha256" {disk}')
    if bench:
        return "\n".join(head + ['. "$here/../../tools/llamacpp/bench.sh"', "lc_bench"]) + "\n"
    if r["plan"] == "llamacpp-embed":
        body = open(ROOT / "workloads/llamacpp-qwen3-embedding-4b/run.sh", encoding="utf-8").read()
        tail = body[body.index("# Text 0 is a query"):]
        return "\n".join(head) + "\n\n" + tail
    body = [
        "",
        "# Greedy (temp 0), fixed seed, no chat template (-no-cnv), prompt not echoed: stdout is the 3 new tokens.",
        'env "${LC_ENV[@]}" "$LC_BIN/llama-completion" -m "$LC_MODEL" -p "The capital of France is" -n 3 \\',
        '  --temp 0 -s 1 -no-cnv --no-display-prompt -ngl "$LC_NGL" -t "${PW_THREADS:-2}" -c 256 \\',
        '  >"$PW_OUT/completion.out" 2>"$PW_OUT/completion.err" \\',
        '  || { echo "llama-completion failed" >&2; tail -20 "$PW_OUT/completion.err" >&2; exit 1; }',
        'PW_OUTFILE="$PW_OUT/completion.out" PW_DETAIL="llama.cpp $LLAMACPP_TAG $LC_BACKEND, $(basename "$LC_MODEL")" python3 - <<\'PY\'',
        "import json, os",
        'text = open(os.environ["PW_OUTFILE"], encoding="utf-8", errors="replace").read()',
        'print(json.dumps({"output": text, "detail": os.environ["PW_DETAIL"], "metrics": {}}))',
        "PY",
    ]
    return "\n".join(head + body) + "\n"


# ------------------------------------------------------------------ vLLM

def vllm(r):
    f_name, b_name = names(r)
    hw = r["hw"]
    files = [f for f in r["files"] if f["sha256"]]
    model_id = f"{r['repo']} (vLLM, tensor parallel {hw['gpus']}, {r['dtype']}; {len(files)} weights file(s))"
    revision = f"{r['repo_sha']} (HF commit); the weights files are pinned by sha256 in model.sha256 and verified on every run"
    out = {}
    for bench, name in ((False, f_name), (True, b_name)):
        notes = common_notes(r, bench)
        if bench:
            notes += (f" Task: `vllm bench throughput` ({VLLM_VERSION}) on the verified local snapshot: random dataset, 64 prompts of 128 input and 128 output tokens, "
                      "--max-model-len 4096, tensor parallel = requires.gpus; metrics requests_per_s, total_tokens_per_s, output_tokens_per_s. Real GPUs only.")
        else:
            notes += (f' Task: {VLLM_VERSION} greedy-decodes 8 tokens after "The capital of France is" (eager mode, max_model_len 4096, gpu_memory_utilization 0.90, '
                      "seed 0, temperature 0); output compared exactly once a reference is recorded.")
        notes += (f" Not verified: that vLLM v0.30.0 loads this checkpoint on the GPUs it will meet (the config declares architecture "
                  f"{r.get('architecture')}, which the pinned registry lists" + ("; Mistral-format repositories carry no config.json, so the architecture was not checked" if r["spec"].get("mistral_format") else "") +
                  "), quantisation kernels for the GPU generation, the output, speed. No reference.json" + (", no bench record" if bench else "") + ".")
        desc = (f"vLLM's own throughput benchmark on {r['title']} ({hw['gpus']} x {hw['gpu_memory_gb']} GiB)." if bench else
                f"vLLM greedy-decodes 8 tokens from {r['title']} (tensor parallel {hw['gpus']}).")
        out[name] = {"manifest": manifest(r, bench, name, {"description": desc, "runtime": "vllm", "compare": ["compare: exact"], "model_id": model_id,
                                                           "revision": revision, "notes": notes}),
                     "run.sh": vllm_run(r, bench, name),
                     "model.sha256": sums_text(r, f"# {r['repo']} @ {r['repo_sha']} (LFS oids from the Hub API, read {mc.DATE}; never downloaded when this was written)")}
    return out


def vllm_run(r, bench, name):
    hw = r["hw"]
    disk = math.ceil(r["disk_gib"])
    mistral = bool(r["spec"].get("mistral_format"))
    lines = [
        "#!/usr/bin/env bash",
        f"# {name}: " + ("`vllm bench throughput`" if bench else "vLLM greedy decode of 8 tokens") + f" with {r['title']}, tensor parallel {hw['gpus']}.",
        "# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md. Exits 77, with the reason, when the host lacks the GPUs,",
        "# the disk, torch or vLLM, or cannot fetch the model. The pinned snapshot is downloaded and sha256-verified by workloads/_shared/vllm_catalog.py",
        "# before vLLM starts, so vLLM loads from the verified local copy.",
        "#   PW_PYTHON  Python with vLLM (default python3)    PW_CACHE  model cache (default ~/.cache/pantheonworkloads)    PW_IGNORE_REQUIRES=1",
        "set -o pipefail",
        'source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"',
        f'[[ "$PW_TARGET" == gpu ]] || pw_skip "target $PW_TARGET: {r["title"]} needs real GPUs ({hw["gpus"]} x {hw["gpu_memory_gb"]} GiB)"',
        'source "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"',
        f"pw_require_gpus {hw['gpus']} {hw['gpu_memory_gb']}",
        f"pw_require_disk {disk}",
        "pw_setup vllm",
        '''"$PW_PY" -c 'import vllm' 2>/dev/null || pw_skip "$PW_PY cannot import vllm (pip install vllm==0.30.0)"''',
        f'export PW_VLLM_MODEL="{r["repo"]}" PW_VLLM_REVISION="{r["repo_sha"]}" PW_VLLM_TP={hw["gpus"]} PW_VLLM_MAX_LEN=4096' + (" PW_VLLM_MISTRAL=1" if mistral else ""),
        'export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # the SmolLM2 workload needed this on its first run (ninja was not on PATH)',
    ]
    if not bench:
        lines += [
            'result="$PW_OUT/vllm-catalog-result.json"',
            'rm -f "$result"',
            'pw_python "$PW_WORKLOAD_DIR/../_shared/vllm_catalog.py" "$result" >"$PW_OUT/vllm-catalog.log" 2>&1',
            "status=$?",
            'if [[ $status == 77 ]]; then',
            '  grep \'^SKIP\' "$PW_OUT/vllm-catalog.log" | tail -1 || echo "SKIP: vLLM cannot run here"',
            "  exit 77",
            "fi",
            'if [[ $status != 0 || ! -s "$result" ]]; then',
            '  tail -15 "$PW_OUT/vllm-catalog.log" >&2',
            '  echo "vLLM failed (exit $status)" >&2',
            "  exit 1",
            "fi",
            'cat "$result"; echo',
        ]
    else:
        flags = " --tokenizer-mode mistral --config-format mistral --load-format mistral" if mistral else ""
        lines += [
            '[[ "$PW_TARGET" == gpu ]] || pw_skip "benchmark numbers need a real GPU"',
            'pathfile="$PW_OUT/snapshot-path.txt"',
            'pw_python "$PW_WORKLOAD_DIR/../_shared/vllm_catalog.py" --prefetch "$pathfile" >"$PW_OUT/vllm-prefetch.log" 2>&1',
            "status=$?",
            'if [[ $status == 77 ]]; then grep \'^SKIP\' "$PW_OUT/vllm-prefetch.log" | tail -1 || echo "SKIP: cannot fetch the model"; exit 77; fi',
            'if [[ $status != 0 || ! -s "$pathfile" ]]; then tail -15 "$PW_OUT/vllm-prefetch.log" >&2; echo "snapshot failed (exit $status)" >&2; exit 1; fi',
            'snap=$(cat "$pathfile")',
            'json="$PW_OUT/vllm-throughput.json"',
            'rm -f "$json"',
            f'pw_python -m vllm.entrypoints.cli.main bench throughput --model "$snap" --tensor-parallel-size {hw["gpus"]} --max-model-len 4096 --dtype auto --seed 0 --backend vllm{flags} \\',
            '  --dataset-name random --random-input-len 128 --random-output-len 128 --num-prompts 64 --num-warmups 8 --output-json "$json" >"$PW_OUT/vllm-bench.log" 2>&1',
            "status=$?",
            'if [[ $status != 0 || ! -s "$json" ]]; then tail -15 "$PW_OUT/vllm-bench.log" >&2; echo "vllm bench throughput failed (exit $status)" >&2; exit 1; fi',
            'PW_JSON="$json" PW_LOG="$PW_OUT/vllm-bench.log" PW_MODEL_USED="$PW_VLLM_MODEL" "$PW_PY" - <<\'PY\'',
            "import json, os, re",
            'r = json.load(open(os.environ["PW_JSON"]))',
            'log = open(os.environ["PW_LOG"], errors="replace").read()',
            'm = re.findall(r"Throughput: ([\\d.]+) requests/s, ([\\d.]+) total tokens/s, ([\\d.]+) output tokens/s", log)',
            'metrics = {"requests_per_s": round(r["requests_per_second"], 3), "total_tokens_per_s": round(r["tokens_per_second"], 1)}',
            "if m:",
            '    metrics["output_tokens_per_s"] = float(m[-1][2])',
            'print(json.dumps({"output": f"{r[\'num_requests\']} requests, 64 prompts of 128 in / 128 out tokens", '
            '"detail": f"{os.environ[\'PW_MODEL_USED\']}, elapsed {r[\'elapsed_time\']:.2f} s", "metrics": metrics}))',
            "PY",
        ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ PyTorch plans (vlm, diffusers, retrieval, vision, asr, tts, timeseries)

PY_REQ = {"vlm": "requirements-vlm.txt", "retrieval": "requirements-vlm.txt", "vision": "requirements-vlm.txt", "diffusers": "requirements-diffusion.txt",
          "asr": "requirements-vlm.txt", "tts": "requirements-speech.txt", "timeseries": "requirements-speech.txt"}
BENCH_ENV = {"vlm": "PW_VLM_MODE", "diffusers": "PW_DIFFUSION_MODE", "retrieval": "PW_RETRIEVAL_MODE", "vision": "PW_VISION_MODE",
             "asr": "PW_ASR_MODE", "tts": "PW_TTS_MODE", "timeseries": "PW_TS_MODE"}


def py_modules(r):
    plan, spec = r["plan"], r["spec"]
    if plan == "vlm":
        return "transformers accelerate PIL"
    if plan == "diffusers":
        return "diffusers accelerate"
    if plan in ("retrieval", "vision"):
        return "transformers accelerate PIL" if plan == "vision" else "transformers accelerate"
    if plan == "asr":
        return "nemo" if spec.get("engine") == "nemo" else "transformers accelerate"
    if plan == "tts":
        return "chatterbox" if spec.get("engine") == "chatterbox" else "f5_tts"
    return "chronos"


def py_requirements(r):
    if r["plan"] == "asr" and r["spec"].get("engine") == "nemo":
        return "requirements-speech.txt"
    return PY_REQ[r["plan"]]


def py_main(r, name):
    plan, spec = r["plan"], r["spec"]
    repo, sha, key = r["repo"], r["repo_sha"], r["key"]
    head = [f'"""{name}: {r["title"]}. Shared body: ../_pytorch/%s.py (functional and bench modes). WRITTEN, NEVER RUN."""' % {
        "vlm": "vlm", "diffusers": "diffusion", "retrieval": "retrieval", "vision": "vision", "asr": "asr", "tts": "tts", "timeseries": "timeseries"}[plan],
            "import os", "import sys", "",
            'sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))']
    if plan == "vlm":
        ocr = spec.get("image") == "text"
        head += ["import vlm  # noqa: E402", ""]
        prompt = spec.get("prompt", "Describe this image in one sentence.")
        call = (f'vlm.run("{repo}", "{sha}", dtype="{spec.get("dtype", "bfloat16")}", new={spec.get("new", 32)}, margin=0.5, label="{key}", '
                f'prompt={prompt!r}, image="{spec.get("image", "astronaut")}", expect={["quick", "brown", "fox"] if ocr else None}, '
                'extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)')
    elif plan == "diffusers":
        head += ["import diffusion  # noqa: E402", ""]
        kind = spec["kind"]
        kw = dict(spec.get("kwargs") or {})
        guidance = spec.get("guidance", 0.0)
        negative = spec.get("negative", key.startswith(("wan", "cogvideox", "mochi", "ltx")))
        size = f"size={spec['size']}, " if kind == "image" else f"width={spec['width']}, height={spec['height']}, frames={spec['frames']}, "
        call = (f'diffusion.run("{repo}", "{sha}", kind="{kind}", steps={spec["steps"]}, guidance={guidance!r}, {size}variant={r.get("variant")!r}, '
                f'label="{key}", kwargs={kw!r}, negative={negative!r})')
    elif plan == "retrieval":
        head += ["import retrieval  # noqa: E402", ""]
        call = f'retrieval.run("{repo}", "{sha}", head="{spec["head"]}", label="{key}")'
    elif plan == "vision":
        head += ["import vision  # noqa: E402", ""]
        call = f'vision.run("{repo}", "{sha}", head="{spec["head"]}", label="{key}")'
    elif plan == "asr":
        head += ["import asr  # noqa: E402", ""]
        call = f'asr.run("{repo}", "{sha}", engine="{spec["engine"]}", label="{key}")'
    elif plan == "tts":
        head += ["import tts  # noqa: E402", ""]
        call = f'tts.run("{repo}", "{sha}", engine="{spec["engine"]}", label="{key}")'
    else:
        head += ["import timeseries  # noqa: E402", ""]
        call = f'timeseries.run("{repo}", "{sha}", pipeline="{spec["pipeline"]}", label="{key}")'
    return "\n".join(head) + call + "\n"


def py_run(r, bench, name, f_name):
    hw = r["hw"]
    disk = math.ceil(r["disk_gib"])
    lines = [
        "#!/usr/bin/env bash",
        f"# {name}: {r['title']} on PyTorch, " + ("bench mode" if bench else "functional") + ". WRITTEN, NEVER RUN (docs/model-registry.md).",
        "# Contract: docs/workload-contract.md; PyTorch discovery/install: ../_pytorch/env.sh. Exits 77 when the host lacks the GPUs, the disk,",
        "# PyTorch or the model download. Real GPUs only.",
        "set -uo pipefail",
        f'[[ "${{PW_TARGET:-}}" == gpu ]] || {{ echo "SKIP: target ${{PW_TARGET:-?}}: {r["title"]} needs real GPUs ({hw["gpus"]} x {hw["gpu_memory_gb"]} GiB)"; exit 77; }}',
        '. "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"',
        f"pw_require_gpus {hw['gpus']} {hw['gpu_memory_gb']}",
        f"pw_require_disk {disk}",
        f'export PW_TORCH_EXTRA_REQ="$PW_WORKLOAD_DIR/../_pytorch/{py_requirements(r)}"',
    ]
    if bench:
        lines.append(f"export {BENCH_ENV[r['plan']]}=bench")
    lines.append('source "$PW_WORKLOAD_DIR/../_pytorch/env.sh"')
    main = '"$PW_WORKLOAD_DIR/main.py"' if not bench else f'"$PW_WORKLOAD_DIR/../{f_name}/main.py"'
    lines.append(f"pw_torch_run {main} {py_modules(r)}")
    return "\n".join(lines) + "\n"


def py_compare(r):
    plan, spec = r["plan"], r["spec"]
    if plan == "vlm":
        return ["compare: exact"]
    if plan == "diffusers":
        return ["compare: tolerance", "tolerance:", f"  abs: {0.05 if spec['kind'] == 'video' else 0.03}", "  rel: 0.0"]
    if plan == "asr":
        return ["compare: tolerance", "tolerance:", "  abs: 0.0", "  rel: 0.0", "  fields:", "    text: {wer: 0.1}"]
    if plan == "timeseries":
        return ["compare: tolerance", "tolerance:", "  abs: 0.2", "  rel: 0.05"]
    return ["compare: tolerance", "tolerance:", "  abs: 0.03", "  rel: 0.0"]


def py_task(r, bench):
    plan, spec = r["plan"], r["spec"]
    if plan == "vlm":
        if spec.get("image") == "text":
            what = ('reads two lines of text drawn at run time with Pillow ("The quick brown fox" / "jumps over 13 lazy dogs.") with greedy decoding; the run also fails when '
                    'the words quick, brown and fox are not in the output, whatever a reference says')
        else:
            what = "describes scikit-image's public-domain astronaut.png (sha256 pinned, fetched at run time) in one sentence with greedy decoding"
        return (f"Task: {what}; text and token ids compared exactly; a decoding step whose top-1 and top-2 logits are within 0.5 (bf16 noise) fails the run. "
                "Weights are loaded bf16 with device_map=auto and the run fails if anything would be offloaded to the CPU. bf16 reasoned, not measured. "
                "If the model's processor or chat template differs from the generic image+text chat message, the first real run will fail and need a model-specific prompt.")
    if plan == "diffusers":
        return (f"Task: one {spec['kind']} from a fixed prompt and seed ({spec['steps']} steps, guidance {spec.get('guidance', 0.0)}); the output is statistics (channel mean/std, 8x8 block means, "
                "gradients" + (", frame-to-frame change" if spec["kind"] == "video" else "") + "), never pixels, with an abs tolerance that is a guess until two GPUs have been compared. "
                "Weights are loaded bf16; with more than one GPU diffusers' device_map=balanced places whole components, so the largest component must fit one GPU.")
    if plan == "retrieval":
        return ("Task: the four fixed texts of llamacpp-qwen3-embedding-4b (a query about the capital of France and three passages); output = cosine similarities / scores and the passage "
                f"ranking, head '{spec['head']}'; the run fails when the Paris passage is not first.")
    if plan == "vision":
        return f"Task: head '{spec['head']}' on scikit-image's public-domain astronaut.png (sha256 pinned, fetched at run time); see workloads/_pytorch/vision.py for the statistics compared."
    if plan == "asr":
        return ("Task: transcribe the 11 s clip jfk.wav of whisper.cpp (pinned commit and sha256, fetched at run time; its provenance is stated nowhere, see docs/models.md); output = the "
                "transcript with only lower-case letters, compared by word error rate (<= 0.1 against a reference) and checked against the known words (<= 0.15) on every run.")
    if plan == "tts":
        return ("Task: synthesise one fixed sentence; the waveform is checked (finite, 1 s or longer, not silent), output = duration rounded to 0.5 s and the RMS level, abs tolerance 0.03. "
                "The generator is sampled, so this is a plausibility check, not a regression test of the audio.")
    return "Task: 24-step forecast of a synthetic seasonal series made with numpy; the run fails when the forecast does not follow the season (correlation >= 0.8)."


def pytorch_plan(r):
    f_name, b_name = names(r)
    hw = r["hw"]
    plan = r["plan"]
    files = [f for f in r["files"] if f["sha256"]]
    model_id = f"{r['repo']} ({plan}; {len(files)} pinned weights file(s), {r['dtype']})"
    revision = f"{r['repo_sha']} (HF commit); the weights files are pinned by sha256 in model.sha256 and verified on every run"
    runtime = "pytorch"
    out = {}
    for bench, name in ((False, f_name), (True, b_name)):
        notes = common_notes(r, bench) + " " + py_task(r, bench)
        if bench:
            notes += " Bench mode of the same script: metrics are throughput and peak memory after warm-up; model load excluded."
        if r.get("architecture"):
            notes += f" Architecture {r['architecture']} is listed by the pinned runtime ({support_line(plan, r)})."
        notes += " Not verified: that the pinned library versions load this checkpoint, the output, the tolerances, speed, memory. No reference.json" + (", no bench record" if bench else "") + "."
        if plan == "vlm":
            desc = (f"{r['title']} (bf16): decode tokens per second, prefill images per second and peak GPU memory." if bench else
                    (f"{r['title']} reads text drawn at run time with greedy decoding." if r["spec"].get("image") == "text" else f"{r['title']} (bf16) describes a fixed public-domain image with greedy decoding."))
        else:
            desc = f"{r['title']}: throughput and peak GPU memory ({plan})." if bench else f"{r['title']} ({plan}) on a fixed input; the result is compared with a tolerance."
        out[name] = {"manifest": manifest(r, bench, name, {"description": desc, "runtime": runtime, "compare": py_compare(r), "model_id": model_id,
                                                           "revision": revision, "notes": notes}),
                     "run.sh": py_run(r, bench, name, f_name),
                     "model.sha256": sums_text(r, f"# {r['repo']} @ {r['repo_sha']} (LFS oids from the Hub API, read {mc.DATE}; never downloaded when this was written)")}
        if not bench:
            out[name]["main.py"] = py_main(r, name)
    return out


def support_line(plan, r):
    return {"vlm": "transformers 5.19.0 image-text-to-text types", "diffusers": "diffusers 0.41.0 pipelines"}.get(plan, "transformers 5.19.0 model types")


# ------------------------------------------------------------------ whisper.cpp

def whispercpp(r):
    f_name, b_name = names(r)
    f = r["files"][0]
    out = {}
    stem = r["key"].removeprefix("whisper-")
    for bench, name in ((False, f_name), (True, b_name)):
        notes = common_notes(r, bench) + (" Same as whisper-cpp-large-v3 with the large-v3-turbo model (whisper.cpp ggml format, converted by the whisper.cpp project; "
                                          "a pruned large-v3 with 4 decoder layers). Runtime: whisper.cpp tag v1.9.5 built by tools/build-whisper-cpp.sh; model and clip are "
                                          "fetched by tools/whisper-assets.sh and checked against pinned checksums. "
                                          + ("Task: whisper-bench (encoder on synthetic input plus decoder steps). " if bench else "Task: transcript of jfk.wav, lower-cased without punctuation, compared exactly and checked against the known text (word error rate <= 0.15). ")
                                          + "Not verified: that the ggml file loads in v1.9.5 and what it outputs. No reference.json" + (", no bench record" if bench else "") + ".")
        revision = (f"ggerganov/whisper.cpp HF commit {r['repo_sha']}; {f['path']} sha256 {f['sha256']} ({f['size']} bytes, the Hub's LFS oid); "
                    f"weights of {r['base']} @ {r['base_sha']}")
        desc = (f"whisper.cpp's own whisper-bench with {r['title']}." if bench else f"whisper.cpp transcribes an 11 s clip with {r['title']} (CUDA or HIP build).")
        out[name] = {"manifest": manifest(r, bench, name, {"description": desc, "runtime": "whisper.cpp", "compare": ["compare: exact"],
                                                           "model_id": f"{r['repo']} {f['path']}   # OpenAI Whisper large-v3-turbo converted to ggml by whisper.cpp",
                                                           "revision": revision, "notes": notes}),
                     "run.sh": whisper_run(r, bench, name, stem),
                     "model.sha256": sums_text(r, f"# {r['repo']} @ {r['repo_sha']} (LFS oid of the ggml file from the Hub API, read {mc.DATE}); tools/whisper-assets.sh model-{stem} checks the same value")}
    return out


def whisper_run(r, bench, name, stem):
    src = (ROOT / ("workloads/whisper-cpp-bench-large-v3/run.sh" if bench else "workloads/whisper-cpp-large-v3/run.sh")).read_text(encoding="utf-8")
    src = src.replace("model-large-v3", f"model-{stem}").replace("whisper-cpp-bench-large-v3", name).replace("whisper-cpp-large-v3", name)
    src = src.replace("with the large-v3 model", f"with the {stem} model").replace("with large-v3 on", f"with {stem} on")
    src = src.replace('detail "backend %s, large-v3"', f'detail "backend %s, {stem}"')
    hw = r["hw"]
    gate = ('. "$here/tools/hwcheck.sh"\n' f"pw_require_gpus {hw['gpus']} {hw['gpu_memory_gb']}\n")
    marker = 'command -v python3 >/dev/null || skip "python3 is not installed"\n'
    return src.replace(marker, marker + gate, 1).replace("#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n# WRITTEN, NEVER RUN (docs/model-registry.md).\n", 1)


# ------------------------------------------------------------------ driver

BUILDERS = {"llamacpp": llamacpp, "llamacpp-embed": llamacpp, "vllm": vllm, "whispercpp": whispercpp}


def build_all():
    """{relative path: (content, executable?)} of everything the catalog writes under workloads/."""
    files = {}
    resolved = mc.resolve_all()
    for r in resolved:
        if not r["workloads"]:
            continue
        build = BUILDERS.get(r["plan"], pytorch_plan)
        for name, parts in build(r).items():
            existing = ROOT / "workloads" / name / "manifest.yaml"
            if run_state(name) != "written":
                continue   # graduated: hand-maintained since its first run
            if existing.exists() and "WRITTEN, NEVER RUN" not in existing.read_text(encoding="utf-8"):
                sys.exit(f"{name}: a workload of that name exists and was not written by the catalog; rename the catalog key '{r['key']}'")
            for fn, content in parts.items():
                fn = "manifest.yaml" if fn == "manifest" else fn
                files[f"workloads/{name}/{fn}"] = (content, fn.endswith(".sh"))
    return resolved, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    import registry
    resolved, files = build_all()
    table = registry.tables(resolved)
    doc_path = ROOT / "docs" / "model-registry.md"
    doc = doc_path.read_text(encoding="utf-8") if doc_path.exists() else registry.skeleton()
    head, rest = doc.split(MARK_START, 1)
    tail = rest.split(MARK_END, 1)[1]
    new_doc = f"{head}{MARK_START}\n\n{table}\n{MARK_END}{tail}"
    stale = []
    for rel, (content, exe) in sorted(files.items()):
        p = ROOT / rel
        if not p.exists() or p.read_text(encoding="utf-8") != content:
            stale.append(rel)
            if not a.check:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
        if not a.check and exe:
            os.chmod(p, 0o755)
    if not doc_path.exists() or doc_path.read_text(encoding="utf-8") != new_doc:
        stale.append("docs/model-registry.md")
        if not a.check:
            doc_path.write_text(new_doc, encoding="utf-8")
    written = sum(1 for r in resolved if r["workloads"])
    print(f"{len(resolved)} catalog entries, {written} with workloads ({len(files)} files), {len(resolved) - written} blocked; {len(stale)} file(s) {'stale' if a.check else 'written'}")
    if a.check and stale:
        for s in stale[:20]:
            print("  stale:", s)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
