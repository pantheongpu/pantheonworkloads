"""The Python half of the model-catalog vLLM workloads (vllm-<model> and vllm-bench-<model>).

    vllm_catalog.py <result.json>            functional: greedy-decode 8 tokens after a fixed prompt, write the contract's JSON
    vllm_catalog.py --prefetch <path.txt>    bench: download and verify the pinned snapshot, write its local path to path.txt
                                             (run.sh then calls `vllm bench throughput` on that path)

Configuration comes from the environment, set by run.sh from the workload's manifest:
  PW_VLLM_MODEL, PW_VLLM_REVISION   Hub repository and the commit it is pinned to
  PW_VLLM_TP                        tensor-parallel size (= requires.gpus)
  PW_VLLM_MISTRAL=1                 the repository is in Mistral's own format (params.json, consolidated.safetensors)
  PW_VLLM_MAX_LEN                   max_model_len (default 4096)
The snapshot is fetched with huggingface_hub at the pinned commit (only the files model.sha256 lists, plus configs and
tokenizers), every listed sha256 is checked (a mismatch fails the run), and vLLM is pointed at the local directory,
so what runs is what was pinned. Engine settings follow the SmolLM2 workload (eager mode, seed 0, temperature 0) but
with a 4096-token context and gpu_memory_utilization 0.90 for the memory arithmetic in docs/model-registry.md.
Exits 77 with a reason when torch, vLLM or a GPU is missing. Never run: written against vLLM v0.30.0's source.
"""
import hashlib
import json
import os
import sys


def skip(why):
    print(f"SKIP: {why}")
    sys.exit(77)


def read_sums(path):
    sums = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                s, n = line.split(None, 1)
                sums[n.strip()] = s
    return sums


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        skip(f"huggingface_hub is not installed ({e})")
    model, revision = os.environ["PW_VLLM_MODEL"], os.environ["PW_VLLM_REVISION"]
    sums_path = os.path.join(os.environ["PW_WORKLOAD_DIR"], "model.sha256")
    sums = read_sums(sums_path)
    small = ["*.json", "*.txt", "*.model", "*.jinja", "*.tiktoken", "tokenizer*", "tekken.json"]
    cache = os.path.join(os.environ.get("PW_CACHE", os.path.expanduser("~/.cache/pantheonworkloads")), "hf")
    try:
        path = snapshot_download(model, revision=revision, cache_dir=cache, allow_patterns=[*small, *sums])
    except Exception as e:   # noqa: BLE001  no network, gated, no such revision: not a failure of the GPU
        skip(f"cannot fetch {model}@{revision} ({type(e).__name__}: {str(e).splitlines()[0][:150]})")
    for name, want in sums.items():
        got = sha256_file(os.path.join(path, name))
        if got != want:
            sys.exit(f"{name} has sha256 {got}, the pin is {want}")
    return path


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--prefetch":
        path = snapshot()
        with open(sys.argv[2], "w") as f:
            f.write(path + "\n")
        return
    out_file = sys.argv[1]
    try:
        import torch
    except ImportError as e:
        skip(f"torch is not installed ({e})")
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as e:
        skip(f"vLLM is not installed ({e})")
    tp = int(os.environ.get("PW_VLLM_TP", "1"))
    if torch.cuda.device_count() < tp:
        skip(f"torch sees {torch.cuda.device_count()} GPU(s), the workload needs {tp}")
    path = snapshot()
    kwargs = {}
    if os.environ.get("PW_VLLM_MISTRAL") == "1":
        kwargs.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
    llm = LLM(model=path, tensor_parallel_size=tp, enforce_eager=True, max_model_len=int(os.environ.get("PW_VLLM_MAX_LEN", "4096")),
              gpu_memory_utilization=0.90, seed=0, **kwargs)
    prompt = os.environ.get("PW_PROMPT", "The capital of France is")
    res = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0))[0].outputs[0]
    with open(out_file, "w") as f:
        json.dump({"output": res.text,
                   "detail": f"{os.environ['PW_VLLM_MODEL']}@{os.environ['PW_VLLM_REVISION']}, tp {tp}, vllm {vllm.__version__}, "
                             f"torch {torch.__version__}, token ids {list(res.token_ids)}",
                   "metrics": {}}, f)


if __name__ == "__main__":
    main()
