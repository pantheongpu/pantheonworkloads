#!/usr/bin/env bash
# vllm-bench-deepseek-v4p1-flash: `vllm bench throughput` with DeepSeek-V4.1-Flash, tensor parallel 8.
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md. Exits 77, with the reason, when the host lacks the GPUs,
# the disk, torch or vLLM, or cannot fetch the model. The pinned snapshot is downloaded and sha256-verified by workloads/_shared/vllm_catalog.py
# before vLLM starts, so vLLM loads from the verified local copy.
#   PW_PYTHON  Python with vLLM (default python3)    PW_CACHE  model cache (default ~/.cache/pantheonworkloads)    PW_IGNORE_REQUIRES=1
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
[[ "$PW_TARGET" == gpu ]] || pw_skip "target $PW_TARGET: DeepSeek-V4.1-Flash needs real GPUs (8 x 80 GiB)"
source "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"
pw_require_gpus 8 80
pw_require_disk 476
pw_setup vllm
"$PW_PY" -c 'import vllm' 2>/dev/null || pw_skip "$PW_PY cannot import vllm (pip install vllm==0.30.0)"
export PW_VLLM_MODEL="deepseek-ai/DeepSeek-V4.1-Flash" PW_VLLM_REVISION="2cba9e42aa026125f3ed06c6d98c1db82f7ca027" PW_VLLM_TP=8 PW_VLLM_MAX_LEN=4096
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # the SmolLM2 workload needed this on its first run (ninja was not on PATH)
[[ "$PW_TARGET" == gpu ]] || pw_skip "benchmark numbers need a real GPU"
pathfile="$PW_OUT/snapshot-path.txt"
pw_python "$PW_WORKLOAD_DIR/../_shared/vllm_catalog.py" --prefetch "$pathfile" >"$PW_OUT/vllm-prefetch.log" 2>&1
status=$?
if [[ $status == 77 ]]; then grep '^SKIP' "$PW_OUT/vllm-prefetch.log" | tail -1 || echo "SKIP: cannot fetch the model"; exit 77; fi
if [[ $status != 0 || ! -s "$pathfile" ]]; then tail -15 "$PW_OUT/vllm-prefetch.log" >&2; echo "snapshot failed (exit $status)" >&2; exit 1; fi
snap=$(cat "$pathfile")
json="$PW_OUT/vllm-throughput.json"
rm -f "$json"
pw_python -m vllm.entrypoints.cli.main bench throughput --model "$snap" --tensor-parallel-size 8 --max-model-len 4096 --dtype auto --seed 0 --backend vllm \
  --dataset-name random --random-input-len 128 --random-output-len 128 --num-prompts 64 --num-warmups 8 --output-json "$json" >"$PW_OUT/vllm-bench.log" 2>&1
status=$?
if [[ $status != 0 || ! -s "$json" ]]; then tail -15 "$PW_OUT/vllm-bench.log" >&2; echo "vllm bench throughput failed (exit $status)" >&2; exit 1; fi
PW_JSON="$json" PW_LOG="$PW_OUT/vllm-bench.log" PW_MODEL_USED="$PW_VLLM_MODEL" "$PW_PY" - <<'PY'
import json, os, re
r = json.load(open(os.environ["PW_JSON"]))
log = open(os.environ["PW_LOG"], errors="replace").read()
m = re.findall(r"Throughput: ([\d.]+) requests/s, ([\d.]+) total tokens/s, ([\d.]+) output tokens/s", log)
metrics = {"requests_per_s": round(r["requests_per_second"], 3), "total_tokens_per_s": round(r["tokens_per_second"], 1)}
if m:
    metrics["output_tokens_per_s"] = float(m[-1][2])
print(json.dumps({"output": f"{r['num_requests']} requests, 64 prompts of 128 in / 128 out tokens", "detail": f"{os.environ['PW_MODEL_USED']}, elapsed {r['elapsed_time']:.2f} s", "metrics": metrics}))
PY
