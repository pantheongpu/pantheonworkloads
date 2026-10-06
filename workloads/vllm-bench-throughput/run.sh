#!/usr/bin/env bash
# vllm-bench-throughput: vLLM's own `vllm bench throughput` (offline engine, v0.30.0) on random
# prompts of fixed length. Contract: docs/workload-contract.md. Real GPUs only.
#
# Exits 77, with the reason, when torch, vLLM or a GPU is missing. Not run anywhere yet.
#
#   PW_PYTHON  Python with vLLM (default python3)
#   PW_MODEL, PW_MODEL_REVISION
#   PW_BENCH_PROMPTS (256), PW_BENCH_INPUT_LEN (128), PW_BENCH_OUTPUT_LEN (128), PW_BENCH_WARMUPS (16)
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
[[ "$PW_TARGET" == gpu ]] || pw_skip "target $PW_TARGET: benchmark numbers need a real GPU"
pw_setup vllm
"$PW_PY" -c 'import vllm' 2>/dev/null || pw_skip "$PW_PY cannot import vllm (pip install vllm==0.30.0)"
"$PW_PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null ||
  pw_skip "torch sees no CUDA/ROCm GPU"

model="${PW_MODEL:-HuggingFaceTB/SmolLM2-135M}"
prompts="${PW_BENCH_PROMPTS:-256}" ilen="${PW_BENCH_INPUT_LEN:-128}" olen="${PW_BENCH_OUTPUT_LEN:-128}"
json="$PW_OUT/vllm-throughput.json"
rm -f "$json"
args=(bench throughput --model "$model" --dtype float16 --seed 0 --backend vllm
      --dataset-name random --random-input-len "$ilen" --random-output-len "$olen"
      --num-prompts "$prompts" --num-warmups "${PW_BENCH_WARMUPS:-16}" --output-json "$json")
[[ -z "${PW_MODEL_REVISION:-}" ]] || args+=(--revision "$PW_MODEL_REVISION")
pw_python -m vllm.entrypoints.cli.main "${args[@]}" >"$PW_OUT/vllm-bench.log" 2>&1
status=$?
if [[ $status != 0 || ! -s "$json" ]]; then
  tail -15 "$PW_OUT/vllm-bench.log" >&2
  echo "vllm bench throughput failed (exit $status)" >&2
  exit 1
fi
# The tool's JSON has requests_per_second and tokens_per_second (prompt + output tokens); the
# output-only rate is on its "Throughput:" line.
PW_JSON="$json" PW_LOG="$PW_OUT/vllm-bench.log" PW_MODEL_USED="$model" PW_N="$prompts" PW_I="$ilen" PW_O="$olen" \
  "$PW_PY" - <<'PY'
import json, os, re, sys
r = json.load(open(os.environ["PW_JSON"]))
log = open(os.environ["PW_LOG"], errors="replace").read()
m = re.findall(r"Throughput: ([\d.]+) requests/s, ([\d.]+) total tokens/s, ([\d.]+) output tokens/s", log)
metrics = {"requests_per_s": round(r["requests_per_second"], 3),
           "total_tokens_per_s": round(r["tokens_per_second"], 1)}
if m:
    metrics["output_tokens_per_s"] = float(m[-1][2])
n = os.environ["PW_N"]
print(json.dumps({"output": f"{r['num_requests']} requests, {n} prompts of {os.environ['PW_I']} in / {os.environ['PW_O']} out tokens",
                  "detail": f"{os.environ['PW_MODEL_USED']}, elapsed {r['elapsed_time']:.2f} s, {r['total_num_tokens']} tokens",
                  "metrics": metrics}))
PY
