#!/usr/bin/env bash
# smollm2-135m-ollama: Ollama generates three greedy tokens after a fixed
# prompt. Contract: docs/workload-contract.md. Needs `ollama`, python3, curl.
#
#   PW_MODEL       model to run (default smollm2:135m)
#   PW_PULL=1      pull the model when it is not already in $OLLAMA_MODELS
#   PANTHEONSIM_DIR  built pantheonsim checkout, for sim:nvidia/* targets
#
# Not run yet: written from pantheonsim's tests/e2e/run_ollama.sh, which it mirrors.
set -uo pipefail
model="${PW_MODEL:-smollm2:135m}"
skip() { echo "SKIP: $*"; exit 77; }
command -v ollama  >/dev/null || skip "ollama is not installed"
command -v curl    >/dev/null || skip "curl is not installed"
command -v python3 >/dev/null || skip "python3 is not installed"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

env_extra=()
case "$PW_TARGET" in
  cpu) env_extra=(CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1) ;;
  gpu) ;;
  sim:nvidia/*)
    [[ -n "${PANTHEONSIM_DIR:-}" ]] || skip "set PANTHEONSIM_DIR to a built pantheonsim checkout"
    shim="$PANTHEONSIM_DIR/build/shim"
    major=$(ls "$shim"/libcudart.so.[0-9]* 2>/dev/null | head -1 | sed 's/.*\.so\.//')
    [[ -n "$major" ]] || skip "no libcudart in $shim (pantheonsim built without CUDA headers)"
    # Ollama bundles its own libcudart and libcublas and puts its directory first, so the
    # simulator's are preloaded: a library already loaded under a soname satisfies later needs.
    preload="$shim/libcuda.so.1:$shim/libcudart.so.$major:$shim/libcublas.so.$major:$shim/libcublasLt.so.$major"
    env_extra=(LD_PRELOAD="$preload" LD_LIBRARY_PATH="$shim" VGPU_GPU="$PW_SIM_PROFILE" VGPU_QUIET=1
               VGPU_TELEMETRY_PATH="$PW_OUT/vgpu-run" VGPU_STATE_DIR="$PW_OUT/vgpu-state") ;;
  *) skip "target $PW_TARGET is not supported by this workload" ;;
esac

port=$((20000 + RANDOM % 20000))
export OLLAMA_HOST="127.0.0.1:$port"
log="$PW_OUT/ollama-serve.log"
env "${env_extra[@]}" ollama serve >"$log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null' EXIT
for _ in $(seq 150); do curl -s -m 2 "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 0.2; done
curl -s -m 2 "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "ollama serve did not come up" >&2; tail -20 "$log" >&2; exit 1; }

if ! ollama list 2>/dev/null | grep -q "^$model "; then
  [[ "${PW_PULL:-0}" == 1 ]] || skip "$model is not in $OLLAMA_MODELS (PW_PULL=1 pulls it)"
  ollama pull "$model" >/dev/null || { echo "could not pull $model" >&2; exit 1; }
fi

reply=$(curl -s -m 1700 "http://$OLLAMA_HOST/api/generate" \
  -d "{\"model\":\"$model\",\"prompt\":\"The capital of France is\",\"stream\":false,\"options\":{\"num_predict\":3,\"temperature\":0,\"seed\":1}}")
PW_REPLY="$reply" PW_SIM="$([[ $PW_TARGET == sim:* ]] && echo 1)" python3 - <<'PY'
import json, os, sys
d = json.loads(os.environ["PW_REPLY"])
if "response" not in d:
    sys.exit("ollama error: " + str(d.get("error", d)))
metrics = {}
# Timings only mean something on a real target.
if not os.environ.get("PW_SIM") and d.get("eval_duration"):
    metrics = {"decode_tokens_per_s": round(d["eval_count"] / (d["eval_duration"] / 1e9), 2)}
print(json.dumps({"output": d["response"], "detail": "model " + os.environ.get("PW_MODEL", "smollm2:135m"), "metrics": metrics}))
PY
