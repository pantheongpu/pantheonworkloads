#!/usr/bin/env bash
# vllm-pixtral-large-instruct-2411: vLLM greedy decode of 8 tokens with Pixtral-Large-Instruct-2411, tensor parallel 4.
# WRITTEN, NEVER RUN (docs/model-registry.md). Contract: docs/workload-contract.md. Exits 77, with the reason, when the host lacks the GPUs,
# the disk, torch or vLLM, or cannot fetch the model. The pinned snapshot is downloaded and sha256-verified by workloads/_shared/vllm_catalog.py
# before vLLM starts, so vLLM loads from the verified local copy.
#   PW_PYTHON  Python with vLLM (default python3)    PW_CACHE  model cache (default ~/.cache/pantheonworkloads)    PW_IGNORE_REQUIRES=1
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
[[ "$PW_TARGET" == gpu ]] || pw_skip "target $PW_TARGET: Pixtral-Large-Instruct-2411 needs real GPUs (4 x 80 GiB)"
source "$PW_WORKLOAD_DIR/../../tools/hwcheck.sh"
pw_require_gpus 4 80
pw_require_disk 232
pw_setup vllm
"$PW_PY" -c 'import vllm' 2>/dev/null || pw_skip "$PW_PY cannot import vllm (pip install vllm==0.30.0)"
export PW_VLLM_MODEL="mistralai/Pixtral-Large-Instruct-2411" PW_VLLM_REVISION="c1e51f6f11974a1199685d35c62f5a425c2d001e" PW_VLLM_TP=4 PW_VLLM_MAX_LEN=4096 PW_VLLM_MISTRAL=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # the SmolLM2 workload needed this on its first run (ninja was not on PATH)
result="$PW_OUT/vllm-catalog-result.json"
rm -f "$result"
pw_python "$PW_WORKLOAD_DIR/../_shared/vllm_catalog.py" "$result" >"$PW_OUT/vllm-catalog.log" 2>&1
status=$?
if [[ $status == 77 ]]; then
  grep '^SKIP' "$PW_OUT/vllm-catalog.log" | tail -1 || echo "SKIP: vLLM cannot run here"
  exit 77
fi
if [[ $status != 0 || ! -s "$result" ]]; then
  tail -15 "$PW_OUT/vllm-catalog.log" >&2
  echo "vLLM failed (exit $status)" >&2
  exit 1
fi
cat "$result"; echo
