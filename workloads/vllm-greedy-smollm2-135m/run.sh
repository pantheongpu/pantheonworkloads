#!/usr/bin/env bash
# vllm-greedy-smollm2-135m: vLLM (pinned v0.30.0) greedy-decodes 8 tokens after a fixed prompt.
# Contract: docs/workload-contract.md. Mirrors pantheonsim's amd/tests/e2e/run_vllm_amd.sh.
#
# Needs a real GPU stack (CUDA or ROCm) with vLLM installed, or, for sim:amd/*, pantheonsim built
# and a ROCm vLLM (amd/tests/vllm/install.sh there makes one). It exits 77, with the reason, when
# torch, vLLM, a GPU or the simulator is missing; it never runs on the CPU target.
#
#   PW_PYTHON            Python with vLLM, for the gpu target (default python3)
#   VGPU_VLLM_PYTHON     the same, for sim:amd/* targets
#   PW_MODEL, PW_MODEL_REVISION, PW_PROMPT, PW_GPU_MEM_UTIL
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
case "$PW_TARGET" in
  gpu) ;;
  sim:amd/rx6900xt) pw_skip "vLLM ships no gfx1030 kernels (pantheonsim's vLLM test leaves the RX 6900 XT out for the same reason)" ;;
  sim:amd/*) ;;
  *) pw_skip "target $PW_TARGET: vLLM needs a CUDA or ROCm GPU (real, or simulated AMD)" ;;
esac
pw_setup vllm
if [[ "$PW_TARGET" != sim:* ]]; then
  "$PW_PY" -c 'import vllm' 2>/dev/null || pw_skip "$PW_PY cannot import vllm (pip install vllm==0.30.0)"
fi
result="$PW_OUT/vllm-greedy-result.json"
rm -f "$result"
pw_python "$PW_WORKLOAD_DIR/generate.py" "$result" >"$PW_OUT/vllm-greedy.log" 2>&1
status=$?
if [[ $status == 77 ]]; then
  grep '^SKIP' "$PW_OUT/vllm-greedy.log" | tail -1 || echo "SKIP: vLLM cannot run here"
  exit 77
fi
if [[ $status != 0 || ! -s "$result" ]]; then
  tail -15 "$PW_OUT/vllm-greedy.log" >&2
  echo "vLLM failed (exit $status)" >&2
  exit 1
fi
cat "$result"; echo
