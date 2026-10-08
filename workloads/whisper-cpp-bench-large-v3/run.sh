#!/usr/bin/env bash
# whisper-cpp-bench-large-v3: whisper.cpp's own `whisper-bench` (encoder on synthetic input plus
# decoder steps) with large-v3 on a real GPU. Contract: docs/workload-contract.md.
#
#   PW_WHISPER_BACKEND  cuda | hip (default: cuda when nvidia-smi exists, hip when hipconfig exists)
# CPU runs are refused: this workload exists to record GPU numbers.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
skip() { echo "SKIP: $*"; exit 77; }
command -v python3 >/dev/null || skip "python3 is not installed"
[[ "$PW_TARGET" == gpu ]] || skip "target $PW_TARGET is not supported: real GPU only"

backend="${PW_WHISPER_BACKEND:-}"
if [[ -z "$backend" ]]; then
  if command -v nvidia-smi >/dev/null; then backend=cuda
  elif command -v hipconfig >/dev/null; then backend=hip
  else skip "no GPU runtime found (nvidia-smi / hipconfig)"; fi
fi
[[ "$backend" == cuda || "$backend" == hip ]] || skip "backend $backend is not a GPU backend"

need() { local v=$1 out rc; shift; out=$("$@"); rc=$?; [[ $rc -eq 0 ]] || { echo "$out"; exit "$rc"; }; printf -v "$v" %s "$(tail -1 <<<"$out")"; }
need bin   "$here/tools/build-whisper-cpp.sh" "$backend"
need model "$here/tools/whisper-assets.sh" model-large-v3

"$bin/whisper-bench" -m "$model" -t 4 >"$PW_OUT/whisper-bench.log" 2>&1 \
  || { tail -5 "$PW_OUT/whisper-bench.log" >&2; echo "whisper-bench failed" >&2; exit 1; }
# A build without the GPU backend would silently benchmark the CPU: refuse that.
grep -qiE "using (CUDA|ROCm|HIP)|ggml_cuda_init: found [1-9]|ggml_cuda_init.*devices" "$PW_OUT/whisper-bench.log" \
  || { echo "whisper-bench did not report a GPU backend; see $PW_OUT/whisper-bench.log" >&2; exit 1; }
metrics=$(python3 -I "$here/tools/whisper_bench_parse.py" <"$PW_OUT/whisper-bench.log") || exit 1
printf '{"output": "whisper-bench completed", "detail": "backend %s, large-v3", "metrics": %s}\n' "$backend" "$metrics"
