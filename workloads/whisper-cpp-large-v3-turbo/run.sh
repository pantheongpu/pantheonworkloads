#!/usr/bin/env bash
# WRITTEN, NEVER RUN (docs/model-registry.md).
# whisper-cpp-large-v3-turbo: whisper.cpp transcribes an 11 s clip with the large-v3-turbo model; the
# normalised transcript is the output. Contract: docs/workload-contract.md.
#
#   PW_WHISPER_BACKEND  cuda | hip (default: cuda when
#                       nvidia-smi exists, hip when hipconfig exists)
#   PW_WHISPER_WER_MAX  fail when the word error rate against the known text exceeds this (default 0.15)
# Build and downloads: tools/build-whisper-cpp.sh, tools/whisper-assets.sh (both exit 77 when
# they cannot do their job). Greedy decoding, no temperature fallback.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
skip() { echo "SKIP: $*"; exit 77; }
command -v python3 >/dev/null || skip "python3 is not installed"
. "$here/tools/hwcheck.sh"
pw_require_gpus 1 10

backend="${PW_WHISPER_BACKEND:-}"
case "$PW_TARGET" in
  gpu)
    if [[ -z "$backend" ]]; then
      if command -v nvidia-smi >/dev/null; then backend=cuda
      elif command -v hipconfig >/dev/null; then backend=hip
      else skip "no GPU runtime found (nvidia-smi / hipconfig)"; fi
    fi
    gpu_flag=() ;;
  *) skip "target $PW_TARGET is not supported by this workload" ;;
esac

# need VAR cmd...: run a helper, pass its SKIP (exit 77) through, keep its last stdout line in VAR.
need() { local v=$1 out rc; shift; out=$("$@"); rc=$?; [[ $rc -eq 0 ]] || { echo "$out"; exit "$rc"; }; printf -v "$v" %s "$(tail -1 <<<"$out")"; }
need bin   "$here/tools/build-whisper-cpp.sh" "$backend"
need model "$here/tools/whisper-assets.sh" model-large-v3-turbo
need clip  "$here/tools/whisper-assets.sh" clip

text=$("$bin/whisper-cli" -m "$model" -f "$clip" -l en -t 4 -bs 1 -bo 1 -nf -nt -np "${gpu_flag[@]}" 2>"$PW_OUT/whisper-cli.err") \
  || { tail -5 "$PW_OUT/whisper-cli.err" >&2; echo "whisper-cli failed" >&2; exit 1; }

PW_WHISPER_CPP_TAG=$(sed -n "s/^TAG=//p" "$here/tools/build-whisper-cpp.sh") PW_TEXT="$text" PW_BACKEND="$backend" python3 -I "$here/tools/whisper_transcript.py"
