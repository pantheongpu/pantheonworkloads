# Shared by the llamacpp-synth-* workloads' run.sh (sourced after common.sh and lc_setup). Provides:
#   lc_synth_prepare                  makes sure pw-probe, llama-perplexity and llama-quantize exist in $LC_BIN
#                                     (building them with build.sh when missing) and that python has numpy and
#                                     the MIT-licensed `gguf` package; sets LC_PY. Exits 77 when it cannot.
#   lc_synth_functional <arch> <quants> <prompt> <models.sha256 file>
#                                     builds the random-weight models, runs them, prints the contract's JSON line
#   lc_synth_bench <arch>             builds one model (PW_SYNTH_SIZE, PW_SYNTH_QUANT) and runs llama-bench on it
# The models are random-weight models for kernel and graph coverage, not any named pretrained model:
# docs/llamacpp-synth.md. Knobs (all optional):
#   PW_SYNTH_SIZE     tiny (functional) | s | m (default for benchmarks) | l | xl: see `synth_gguf.py --list`
#   PW_SYNTH_QUANT    quantisation type for the benchmark models (default Q4_K_M)
#   PW_SYNTH_PIP      pip requirement for the gguf package when a venv is made (default gguf==0.19.0)
#   PW_SYNTH_SKIP_SHA 1: do not fail when a generated model's sha256 differs from models.sha256

LC_SYNTH_TOOLS="pw-probe llama-perplexity llama-quantize"

lc_synth_missing_tools() {
  local t
  for t in $LC_SYNTH_TOOLS; do [[ -x "$LC_BIN/$t" ]] || { echo "$t"; return 0; }; done
  return 1
}

lc_synth_prepare() {
  if lc_synth_missing_tools >/dev/null; then
    [[ -z "${PW_LLAMACPP_BIN_DIR:-}" ]] || lc_skip "PW_LLAMACPP_BIN_DIR=$LC_BIN lacks $(lc_synth_missing_tools) (build with LLAMACPP_EXTRA_TARGETS and LLAMACPP_BUILD_PROBE=1, see tools/llamacpp/build.sh)"
    [[ "${PW_LLAMACPP_NO_BUILD:-0}" != 1 ]] || lc_skip "no $(lc_synth_missing_tools) in $LC_BIN and PW_LLAMACPP_NO_BUILD=1"
    local prefix="$PW_CACHE/llama.cpp-$LLAMACPP_TAG-$LC_BACKEND" st
    rm -rf "$prefix.synth"
    LLAMACPP_EXTRA_TARGETS="llama-quantize llama-perplexity" LLAMACPP_BUILD_PROBE=1 \
      "$LC_ROOT/build.sh" "$LC_BACKEND" "$prefix.synth" >&2; st=$?
    if (( st == 77 )); then lc_skip "cannot build the synthetic-model tools for $LC_BACKEND here (see stderr)"; fi
    (( st == 0 )) || { echo "building the synthetic-model tools failed" >&2; exit 1; }
    # merge, never overwrite: another run may be using the existing files
    cp -an "$prefix.synth/bin/." "$LC_BIN/" && cp -an "$prefix.synth/lib/." "$LC_LIB/" && rm -rf "$prefix.synth"
    lc_synth_missing_tools >/dev/null && lc_skip "still no $(lc_synth_missing_tools) in $LC_BIN"
  fi
  LC_PY=python3
  if ! python3 -I -c 'import numpy, gguf' 2>/dev/null; then
    local venv="$PW_CACHE/venvs/synth"
    if [[ ! -x "$venv/bin/python" ]] || ! "$venv/bin/python" -I -c 'import numpy, gguf' 2>/dev/null; then
      python3 -m venv "$venv" >&2 && "$venv/bin/pip" install -q numpy "${PW_SYNTH_PIP:-gguf==0.19.0}" >&2 \
        || lc_skip "python has no numpy/gguf and a venv with them could not be made (pip install numpy gguf)"
    fi
    LC_PY="$venv/bin/python"
  fi
}

# lc_synth_functional <arch> <comma-separated quantisation types> <prompt> <models.sha256>
lc_synth_functional() {
  local arch="$1" quants="$2" prompt="$3" shas="$4" env_lines skip=()
  env_lines=$(printf '%s\n' ${LC_ENV[@]+"${LC_ENV[@]}"})
  [[ "${PW_SYNTH_SKIP_SHA:-0}" == 1 ]] && shas=""
  PW_LC_ENV="$env_lines" "$LC_PY" -I "$LC_ROOT/synth_run.py" --arch "$arch" --quants "$quants" --prompt "$prompt" \
    --bin "$LC_BIN" --cache "$PW_CACHE" --threads "${PW_THREADS:-2}" --ngl "$LC_NGL" --expected-shas "$shas" \
    >"$PW_OUT/synth.out" 2>"$PW_OUT/synth.err" \
    || { echo "synthetic run failed" >&2; tail -20 "$PW_OUT/synth.err" >&2; exit 1; }
  python3 -I - "$PW_OUT/synth.out" "llama.cpp $LLAMACPP_TAG $LC_BACKEND" <<'PY'
import json, sys
rec = json.loads(open(sys.argv[1], encoding="utf-8").read().strip().splitlines()[-1])
rec["detail"] = sys.argv[2] + ", " + rec["detail"]
print(json.dumps(rec))
PY
}

# lc_synth_bench <arch>: PW_SYNTH_SIZE (default m), PW_SYNTH_QUANT (default Q4_K_M)
lc_synth_bench() {
  local arch="$1" size="${PW_SYNTH_SIZE:-m}" quant="${PW_SYNTH_QUANT:-Q4_K_M}"
  LC_MODEL=$("$LC_PY" -I "$LC_ROOT/synth_run.py" --arch "$arch" --quants "$quant" --size "$size" \
      --bin "$LC_BIN" --cache "$PW_CACHE" --threads "${PW_THREADS:-4}" --prepare-only 2>"$PW_OUT/synth.err") \
    || { echo "could not build the synthetic $arch $size $quant model" >&2; tail -20 "$PW_OUT/synth.err" >&2; exit 1; }
  # shellcheck source=bench.sh
  . "$LC_ROOT/bench.sh"
  lc_bench
}
