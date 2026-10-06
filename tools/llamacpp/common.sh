# Shared by the llamacpp-* workloads' run.sh (sourced, not executed). Provides:
#   lc_skip <why>            print SKIP: <why> and exit 77
#   lc_setup                 resolves PW_TARGET to a backend, finds or builds the binaries,
#                            sets LC_BIN, LC_NGL and the environment the simulated targets need
#   lc_fetch_model <url> <sha256|""> <file>   find or download the model, sets LC_MODEL
# Knobs (all optional):
#   PW_LLAMACPP_BACKEND   cuda | hip | cpu: which build a `gpu` target uses (default: detected)
#   PW_LLAMACPP_BIN_DIR   a directory holding llama-completion and llama-bench you built yourself
#   PW_LLAMACPP_NO_BUILD  1: never build, skip when there is no binary
#   PW_CACHE              where builds and models are kept (default ~/.cache/pantheonworkloads)
#   PW_MODEL_FILE         use this local GGUF instead of downloading
#   PW_NGL                layers offloaded (default 99 on gpu targets, 0 on cpu)
#   PW_THREADS            CPU threads (default 2)
# The simulated targets are wired from pantheonsim's tests/e2e/run_ollama.sh and
# amd/tests/e2e/run_ollama_amd.sh and have NOT been run (no GPU toolchain was available).

LC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pin.env
. "$LC_ROOT/pin.env"
PW_CACHE="${PW_CACHE:-$HOME/.cache/pantheonworkloads}"

lc_skip() { echo "SKIP: $*"; exit 77; }

lc_backend() {
  case "$PW_TARGET" in
    cpu) echo cpu ;;
    sim:nvidia/*) echo cuda ;;
    sim:amd/*) echo hip ;;
    gpu)
      if [[ -n "${PW_LLAMACPP_BACKEND:-}" ]]; then echo "$PW_LLAMACPP_BACKEND"
      elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then echo cuda
      elif [[ -e /dev/kfd ]] || command -v rocminfo >/dev/null; then echo hip
      else lc_skip "no NVIDIA or AMD GPU found for target gpu (set PW_LLAMACPP_BACKEND to override)"; fi ;;
    *) lc_skip "target $PW_TARGET is not supported by this workload" ;;
  esac
}

lc_setup() {
  command -v python3 >/dev/null || lc_skip "python3 is not installed"
  LC_BACKEND=$(lc_backend)
  if [[ -n "${PW_LLAMACPP_BIN_DIR:-}" ]]; then
    LC_BIN="$PW_LLAMACPP_BIN_DIR"
    LC_LIB="${PW_LLAMACPP_LIB_DIR:-$(dirname "$LC_BIN")/lib}"
  else
    local prefix="$PW_CACHE/llama.cpp-$LLAMACPP_TAG-$LC_BACKEND"
    LC_BIN="$prefix/bin"; LC_LIB="$prefix/lib"
    if [[ ! -x "$LC_BIN/llama-bench" || ! -x "$LC_BIN/llama-completion" ]]; then
      [[ "${PW_LLAMACPP_NO_BUILD:-0}" != 1 ]] || lc_skip "no llama.cpp $LC_BACKEND build in $prefix and PW_LLAMACPP_NO_BUILD=1"
      mkdir -p "$PW_CACHE"
      "$LC_ROOT/build.sh" "$LC_BACKEND" "$prefix.part" >&2; local st=$?
      if (( st == 77 )); then lc_skip "cannot build llama.cpp for $LC_BACKEND here (see stderr)"; fi
      (( st == 0 )) || { echo "llama.cpp build failed" >&2; exit 1; }
      rm -rf "$prefix"; mv "$prefix.part" "$prefix"
    fi
  fi
  [[ -x "$LC_BIN/llama-completion" && -x "$LC_BIN/llama-bench" ]] || lc_skip "no llama-completion/llama-bench in $LC_BIN"
  export LD_LIBRARY_PATH="$LC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  LC_ENV=()
  LC_NGL="${PW_NGL:-99}"
  case "$PW_TARGET" in
    cpu) LC_NGL="${PW_NGL:-0}"; LC_ENV=(CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1) ;;
    gpu) ;;
    sim:nvidia/*)
      [[ -n "${PANTHEONSIM_DIR:-}" ]] || lc_skip "set PANTHEONSIM_DIR to a built pantheonsim checkout"
      local shim="$PANTHEONSIM_DIR/build/shim" major
      major=$(ls "$shim"/libcudart.so.[0-9]* 2>/dev/null | head -1 | sed 's/.*\.so\.//')
      [[ -n "$major" ]] || lc_skip "no libcudart in $shim (pantheonsim built without CUDA headers)"
      # As in pantheonsim tests/e2e/run_ollama.sh: a library already loaded under a soname
      # satisfies the binary's later need for it, so the simulator's go first.
      LC_ENV=(LD_PRELOAD="$shim/libcuda.so.1:$shim/libcudart.so.$major:$shim/libcublas.so.$major:$shim/libcublasLt.so.$major"
              LD_LIBRARY_PATH="$shim:$LD_LIBRARY_PATH" VGPU_GPU="$PW_SIM_PROFILE" VGPU_QUIET=1
              VGPU_TELEMETRY_PATH="$PW_OUT/vgpu-run" VGPU_STATE_DIR="$PW_OUT/vgpu-state") ;;
    sim:amd/*)
      [[ -n "${PANTHEONSIM_DIR:-}" ]] || lc_skip "set PANTHEONSIM_DIR to a built pantheonsim checkout"
      local shim="$PANTHEONSIM_DIR/build/shim"
      [[ -e "$shim/libamdhip64.so.7" ]] || lc_skip "no HIP shim in $shim"
      # As in pantheonsim amd/tests/e2e/run_ollama_amd.sh: copies with the links resolved, preloaded;
      # the one simulated GPU is HIP device 0; rocBLAS stays off hipBLASLt (no kernel library for gfx12 there).
      mkdir -p "$PW_OUT/shim"
      cp -L "$shim"/libamdhip64.so.7 "$shim"/libhsa-runtime64.so.1 "$PW_OUT/shim/"
      LC_ENV=(LD_PRELOAD="$PW_OUT/shim/libamdhip64.so.7:$PW_OUT/shim/libhsa-runtime64.so.1" VGPU_GPU="$PW_SIM_PROFILE"
              CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=0 ROCBLAS_USE_HIPBLASLT=0
              VGPU_TELEMETRY_PATH="$PW_OUT/vgpu-run" VGPU_STATE_DIR="$PW_OUT/vgpu-state") ;;
  esac
}

# lc_fetch_model <url> <sha256 or ""> <file name>
lc_fetch_model() {
  local url="$1" sha="$2" name="$3"
  if [[ -n "${PW_MODEL_FILE:-}" ]]; then
    [[ -f "$PW_MODEL_FILE" ]] || lc_skip "PW_MODEL_FILE=$PW_MODEL_FILE does not exist"
    LC_MODEL="$PW_MODEL_FILE"; sha="${PW_MODEL_SHA256:-}"
  else
    [[ -n "$url" ]] || lc_skip "no model: set PW_MODEL_URL (a direct .gguf link) or PW_MODEL_FILE"
    LC_MODEL="$PW_CACHE/models/$name"
    if [[ ! -f "$LC_MODEL" ]]; then
      command -v curl >/dev/null || lc_skip "curl is not installed"
      mkdir -p "$PW_CACHE/models"
      curl -fsSL --retry 2 -m "${PW_DOWNLOAD_TIMEOUT_S:-3000}" -o "$LC_MODEL.part" "$url" \
        || { rm -f "$LC_MODEL.part"; lc_skip "cannot download $url"; }
      mv "$LC_MODEL.part" "$LC_MODEL"
    fi
  fi
  if [[ -n "$sha" ]]; then
    local got; got=$(sha256sum "$LC_MODEL" | cut -d' ' -f1)
    [[ "$got" == "$sha" ]] || { echo "sha256 of $LC_MODEL is $got, expected $sha" >&2; exit 1; }
  else
    echo "warning: model sha256 is not pinned; this result is not reproducible. sha256 is $(sha256sum "$LC_MODEL" | cut -d' ' -f1)" >&2
  fi
}
