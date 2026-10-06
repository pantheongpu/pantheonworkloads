# Sourced by the PyTorch/vLLM workloads' run.sh. Gives them `pw_python`, a function that runs
# a Python script on the workload's target, and `pw_skip`.
#
#   source "$PW_WORKLOAD_DIR/../_shared/torch_env.sh"
#   pw_setup torch|vllm            # picks the interpreter and sets the target up; may exit 77
#   pw_python script.py args...    # runs it, stdout and stderr as they come
#
# cpu                  PW_PYTHON, or python3. GPUs are hidden (CUDA_VISIBLE_DEVICES=-1).
# gpu                  PW_PYTHON, or python3: whatever GPU torch finds.
# sim:nvidia/<gpu>     pantheonsim's `vgpu run --preload` with a Python that has PyTorch for CUDA 13
#                      ($VGPU_TORCH_CUDA_PYTHON or ~/.local/share/torch-cu13*), as
#                      pantheonsim's nvidia/tests/e2e/run_pytorch.sh does it.
# sim:amd/<gpu>        PyTorch (or vLLM) for ROCm ($VGPU_TORCH_PYTHON or ~/.local/share/torch-rocm*;
#                      $VGPU_VLLM_PYTHON for vllm) in a tree of links to its package whose
#                      libamdhip64 and librocm_smi64 are pantheonsim's, as amd/tests/e2e/run_pytorch.sh
#                      and run_vllm_amd.sh do it.
# Needs PANTHEONSIM_DIR (a built checkout) for sim: targets.

pw_skip() { echo "SKIP: $*"; exit 77; }

pw_setup() {
  local flavour=$1 python="" c
  PW_ENV=()
  PW_PREFIX=()
  case "$PW_TARGET" in
    cpu)
      python="${PW_PYTHON:-python3}"
      PW_ENV=(CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1) ;;
    gpu)
      python="${PW_PYTHON:-python3}" ;;
    sim:nvidia/*)
      [[ "$flavour" == torch ]] || pw_skip "pantheonsim runs vLLM on simulated AMD GPUs only (amd/tests/e2e/run_vllm_amd.sh)"
      [[ -n "${PANTHEONSIM_DIR:-}" ]] || pw_skip "set PANTHEONSIM_DIR to a built pantheonsim checkout"
      local build="$PANTHEONSIM_DIR/build"
      [[ -e "$build/shim/libcudart.so.13" ]] || pw_skip "no CUDA 13 runtime shim in $build/shim"
      [[ -x "$build/vgpu" ]] || pw_skip "$build/vgpu is not built"
      for c in "${VGPU_TORCH_CUDA_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-cu13*/bin/python 2>/dev/null); do
        [[ -n "$c" && -x "$c" ]] && "$c" -c 'import torch, sys; sys.exit(0 if (torch.version.cuda or "").startswith("13") else 1)' 2>/dev/null &&
          { python=$c; break; }
      done
      [[ -n "$python" ]] || pw_skip "no Python with PyTorch for CUDA 13 (set VGPU_TORCH_CUDA_PYTHON)"
      PW_ENV=(TRITON_CACHE_DIR="$PW_OUT/triton" TORCHINDUCTOR_CACHE_DIR="$PW_OUT/inductor")
      PW_PREFIX=("$build/vgpu" run --gpu "$PW_SIM_PROFILE" --preload) ;;
    sim:amd/*)
      [[ -n "${PANTHEONSIM_DIR:-}" ]] || pw_skip "set PANTHEONSIM_DIR to a built pantheonsim checkout"
      local build="$PANTHEONSIM_DIR/build"
      [[ -e "$build/shim/libamdhip64.so.7" ]] || pw_skip "no HIP shim in $build/shim"
      if [[ "$flavour" == vllm ]]; then
        python="${VGPU_VLLM_PYTHON:-}"
        [[ -n "$python" && -x "$python" ]] || pw_skip "no Python with vLLM for ROCm (set VGPU_VLLM_PYTHON; pantheonsim's amd/tests/vllm/install.sh makes one)"
      else
        for c in "${VGPU_TORCH_PYTHON:-}" $(ls -d "$HOME"/.local/share/torch-rocm*/bin/python 2>/dev/null); do
          [[ -n "$c" && -x "$c" ]] && "$c" -c 'import torch, sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null &&
            { python=$c; break; }
        done
        [[ -n "$python" ]] || pw_skip "no Python with PyTorch for ROCm (set VGPU_TORCH_PYTHON)"
      fi
      # The package's directory, found without importing it (its libraries load only with the
      # simulator's in front, below).
      local package link="$PW_OUT/torch-links"
      package=$("$python" -c 'import os, importlib.util as u; print(os.path.dirname(u.find_spec("torch").origin))') ||
        pw_skip "$python cannot find torch"
      rm -rf "$link"; mkdir -p "$link/torch/lib"
      local e f
      for e in "$package"/*; do [[ "$(basename "$e")" == lib ]] || ln -s "$e" "$link/torch/"; done
      for f in "$package"/lib/*; do
        case "$(basename "$f")" in libamdhip64.so|librocm_smi64.so) ;; *) ln -s "$f" "$link/torch/lib/" ;; esac
      done
      ln -s "$(readlink -f "$build/shim/libamdhip64.so.7")" "$link/torch/lib/libamdhip64.so"
      ln -s "$(readlink -f "$build/shim/librocm_smi64.so")" "$link/torch/lib/librocm_smi64.so"
      PW_ENV=(PYTHONPATH="$link" VGPU_GPU="$PW_SIM_PROFILE" VGPU_DEVICE_COUNT=1
              VGPU_MEMORY_RAM_MB="${VGPU_MEMORY_RAM_MB:-4096}" VGPU_VRAM_MB="${VGPU_VRAM_MB:-3072}"
              VGPU_TELEMETRY_PATH="$PW_OUT/vgpu-telemetry"
              TRITON_CACHE_DIR="$PW_OUT/triton" TORCHINDUCTOR_CACHE_DIR="$PW_OUT/inductor")
      if [[ "$flavour" == vllm ]]; then
        PW_ENV+=(LD_LIBRARY_PATH="$build/shim" VLLM_ENABLE_V1_MULTIPROCESSING=0)
        mkdir -p "$PW_OUT/vgpu-telemetry"
      fi
      # PyTorch on a simulated device can grow to many gigabytes of host memory and these machines
      # are shared: hold the run to a cap where systemd can, as pantheonsim's own tests do.
      if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
        PW_PREFIX=(systemd-run --user --scope -q -p "MemoryMax=${VGPU_TORCH_MEMORY_MAX:-10G}" -p MemorySwapMax=0)
      fi ;;
    *) pw_skip "target $PW_TARGET is not supported by this workload" ;;
  esac
  command -v "$python" >/dev/null 2>&1 || pw_skip "$python not found"
  PW_PY="$python"
  if [[ "$PW_TARGET" != sim:* ]]; then
    "$PW_PY" -c 'import torch' 2>/dev/null || pw_skip "$PW_PY cannot import torch (set PW_PYTHON to a Python with PyTorch)"
  fi
}

pw_python() {
  # -I would hide PYTHONPATH from the ROCm link tree; the scripts run are this repository's own.
  env -u ROCM_PATH -u ROCM_HOME "${PW_ENV[@]}" "${PW_PREFIX[@]}" "$PW_PY" "$@"
}
