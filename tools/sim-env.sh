#!/usr/bin/env bash
# sim-env.sh: build pantheonsim and find a PyTorch for CUDA 13, so the workloads can run on `sim:` targets
# on a machine with NO NVIDIA toolkit (and no GPU). Idempotent: every step is skipped when its result exists.
#
#   tools/sim-env.sh              set everything up, print the exports on stdout (progress goes to stderr)
#   eval "$(tools/sim-env.sh)"    ... and apply them
#   tools/sim-env.sh --check      only report what is missing (exit 1 if anything is)
#
# What it does, and why:
#  1. The simulator's CUDA shims (libcudart.so.13, libcublas, libcudnn, libnvml ...) are compiled against
#     the CUDA ABI *headers* (pantheonsim CMakeLists.txt, "CUDA_ABI_INCLUDE"; CMake looks in nvcc's toolkit,
#     /usr/local/cuda, or $CUDA_HOME/include). There is no nvcc here and the simulator never needs a compiler:
#     the headers come from the pip wheels PyTorch itself depends on (nvidia-cuda-runtime, -cublas, -cufft,
#     -curand, -cusolver, -cusparse, -nvrtc, -cupti ...) plus nvidia-nvml-dev, -cuda-crt, -npp, -nvjpeg
#     (+ -cuda-cccl for libcu++) for the headers PyTorch does not pull in. They are linked into $SIM_CUDA_HOME/include; the wheels' lib
#     directory is $SIM_CUDA_HOME/lib, from which CMake reads each library's soname major (cublas .13, cufft .12).
#  2. PyTorch's CUDA 13 build is PyPI's default Linux wheel (torch 2.14.1 = +cu130, 0.53 GB, + torchvision; with nvidia-* deps
#     2.8 GB downloaded, 5.3 GB installed). It goes into a venv at $SIM_TORCH_VENV (~/.local/share/torch-cu13),
#     which is where workloads/_shared/torch_env.sh looks. Skipped when less than 4 GB would stay free.
#  3. pantheonsim is built Release into $SIM_BUILD with cmake (`-j$SIM_JOBS`, default 3): the `vgpu` CLI and
#     every shim the workloads load (CUDA driver+runtime, cuBLAS/cuBLASLt, cuDNN, NVML, NVRTC, cuFFT, cuRAND,
#     cuSOLVER, cuSPARSE, NCCL, HIP/HSA, ROCm SMI/AMD SMI), as `cmake --build` targets (the `vgpu_cli` executable and the shim libraries)
#     (see TARGETS below). build/shim/libcudart.so.13 is what pantheonsim's run_pytorch.sh, and
#     torch_env.sh, expect.
#  4. It prints   export VGPU_BUILD_DIR=$SIM_BUILD   (torch_env.sh honours it directly)
#                 export PANTHEONSIM_DIR=<dir>       (a directory whose build/ is $SIM_BUILD: a symlink farm
#                                                      at $SIM_BUILD-dir when SIM_SRC/build is not it, so
#                                                      other tools that insist on $PANTHEONSIM_DIR/build work)
#                 export VGPU_TORCH_CUDA_PYTHON=$SIM_TORCH_VENV/bin/python
#
# sim:amd PyTorch needs a ROCm build of torch (download.pytorch.org/whl/rocm*), which is not on PyPI; this
# script does not fake one. Environment variables (defaults):
#   SIM_ROOT=/home/user (else $HOME)
#   SIM_SRC=$SIM_ROOT/wt-sim-main    pantheonsim checkout (a worktree of origin/main)
#   SIM_BUILD=$SIM_ROOT/sim-main-build   build directory
#   SIM_TORCH_VENV=$HOME/.local/share/torch-cu13
#   SIM_CUDA_HOME=$HOME/.local/share/sim-cuda-home
#   SIM_JOBS=3    SIM_MIN_FREE_GB=4    SIM_SKIP_TORCH=1 (do not install torch)
set -euo pipefail

SIM_ROOT=${SIM_ROOT:-$([[ -d /home/user ]] && echo /home/user || echo "$HOME")}
SIM_SRC=${SIM_SRC:-$SIM_ROOT/wt-sim-main}
SIM_BUILD=${SIM_BUILD:-$SIM_ROOT/sim-main-build}
SIM_TORCH_VENV=${SIM_TORCH_VENV:-$HOME/.local/share/torch-cu13}
SIM_CUDA_HOME=${SIM_CUDA_HOME:-$HOME/.local/share/sim-cuda-home}
SIM_JOBS=${SIM_JOBS:-3}
SIM_MIN_FREE_GB=${SIM_MIN_FREE_GB:-4}
CHECK=0; [[ "${1:-}" == --check ]] && CHECK=1

# CMake targets: the CLI and every shim a workload loads (names are CMake target names in pantheonsim).
TARGETS=(vgpu vgpu_cli vgpucudart vgpucuda vgpucublas vgpucublaslt vgpucudnn vgpunvml vgpunvrtc vgpucufft vgpucurand
         vgpucusolver vgpucusolvermg vgpucusparse vgpunccl vgpucupti vgpuhip vgpuamdsmi vgpursmi vgpurocprof)

log() { echo "sim-env: $*" >&2; }
free_gb() { df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9; }
missing=()

# ---- 2. PyTorch for CUDA 13 -------------------------------------------------------------------------------
torch_ok() { "$SIM_TORCH_VENV/bin/python" -c 'import torch,sys; sys.exit(0 if (torch.version.cuda or "").startswith("13") else 1)' 2>/dev/null; }
if ! torch_ok; then
  if (( CHECK )); then missing+=("PyTorch for CUDA 13 in $SIM_TORCH_VENV");
  elif [[ -n "${SIM_SKIP_TORCH:-}" ]]; then log "SIM_SKIP_TORCH set: not installing torch"
  elif (( $(free_gb) < 5 + SIM_MIN_FREE_GB + 2 )); then
    log "NOT installing torch: $(free_gb) GB free, the wheels need ~8 GB at peak and $SIM_MIN_FREE_GB GB must stay free"
    missing+=("PyTorch for CUDA 13 (no disk)")
  else
    log "installing torch (CUDA 13 build from PyPI) into $SIM_TORCH_VENV"
    python3 -m venv "$SIM_TORCH_VENV"
    "$SIM_TORCH_VENV/bin/pip" install --no-cache-dir -q torch torchvision numpy >&2
  fi
fi

# ---- 1. CUDA ABI headers ------------------------------------------------------------------------------------
NV="$SIM_TORCH_VENV/lib/python3*/site-packages/nvidia/cu13"
NV=$(ls -d $NV 2>/dev/null | head -1 || true)
if [[ ! -e "$SIM_CUDA_HOME/include/nvml.h" || ! -e "$SIM_CUDA_HOME/include/npp.h" || ! -e "$SIM_CUDA_HOME/include/nv/target" || ! -e "$SIM_CUDA_HOME/include/cublas_v2.h" ]]; then
  if (( CHECK )); then missing+=("CUDA ABI headers in $SIM_CUDA_HOME")
  elif [[ -z "$NV" ]]; then log "no torch venv, so no CUDA headers"; missing+=("CUDA ABI headers (need torch venv)")
  else
    log "assembling CUDA headers in $SIM_CUDA_HOME"
    rm -rf "$SIM_CUDA_HOME"; mkdir -p "$SIM_CUDA_HOME/include" "$SIM_CUDA_HOME/dl"
    for e in "$NV"/include/*; do ln -s "$e" "$SIM_CUDA_HOME/include/"; done
    ln -s "$NV/lib" "$SIM_CUDA_HOME/lib"
    # headers PyTorch's dependencies do not carry; only the include/ trees of these wheels are kept
    "$SIM_TORCH_VENV/bin/pip" download --no-cache-dir -q --no-deps -d "$SIM_CUDA_HOME/dl" \
        'nvidia-nvml-dev==13.0.*' 'nvidia-cuda-crt==13.0.*' 'nvidia-npp==13.0.*' 'nvidia-nvjpeg==13.0.*' 'nvidia-cuda-cccl==13.0.*' >&2 ||
      "$SIM_TORCH_VENV/bin/pip" download --no-cache-dir -q --no-deps -d "$SIM_CUDA_HOME/dl" \
        nvidia-nvml-dev nvidia-cuda-crt nvidia-npp nvidia-nvjpeg nvidia-cuda-cccl >&2
    python3 -I - "$SIM_CUDA_HOME" <<'PY'
import sys, zipfile, glob, os
home = sys.argv[1]
for w in glob.glob(home + "/dl/*.whl"):
    z = zipfile.ZipFile(w)
    for n in z.namelist():
        if "/include/" in n and not n.endswith("/"):
            dest = os.path.join(home, "include", n.split("/include/", 1)[1])
            if os.path.islink(dest): os.remove(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            open(dest, "wb").write(z.read(n))
PY
    rm -rf "$SIM_CUDA_HOME/dl"
  fi
fi

# ---- 3. build -----------------------------------------------------------------------------------------------
if [[ ! -e "$SIM_BUILD/shim/libcudart.so.13" || ! -x "$SIM_BUILD/vgpu" || ! -e "$SIM_BUILD/shim/libamdhip64.so.7" ]]; then
  if (( CHECK )); then missing+=("pantheonsim build in $SIM_BUILD")
  elif [[ ! -e "$SIM_CUDA_HOME/include/cuda_runtime_api.h" ]]; then missing+=("pantheonsim build (no CUDA headers)")
  else
    [[ -d "$SIM_SRC" ]] || { log "no pantheonsim checkout at $SIM_SRC (set SIM_SRC)"; exit 2; }
    log "building pantheonsim ($SIM_SRC) into $SIM_BUILD, -j$SIM_JOBS"
    CUDA_HOME="$SIM_CUDA_HOME" cmake -S "$SIM_SRC" -B "$SIM_BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_ABI_INCLUDE="$SIM_CUDA_HOME/include" >&2
    cmake --build "$SIM_BUILD" -j"$SIM_JOBS" --target "${TARGETS[@]}" >&2
  fi
fi

# ---- 4. exports ---------------------------------------------------------------------------------------------
# torch_env.sh reads $VGPU_BUILD_DIR first; other pantheonsim tooling wants $PANTHEONSIM_DIR/build.
if (( CHECK )); then
  (( ${#missing[@]} )) && { for m in "${missing[@]}"; do log "MISSING: $m"; done; exit 1; }
  log "everything is in place"; exit 0
fi
farm="$SIM_BUILD-dir"
if [[ "$(readlink -f "$SIM_SRC/build" 2>/dev/null)" != "$(readlink -f "$SIM_BUILD")" ]]; then
  mkdir -p "$farm"; ln -sfn "$SIM_BUILD" "$farm/build"
  for e in "$SIM_SRC"/*; do [[ "$(basename "$e")" == build ]] || ln -sfn "$e" "$farm/$(basename "$e")"; done
  pdir="$farm"
else pdir="$SIM_SRC"; fi

if (( ${#missing[@]} )); then
  for m in "${missing[@]}"; do log "MISSING: $m"; done
  (( CHECK )) && exit 1
fi
echo "export VGPU_BUILD_DIR=$SIM_BUILD"
echo "export PANTHEONSIM_DIR=$pdir"
[[ -x "$SIM_TORCH_VENV/bin/python" ]] && echo "export VGPU_TORCH_CUDA_PYTHON=$SIM_TORCH_VENV/bin/python"
echo "export CUDA_HOME=$SIM_CUDA_HOME"
exit 0
