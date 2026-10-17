#!/usr/bin/env bash
# bootstrap.sh: run ON the rig (Ubuntu 22.04 Deep Learning Base AMI) from the repository checkout, once per rig.
# Installs what docs/rigs.md and docs/first-gpu-run-a10g.md describe by hand:
#   uv + Python 3.12, the cu130 PyTorch/transformers/diffusers venv the pytorch workloads find by themselves,
#   a separate vLLM venv (PW_VLLM_VENV, for the vllm-* workloads: run them with PW_PYTHON=<it>/bin/python),
#   and one llama.cpp CUDA build for the rig's compute capability, kept in $PW_CACHE where lc_setup looks for it
#   (built once here so concurrent --pack jobs never race to build it).
# Usage: tools/rig/bootstrap.sh [--archs 80] [--jobs 48] [--no-vllm]
# Needs network (uv, PyPI, download.pytorch.org, github.com for llama.cpp). No credentials are used or stored.
set -euo pipefail
archs=80 jobs=$(nproc) vllm=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archs) archs=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --no-vllm) vllm=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
here="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
export PW_CACHE="${PW_CACHE:-$HOME/.cache/pantheonworkloads}"
mkdir -p "$PW_CACHE"
log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

log "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader
command -v uv >/dev/null || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
uv python install 3.12
venv="${PW_VENV_ROOT:-$HOME/.cache/pantheonworkloads/venvs}/cu130"
if [[ ! -x "$venv/bin/python" ]]; then
  log "torch venv $venv"
  uv venv --python 3.12 "$venv"
  env -u LD_LIBRARY_PATH uv pip install --python "$venv/bin/python" --index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match -r "$here/workloads/_pytorch/requirements-torch.txt"
  for r in common diffusion vlm; do
    env -u LD_LIBRARY_PATH uv pip install --python "$venv/bin/python" -r "$here/workloads/_pytorch/requirements-$r.txt"
  done
fi
env -u LD_LIBRARY_PATH "$venv/bin/python" -c 'import torch, transformers, diffusers; print("torch", torch.__version__, torch.version.cuda, "cuda ok:", torch.cuda.is_available(), "transformers", transformers.__version__, "diffusers", diffusers.__version__)'

if (( vllm )); then
  vv="${PW_VLLM_VENV:-$HOME/venvs/vllm}"
  if [[ ! -x "$vv/bin/python" ]]; then
    log "vLLM venv $vv"
    uv venv --python 3.12 "$vv"
    env -u LD_LIBRARY_PATH uv pip install --python "$vv/bin/python" vllm==0.30.0 ninja
  fi
  env -u LD_LIBRARY_PATH "$vv/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'
fi

. "$here/tools/llamacpp/pin.env"
prefix="$PW_CACHE/llama.cpp-$LLAMACPP_TAG-cuda"
if [[ ! -x "$prefix/bin/llama-bench" || ! -x "$prefix/bin/llama-completion" ]]; then
  log "llama.cpp $LLAMACPP_TAG for CUDA arch $archs (-j$jobs)"
  PW_BUILD_DIR="$PW_CACHE/llama.cpp-build" PW_CUDA_ARCHS=$archs PW_BUILD_JOBS=$jobs "$here/tools/llamacpp/build.sh" cuda "$prefix.part"
  rm -rf "$prefix"; mv "$prefix.part" "$prefix"
fi
ls "$prefix/bin"
log "done"
