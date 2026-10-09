#!/usr/bin/env bash
# pw-conda-torch.sh: get a CPU-only PyTorch. By default a venv with torch 2.14.1+cpu from
# download.pytorch.org/whl/cpu (tools/cpu-torch-pip.sh), the version the workloads are standardised on.
# PW_TORCH_ROUTE=conda is the old route for hosts where only conda-forge (conda.anaconda.org) and PyPI are
# reachable: conda-forge stops at pytorch 2.13.0, so it gives a DIFFERENT version and warns.
# (PyPI's default Linux torch wheel is the multi-GB CUDA build.)
#
#   tools/pw-conda-torch.sh            prints the path of a Python that imports torch, making one first if needed
#   PW_PYTHON=$(tools/pw-conda-torch.sh) bin/pw run lib-fft-linalg --target cpu
#
# What the conda route does when no environment exists yet: downloads micromamba 2.0.5 from conda-forge (sha256 checked),
# then creates an environment with python 3.12, numpy and pytorch 2.13.0 (the cpu_mkl build) from conda-forge.
# Progress goes to stderr, stdout carries only the Python path.
#
# Environment:
#   PW_CONDA_PREFIX    root (default ~/.cache/pantheonworkloads/conda-torch); the environment is $PW_CONDA_PREFIX/envs/torch
#   PW_MICROMAMBA      an existing micromamba binary to use instead of downloading one
#   PW_CONDA_DRY_RUN=1 solve and print what would be installed, install nothing
set -euo pipefail

PREFIX="${PW_CONDA_PREFIX:-$HOME/.cache/pantheonworkloads/conda-torch}"
ENV_DIR="$PREFIX/envs/torch"
MM_URL="https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.0.5-0.tar.bz2"
MM_SHA256="bfc2e3a414d651af7508c49998a12b5cf3c7029d56c5ef37c9a3248cd7faef78"
SPECS=("python=3.12" "numpy" "pytorch=2.13.0=cpu*")

say() { echo "pw-conda-torch: $*" >&2; }

if [[ "${PW_TORCH_ROUTE:-pip}" != conda ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/cpu-torch-pip.sh" "${PW_CPU_TORCH_VENV:-$PREFIX/venv}"
fi
say "WARNING: PW_TORCH_ROUTE=conda gives torch 2.13.0, not the standard 2.14.1"

if [[ -x "$ENV_DIR/bin/python" ]] && "$ENV_DIR/bin/python" -c 'import torch' 2>/dev/null; then
  echo "$ENV_DIR/bin/python"
  exit 0
fi

MM="${PW_MICROMAMBA:-}"
if [[ -z "$MM" ]]; then
  MM="$PREFIX/bin/micromamba"
  if [[ ! -x "$MM" ]]; then
    [[ "$(uname -s)-$(uname -m)" == Linux-x86_64 ]] || { say "only linux-64 is wired up (set PW_MICROMAMBA for another platform)"; exit 1; }
    say "downloading micromamba from conda-forge"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    curl -fsSL --retry 3 -o "$tmp/mm.tar.bz2" "$MM_URL"
    echo "$MM_SHA256  $tmp/mm.tar.bz2" | sha256sum -c - >&2 || { say "checksum mismatch"; exit 1; }
    mkdir -p "$tmp/x" "$PREFIX/bin"
    tar -xjf "$tmp/mm.tar.bz2" -C "$tmp/x" bin/micromamba
    install -m 755 "$tmp/x/bin/micromamba" "$MM"
  fi
fi

export MAMBA_ROOT_PREFIX="$PREFIX"
args=(create -y -p "$ENV_DIR" -c conda-forge --override-channels "${SPECS[@]}")
if [[ "${PW_CONDA_DRY_RUN:-0}" == 1 ]]; then
  "$MM" "${args[@]}" --dry-run >&2
  exit 0
fi
say "creating $ENV_DIR (CPU PyTorch from conda-forge; a few hundred MB)"
"$MM" "${args[@]}" >&2
"$ENV_DIR/bin/python" -c 'import torch' >&2
echo "$ENV_DIR/bin/python"
