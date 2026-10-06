#!/usr/bin/env bash
# Prints the path of a Python that has CPU-only PyTorch and transformers, making it first if needed.
#
#   PW_PYTHON=$(tools/torch-cpu-env.sh) bin/pw run arch-llama-family --target cpu
#
# Why: PyPI's Linux torch wheel is the CUDA build (about 900 MB plus several GB of NVIDIA
# libraries) and download.pytorch.org may be blocked. conda-forge (conda.anaconda.org) carries a
# CPU build. This downloads micromamba from conda-forge (a static binary), then creates an
# environment with `pytorch=*=cpu_*`, `transformers`, `numpy`, `pyyaml` from conda-forge only.
#
# Env: PW_TORCH_CPU_PREFIX  where the environment goes (default ~/.cache/pantheonworkloads/torch-cpu)
#      PW_MAMBA_ROOT        micromamba's package cache (default <prefix>/../mamba-root); share it to save downloads
#      PW_TORCH_SPEC        extra/override package specs, space separated (default below)
#      PW_MICROMAMBA        path of an existing micromamba binary
# stdout: only the python path (progress goes to stderr). Exit 77 when it cannot be made (offline).
set -eo pipefail
PREFIX="${PW_TORCH_CPU_PREFIX:-$HOME/.cache/pantheonworkloads/torch-cpu}"
ROOT="${PW_MAMBA_ROOT:-$(dirname "$PREFIX")/mamba-root}"
SPEC="${PW_TORCH_SPEC:-pytorch=2.13.0=cpu_* transformers numpy pyyaml}"
MM_URL="https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.0.5-0.tar.bz2"
MM_SHA256="bfc2e3a414d651af7508c49998a12b5cf3c7029d56c5ef37c9a3248cd7faef78"
PY="$PREFIX/bin/python"

if [[ -x "$PY" ]] && "$PY" -c 'import torch, transformers' 2>/dev/null; then echo "$PY"; exit 0; fi
if [[ "${1:-}" == --check ]]; then exit 1; fi

fail() { echo "SKIP: $*" >&2; exit 77; }
MM="${PW_MICROMAMBA:-$(dirname "$PREFIX")/micromamba/bin/micromamba}"
if [[ ! -x "$MM" ]]; then
  [[ "$(uname -m)" == x86_64 && "$(uname -s)" == Linux ]] || fail "this helper only knows the linux-64 micromamba; set PW_MICROMAMBA"
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  curl -fsSL -o "$tmp/mm.tar.bz2" "$MM_URL" || fail "cannot download micromamba from conda.anaconda.org"
  echo "$MM_SHA256  $tmp/mm.tar.bz2" | sha256sum -c --quiet - || fail "micromamba checksum mismatch"
  mkdir -p "$(dirname "$MM")/.."; tar xjf "$tmp/mm.tar.bz2" -C "$(dirname "$MM")/.." bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$ROOT"
# shellcheck disable=SC2086
"$MM" create -y -q -p "$PREFIX" --override-channels -c conda-forge $SPEC >&2 || fail "micromamba could not solve/download $SPEC"
"$PY" -c 'import torch, transformers' >&2 || fail "environment built but torch/transformers do not import"
echo "$PY"
