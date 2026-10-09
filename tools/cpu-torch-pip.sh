#!/usr/bin/env bash
# cpu-torch-pip.sh: make a venv with CPU-only torch 2.14.1 (the version the workloads are standardised on) from
# download.pytorch.org/whl/cpu (about 200 MB), plus numpy, pyyaml and transformers. Used by torch-cpu-env.sh and
# pw-conda-torch.sh, which print the venv's Python. conda-forge stops at pytorch 2.13.0, so it cannot give 2.14.1.
#
#   tools/cpu-torch-pip.sh <venv-dir>      prints <venv-dir>/bin/python; exit 77 when it cannot be made (offline/blocked)
#
# Environment:
#   PW_CPU_TORCH_VERSION  default 2.14.1
#   PW_CPU_TORCH_INDEX    default https://download.pytorch.org/whl/cpu
#   PW_CPU_TORCH_DRY_RUN=1  print the pip commands instead of running them
set -eo pipefail
VENV="${1:?usage: cpu-torch-pip.sh <venv-dir>}"
VERSION="${PW_CPU_TORCH_VERSION:-2.14.1}"
INDEX="${PW_CPU_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$VENV/bin/python"
fail() { echo "SKIP: $*" >&2; exit 77; }

if [[ "${PW_CPU_TORCH_DRY_RUN:-0}" == 1 ]]; then
  echo "python3 -m venv $VENV"
  echo "pip install --index-url $INDEX torch==$VERSION"
  echo "pip install -r $HERE/workloads/_pytorch/requirements-common.txt pyyaml"
  exit 0
fi
if [[ -x "$PY" ]] && "$PY" -c "import torch, transformers, sys; sys.exit(0 if torch.__version__.split('+')[0] == '$VERSION' else 1)" 2>/dev/null; then
  echo "$PY"; exit 0
fi
rm -rf "$VENV"
python3 -m venv "$VENV" >&2 || fail "python3 -m venv failed"
# The CPU index only, with no PyPI fallback: PyPI's Linux torch wheel is the multi-GB CUDA build, and pip would
# pick it silently if the CPU index were unreachable. (The index also hosts torch's pure-Python dependencies.)
"$VENV/bin/pip" install -q --retries 1 --timeout 30 --index-url "$INDEX" "torch==$VERSION" >&2 \
  || { rm -rf "$VENV"; fail "cannot install torch==$VERSION from $INDEX (blocked or offline; conda-forge only has 2.13.0: PW_TORCH_ROUTE=conda uses it, with a different result)"; }
"$VENV/bin/pip" install -q -r "$HERE/workloads/_pytorch/requirements-common.txt" pyyaml >&2 || { rm -rf "$VENV"; fail "cannot install transformers/numpy"; }
"$PY" -c 'import torch, transformers' >&2 || fail "environment built but torch/transformers do not import"
echo "$PY"
