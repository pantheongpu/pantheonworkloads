#!/usr/bin/env bash
# lib-fft-linalg. Contract: docs/workload-contract.md. Ops and checks: ../_lib/grp_lib_fft_linalg.py, framework ../_lib/libkernels.py.
# Exits 77 when there is no PyTorch (or GPU / pantheonsim for those targets); see ../_lib/env.sh.
source "$PW_WORKLOAD_DIR/../_lib/env.sh"
lib_run lib-fft-linalg
