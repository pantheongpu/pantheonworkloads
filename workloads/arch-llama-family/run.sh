#!/usr/bin/env bash
# arch-llama-family: tiny seeded-random-weight models of open architectures, built from transformers configs
# (no weights, no download). Engine: ../_shared/arch_cases.py, group "llama-family". Contract: docs/workload-contract.md.
# Exits 77 when torch, transformers or a GPU is missing. See ../_shared/arch_run.sh for PW_PYTHON.
set -o pipefail
source "$PW_WORKLOAD_DIR/../_shared/arch_run.sh"
pw_arch --mode functional --group llama-family
