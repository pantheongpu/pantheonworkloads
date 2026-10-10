#!/usr/bin/env python3
"""Print the resolved catalog as a table (a working view for choosing entries; docs/model-registry.md is the record)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_catalog as mc  # noqa: E402

for r in mc.resolve_all():
    hw = r.get("hw")
    need = f"{hw['gpus']}x{hw['gpu_memory_gb']}" if hw else "-"
    print(f"{r['key']:36s} {r['status'][:72]:72s} {r.get('repo', '-')[:46]:46s} {str(r.get('quant') or r.get('dtype') or ''):12s} "
          f"{r.get('weights_gib', 0):8.1f} {need:7s} {r['licence']['label'][:30]}{' R' if r['licence']['restricted'] else ''}")
