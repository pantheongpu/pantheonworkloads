"""parakeet-tdt-0p6b-v3-pytorch: parakeet-tdt-0.6b-v3. Shared body: ../_pytorch/asr.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import asr  # noqa: E402
asr.run("nvidia/parakeet-tdt-0.6b-v3", "541d1f99c6b0c3cd0b11a95167540bb8edefd82b", engine="transformers-pipeline", label="parakeet-tdt-0p6b-v3")
