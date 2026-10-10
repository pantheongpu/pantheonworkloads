"""canary-1b-v2-pytorch: canary-1b-v2. Shared body: ../_pytorch/asr.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import asr  # noqa: E402
asr.run("nvidia/canary-1b-v2", "d455706339a6b32e1aa40f82c713a482a0c938e2", engine="nemo", label="canary-1b-v2")
