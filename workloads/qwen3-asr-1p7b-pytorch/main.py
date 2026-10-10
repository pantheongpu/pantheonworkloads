"""qwen3-asr-1p7b-pytorch: Qwen3-ASR-1.7B. Shared body: ../_pytorch/asr.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import asr  # noqa: E402
asr.run("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5", engine="qwen3-asr", label="qwen3-asr-1p7b")
