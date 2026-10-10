"""depth-anything-v2-small-pytorch: Depth Anything V2 Small. Shared body: ../_pytorch/vision.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vision  # noqa: E402
vision.run("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99", head="depth", label="depth-anything-v2-small")
