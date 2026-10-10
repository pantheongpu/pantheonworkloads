"""depth-anything-v2-large-pytorch: Depth Anything V2 Large. Shared body: ../_pytorch/vision.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vision  # noqa: E402
vision.run("depth-anything/Depth-Anything-V2-Large-hf", "7581137eff8d4e94f6e796d3baea0e9fa79b22d2", head="depth", label="depth-anything-v2-large")
