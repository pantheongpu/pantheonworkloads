"""dinov2-giant-pytorch: DINOv2 giant. Shared body: ../_pytorch/vision.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vision  # noqa: E402
vision.run("facebook/dinov2-giant", "611a9d42f2335e0f921f1e313ad3c1b7178d206d", head="dino", label="dinov2-giant")
