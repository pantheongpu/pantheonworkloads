"""sam2p1-hiera-large-pytorch: SAM 2.1 Hiera Large. Shared body: ../_pytorch/vision.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vision  # noqa: E402
vision.run("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610", head="sam2", label="sam2p1-hiera-large")
