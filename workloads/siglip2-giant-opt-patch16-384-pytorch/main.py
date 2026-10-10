"""siglip2-giant-opt-patch16-384-pytorch: SigLIP 2 giant-opt patch16 384. Shared body: ../_pytorch/vision.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vision  # noqa: E402
vision.run("google/siglip2-giant-opt-patch16-384", "a713301b217d38485fb2204c808367d10bc3cc40", head="siglip2", label="siglip2-giant-opt-patch16-384")
