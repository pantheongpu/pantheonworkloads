"""qwen-image-2p1-diffusers: Qwen-Image-2.1. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3", kind="image", steps=30, guidance=4.0, size=1024, variant=None, label="qwen-image-2p1", kwargs={}, negative=False)
