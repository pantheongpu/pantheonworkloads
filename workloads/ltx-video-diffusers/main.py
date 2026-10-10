"""ltx-video-diffusers: LTX-Video 0.9.x. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("Lightricks/LTX-Video", "8984fa25007f376c1a299016d0957a37a2f797bb", kind="video", steps=30, guidance=3.0, width=704, height=480, frames=49, variant=None, label="ltx-video", kwargs={}, negative=True)
