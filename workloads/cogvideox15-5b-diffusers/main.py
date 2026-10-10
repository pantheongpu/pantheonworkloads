"""cogvideox15-5b-diffusers: CogVideoX1.5-5B. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("zai-org/CogVideoX1.5-5B", "fdc5267c90b5c06492985b966e43aae984e189e0", kind="video", steps=30, guidance=6.0, width=1360, height=768, frames=33, variant=None, label="cogvideox15-5b", kwargs={}, negative=True)
