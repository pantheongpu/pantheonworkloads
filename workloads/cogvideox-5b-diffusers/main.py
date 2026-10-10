"""cogvideox-5b-diffusers: CogVideoX-5b. Shared body: ../_pytorch/diffusion.py (functional and bench modes). Verified on an L40S (stage 2a)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("zai-org/CogVideoX-5b", "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321", kind="video", steps=30, guidance=6.0, width=720, height=480, frames=49, variant=None, label="cogvideox-5b", kwargs={}, negative=True)
