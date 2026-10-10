"""wan22-t2v-a14b-diffusers: Wan2.2-T2V-A14B. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "5be7df9619b54f4e2667b2755bc6a756675b5cd7", kind="video", steps=30, guidance=4.0, width=832, height=480, frames=33, variant=None, label="wan22-t2v-a14b", kwargs={}, negative=True)
