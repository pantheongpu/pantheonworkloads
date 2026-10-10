"""wan22-ti2v-5b-diffusers: Wan2.2-TI2V-5B. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "b8fff7315c768468a5333511427288870b2e9635", kind="video", steps=30, guidance=5.0, width=832, height=480, frames=33, variant=None, label="wan22-ti2v-5b", kwargs={}, negative=True)
