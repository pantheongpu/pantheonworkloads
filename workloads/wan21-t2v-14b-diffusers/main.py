"""wan21-t2v-14b-diffusers: Wan2.1-T2V-14B. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("Wan-AI/Wan2.1-T2V-14B-Diffusers", "38ec498cb3208fb688890f8cc7e94ede2cbd7f68", kind="video", steps=30, guidance=5.0, width=832, height=480, frames=33, variant=None, label="wan21-t2v-14b", kwargs={}, negative=True)
