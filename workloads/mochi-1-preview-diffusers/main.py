"""mochi-1-preview-diffusers: Mochi 1 preview. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("genmo/mochi-1-preview", "cd1d0fe89aa8382fcd9ae760bc3d7afe6183579f", kind="video", steps=30, guidance=4.5, width=848, height=480, frames=31, variant='bf16', label="mochi-1-preview", kwargs={}, negative=True)
