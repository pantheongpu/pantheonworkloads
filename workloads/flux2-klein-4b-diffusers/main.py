"""flux2-klein-4b-diffusers: FLUX.2-klein-4B. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294", kind="image", steps=4, guidance=1.0, size=1024, variant=None, label="flux2-klein-4b", kwargs={}, negative=False)
