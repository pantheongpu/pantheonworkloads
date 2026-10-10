"""flux1-schnell-diffusers: FLUX.1-schnell (unsloth mirror of the gated official repo). Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("unsloth/FLUX.1-schnell", "9df3faa7ae3b6ddf0b2b69bb78616372897cc65c", kind="image", steps=4, guidance=0.0, size=1024, variant=None, label="flux1-schnell", kwargs={}, negative=False)
