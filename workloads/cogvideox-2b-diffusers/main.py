"""cogvideox-2b-diffusers: CogVideoX-2b. Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("zai-org/CogVideoX-2b", "1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01", kind="video", steps=30, guidance=6.0, width=720, height=480, frames=49, variant=None, label="cogvideox-2b", kwargs={}, negative=True)
