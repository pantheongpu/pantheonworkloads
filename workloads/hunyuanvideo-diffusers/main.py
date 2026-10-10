"""hunyuanvideo-diffusers: HunyuanVideo (diffusers conversion). Shared body: ../_pytorch/diffusion.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("hunyuanvideo-community/HunyuanVideo", "e8c2aaa66fe3742a32c11a6766aecbf07c56e773", kind="video", steps=30, guidance=1.0, width=848, height=480, frames=33, variant=None, label="hunyuanvideo", kwargs={}, negative=False)
