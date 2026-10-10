"""hunyuanvideo-15-720p-t2v-diffusers: HunyuanVideo-1.5 720p T2V. Shared body: ../_pytorch/diffusion.py (functional and bench modes). Verified on an L40S (stage 2a)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import diffusion  # noqa: E402
diffusion.run("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", "f4dbc4a1efa4ac8ea56680cdf79d9f455105e814", kind="video", steps=30, guidance=6.0, width=848, height=480, frames=33, variant=None, label="hunyuanvideo-15-720p-t2v", kwargs={}, negative=False)
