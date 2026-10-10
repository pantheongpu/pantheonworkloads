"""qwen3-vl-235b-a22b-pytorch: Qwen3-VL-235B-A22B-Instruct. Shared body: ../_pytorch/vlm.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402
vlm.run("Qwen/Qwen3-VL-235B-A22B-Instruct", "710c13861be6c466e66de3f484069440b8f31389", dtype="bfloat16", new=32, margin=0.5, label="qwen3-vl-235b-a22b", prompt='Describe this image in one sentence.', image="astronaut", expect=None, extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)
