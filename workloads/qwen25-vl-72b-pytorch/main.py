"""qwen25-vl-72b-pytorch: Qwen2.5-VL-72B-Instruct. Shared body: ../_pytorch/vlm.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402
vlm.run("Qwen/Qwen2.5-VL-72B-Instruct", "89c86200743eec961a297729e7990e8f2ddbc4c5", dtype="bfloat16", new=32, margin=0.5, label="qwen25-vl-72b", prompt='Describe this image in one sentence.', image="astronaut", expect=None, extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)
