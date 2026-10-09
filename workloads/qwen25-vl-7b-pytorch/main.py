"""qwen25-vl-7b-pytorch: Qwen2.5-VL-7B-Instruct describes one fixed image with greedy decoding. Shared body: ../_pytorch/vlm.py (functional and bench modes)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402

vlm.run("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", dtype="float16", new=32, margin=0.1, label="qwen25-vl-7b")
