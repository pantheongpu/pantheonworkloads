"""qwen3-vl-4b-pytorch: Qwen3-VL-4B-Instruct describes one fixed image with greedy decoding. Shared body: ../_pytorch/vlm.py (functional and bench modes)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402

vlm.run("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", dtype="float16", new=32, margin=0.1, label="qwen3-vl-4b")
