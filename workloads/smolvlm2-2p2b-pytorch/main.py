"""smolvlm2-2p2b-pytorch: SmolVLM2-2.2B-Instruct answers one fixed question about one fixed image with greedy decoding. Shared body: ../_pytorch/vlm.py (functional and bench modes)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402

vlm.run("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "482adb537c021c86670beed01cd58990d01e72e4", dtype="float16", new=32, margin=0.1, label="smolvlm2-2p2b",
        prompt="What is the person holding or standing next to? Answer in one sentence.")
