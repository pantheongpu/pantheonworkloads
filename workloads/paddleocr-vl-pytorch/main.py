"""paddleocr-vl-pytorch: PaddleOCR-VL. Shared body: ../_pytorch/vlm.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402
vlm.run("PaddlePaddle/PaddleOCR-VL", "7fa00a8c55b735ba51ba49a9058f3f9c57a99a11", dtype="bfloat16", new=48, margin=0.5, label="paddleocr-vl", prompt='OCR:', image="text", expect=['quick', 'brown', 'fox'], extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)
