"""glm-ocr-pytorch: GLM-OCR. Shared body: ../_pytorch/vlm.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402
vlm.run("zai-org/GLM-OCR", "2e85a62840ccac27daa451df36c736c4636b8628", dtype="bfloat16", new=48, margin=0.5, label="glm-ocr", prompt='Text Recognition:', image="text", expect=['quick', 'brown', 'fox'], extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)
