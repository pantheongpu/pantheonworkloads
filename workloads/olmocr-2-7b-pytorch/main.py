"""olmocr-2-7b-pytorch: olmOCR-2-7B-1025. Shared body: ../_pytorch/vlm.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402
vlm.run("allenai/olmOCR-2-7B-1025", "e52d6f090b7a9007afffbbd6ce510876222fea93", dtype="bfloat16", new=48, margin=0.5, label="olmocr-2-7b", prompt='Read all the text in the image.', image="text", expect=['quick', 'brown', 'fox'], extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)
