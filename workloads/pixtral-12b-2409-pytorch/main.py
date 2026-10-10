"""pixtral-12b-2409-pytorch: Pixtral-12B-2409 (HF-format conversion). Shared body: ../_pytorch/vlm.py (functional and bench modes). Verified on an L40S (stage 2a)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402
vlm.run("mistral-community/pixtral-12b", "c2756cbbb9422eba9f6c5c439a214b0392dfc998", dtype="bfloat16", new=32, margin=0.5, label="pixtral-12b-2409", prompt='Describe this image in one sentence.', image="astronaut", expect=None, extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)
