"""glm46v-pytorch: GLM-4.6V. Shared body: ../_pytorch/vlm.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vlm  # noqa: E402
vlm.run("zai-org/GLM-4.6V", "4e2d47eb0b41c5280d8294b17cef9e94fdcfff46", dtype="bfloat16", new=32, margin=0.5, label="glm46v", prompt='Describe this image in one sentence.', image="astronaut", expect=None, extra_allow=("*.model", "*.jinja", "*.tiktoken"), device_map="auto", exact_weights=True)
