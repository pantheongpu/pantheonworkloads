"""chatterbox-pytorch: Chatterbox (Resemble AI). Shared body: ../_pytorch/tts.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import tts  # noqa: E402
tts.run("ResembleAI/chatterbox", "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", engine="chatterbox", label="chatterbox")
