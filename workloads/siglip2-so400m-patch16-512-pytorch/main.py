"""siglip2-so400m-patch16-512-pytorch: SigLIP 2 so400m patch16 512. Shared body: ../_pytorch/vision.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import vision  # noqa: E402
vision.run("google/siglip2-so400m-patch16-512", "ceea1cba8130d8271436da4828633198c176a775", head="siglip2", label="siglip2-so400m-patch16-512")
