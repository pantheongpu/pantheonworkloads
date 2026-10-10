"""bge-m3-pytorch: BGE-M3 (dense head). Shared body: ../_pytorch/retrieval.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import retrieval  # noqa: E402
retrieval.run("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", head="embed-cls", label="bge-m3")
