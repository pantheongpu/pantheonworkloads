"""bge-reranker-large-pytorch: bge-reranker-large. Shared body: ../_pytorch/retrieval.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import retrieval  # noqa: E402
retrieval.run("BAAI/bge-reranker-large", "55611d7bca2a7133960a6d3b71e083071bbfc312", head="seqcls", label="bge-reranker-large")
