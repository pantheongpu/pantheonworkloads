"""bge-reranker-v2-m3-pytorch: bge-reranker-v2-m3. Shared body: ../_pytorch/retrieval.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import retrieval  # noqa: E402
retrieval.run("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", head="seqcls", label="bge-reranker-v2-m3")
