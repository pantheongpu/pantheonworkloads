"""qwen3-reranker-4b-pytorch: Qwen3-Reranker-4B. Shared body: ../_pytorch/retrieval.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import retrieval  # noqa: E402
retrieval.run("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5", head="qwen3-reranker", label="qwen3-reranker-4b")
