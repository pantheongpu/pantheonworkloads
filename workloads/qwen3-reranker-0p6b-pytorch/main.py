"""qwen3-reranker-0p6b-pytorch: Qwen3-Reranker-0.6B. Shared body: ../_pytorch/retrieval.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import retrieval  # noqa: E402
retrieval.run("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473", head="qwen3-reranker", label="qwen3-reranker-0p6b")
