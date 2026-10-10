"""qwen3-reranker-8b-pytorch: Qwen3-Reranker-8B. Shared body: ../_pytorch/retrieval.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import retrieval  # noqa: E402
retrieval.run("Qwen/Qwen3-Reranker-8B", "77d193c791ed757ca307ee72715aa132723da912", head="qwen3-reranker", label="qwen3-reranker-8b")
