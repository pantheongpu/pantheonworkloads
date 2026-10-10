"""chronos-t5-large-pytorch: Chronos T5 Large. Shared body: ../_pytorch/timeseries.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import timeseries  # noqa: E402
timeseries.run("amazon/chronos-t5-large", "0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a", pipeline="t5", label="chronos-t5-large")
