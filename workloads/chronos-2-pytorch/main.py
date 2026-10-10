"""chronos-2-pytorch: Chronos-2. Shared body: ../_pytorch/timeseries.py (functional and bench modes). Run on an A10G (stage 2a, docs/model-registry.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import timeseries  # noqa: E402
timeseries.run("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", pipeline="chronos2", label="chronos-2")
