"""chronos-bolt-base-pytorch: Chronos-Bolt Base. Shared body: ../_pytorch/timeseries.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import timeseries  # noqa: E402
timeseries.run("amazon/chronos-bolt-base", "5d9f166d69f47aef3401367a7b842e78fe97b121", pipeline="bolt", label="chronos-bolt-base")
