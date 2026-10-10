"""f5-tts-pytorch: F5-TTS (v1 Base). Shared body: ../_pytorch/tts.py (functional and bench modes). WRITTEN, NEVER RUN."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import tts  # noqa: E402
tts.run("SWivid/F5-TTS", "84e5a410d9cead4de2f847e7c9369a6440bdfaca", engine="f5-tts", label="f5-tts")
