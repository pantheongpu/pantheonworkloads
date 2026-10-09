"""The library versions a workload's result line reports (`versions`), which bin/pw copies into bench records as
`environment.runtime_versions` (docs/benchmarks.md). Read from the installed distributions with importlib.metadata,
never from a pin file, so a record says what actually ran. Distributions that are not installed are left out.

    import versions; versions.collect(versions.ORT)            # {"onnxruntime-gpu": "1.30.0", "numpy": "2.5.3", ...}

The tools run under `python3 -I` without their directory on sys.path, so they load this file by path (see ort_tasks.py).
"""
from importlib import metadata

# ONNX Runtime payloads (tools/ort_tasks.py, ort_vision.py, speech_tasks.py, text_tasks.py). Both onnxruntime
# distributions are listed because both can be installed (rapidocr_onnxruntime pulls in the CPU one), so a record
# shows both rather than guess which was imported.
ORT = ("onnxruntime", "onnxruntime-gpu", "numpy", "onnx", "rapidocr_onnxruntime", "opencv-python", "pillow",
       "pyclipper", "shapely", "sentencepiece", "spacy", "thinc", "cupy-cuda12x", "py3langid")
SPACY = ("spacy", "cupy-cuda12x", "numpy", "thinc", "en_core_web_sm")


def installed(name):
    """The installed version of distribution `name`, or None."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect(packages, extra=None):
    """{distribution: version} for the installed ones of `packages`, plus the non-empty entries of `extra`
    (versions of things that are not Python packages, e.g. a whisper.cpp tag)."""
    found = {p: v for p in packages if (v := installed(p))}
    found.update({k: v for k, v in (extra or {}).items() if v})
    return found
