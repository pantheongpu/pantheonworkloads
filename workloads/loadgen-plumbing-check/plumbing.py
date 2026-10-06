#!/usr/bin/env python3
"""Drives MLCommons LoadGen (pip: mlcommons-loadgen) against a trivial CPU system under test.

    plumbing.py OUT_DIR

This validates the plumbing only: that LoadGen's Python bindings import and run, that the
Offline and Server scenarios issue queries to a SUT, that every sample comes back, and that the
accuracy log carries the SUT's bytes. It is NOT an MLPerf(R) benchmark: the SUT is not a model,
the settings are not MLPerf's, and the numbers are not reported (see manifest.yaml notes).
Last stdout line: JSON {"output", "detail", "metrics"} as docs/workload-contract.md says.
"""
import array
import json
import pathlib
import re
import struct
import sys
import threading

import mlperf_loadgen as lg

N_SAMPLES = 256


def expected(index):
    """The SUT's 'inference': a tiny 8x8 integer matrix-vector product, reduced to 4 bytes."""
    vec = [(index * (j + 3) + 1) % 251 for j in range(8)]
    acc = sum(((i * 7 + j * 3 + 1) % 13) * vec[j] for i in range(8) for j in range(8))
    return struct.pack("<I", acc % (1 << 32))


class Sut:
    def __init__(self):
        self.buffers = {}           # keeps response memory alive until the test ends
        self.completed = 0
        self.lock = threading.Lock()
        self.sample_of_query = {}   # query id -> sample index

    def issue(self, samples):
        responses = []
        for s in samples:
            buf = array.array("B", expected(s.index))
            self.buffers[s.id] = buf
            addr, length = buf.buffer_info()
            responses.append(lg.QuerySampleResponse(s.id, addr, length))
        with self.lock:
            self.completed += len(responses)
        lg.QuerySamplesComplete(responses)

    def flush(self):
        pass


def run(scenario, mode, out, configure):
    out.mkdir(parents=True, exist_ok=True)
    settings = lg.TestSettings()
    settings.scenario = scenario
    settings.mode = mode
    settings.min_duration_ms = 100
    settings.min_query_count = 1024
    configure(settings)
    log = lg.LogSettings()
    log.log_output.outdir = str(out)
    log.log_output.copy_summary_to_stdout = False
    log.enable_trace = False
    sut = Sut()
    handle = lg.ConstructSUT(sut.issue, sut.flush)
    qsl = lg.ConstructQSL(N_SAMPLES, N_SAMPLES, lambda s: None, lambda s: None)
    try:
        lg.StartTestWithLogSettings(handle, qsl, settings, log)
    finally:
        lg.DestroyQSL(qsl)
        lg.DestroySUT(handle)
    return sut


def summary(out):
    text = (out / "mlperf_log_summary.txt").read_text()
    valid = re.search(r"Result is\s*:\s*(\w+)", text)
    return text, (valid.group(1) if valid else "?")


def main(root):
    root = pathlib.Path(root)

    # 1. Accuracy mode: every sample once; the log must carry exactly the SUT's bytes.
    acc_dir = root / "offline-accuracy"
    sut = run(lg.TestScenario.Offline, lg.TestMode.AccuracyOnly, acc_dir,
              lambda s: setattr(s, "offline_expected_qps", 1_000_000))
    entries = json.loads((acc_dir / "mlperf_log_accuracy.json").read_text())
    got = {e["qsl_idx"]: bytes.fromhex(e["data"]) for e in entries}
    if sorted(got) != list(range(N_SAMPLES)):
        sys.exit(f"accuracy log covers {len(got)} of {N_SAMPLES} samples")
    bad = [i for i, d in got.items() if d != expected(i)]
    if bad:
        sys.exit(f"{len(bad)} responses differ from the SUT's output, first sample {bad[0]}")

    # 2. Performance mode, Offline and Server: the run must finish and LoadGen must call it valid
    #    against the settings given here (which are far from MLPerf's).
    verdicts = {}
    for name, scenario, configure in (
        ("offline", lg.TestScenario.Offline, lambda s: setattr(s, "offline_expected_qps", 1_000_000)),
        ("server", lg.TestScenario.Server,
         lambda s: (setattr(s, "server_target_qps", 500), setattr(s, "server_target_latency_ns", 200_000_000))),
    ):
        d = root / f"{name}-performance"
        sut = run(scenario, lg.TestMode.PerformanceOnly, d, configure)
        _, verdicts[name] = summary(d)
        if sut.completed == 0:
            sys.exit(f"{name}: the SUT was never queried")
    invalid = {k: v for k, v in verdicts.items() if v != "VALID"}
    if invalid:
        sys.exit(f"LoadGen called a run invalid: {invalid} (see {root})")

    print(json.dumps({
        "output": f"loadgen plumbing ok: offline accuracy {N_SAMPLES}/{N_SAMPLES} responses match; "
                  "offline and server performance runs VALID against test-only settings",
        "detail": "NOT an MLPerf result: trivial SUT, non-MLPerf settings",
        "metrics": {},
    }))


if __name__ == "__main__":
    main(sys.argv[1])
