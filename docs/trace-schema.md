# Trace schema (DRAFT v0, not implemented)

Goal: capture the same events from a real GPU and from a pantheonsim profile and compare them,
to check the simulator's functional models. Nothing here is implemented yet; it is written
down early so both sides agree on the shape before any capture code exists.

One trace is a JSON-lines file. The first line is a header; every other line is one event.

```json
{"schema": "pw-trace/0", "workload": "smollm2-135m-ollama", "target": "gpu", "device": "NVIDIA H100", "driver": "...", "runtime": "...", "captured_with": "cupti|rocprofv3|sim"}
{"seq": 0, "kind": "kernel", "name": "...", "grid": [1,1,1], "block": [128,1,1], "shared_bytes": 0, "args_hash": "..."}
{"seq": 1, "kind": "memcpy", "dir": "h2d|d2h|d2d", "bytes": 4096}
{"seq": 2, "kind": "sync", "scope": "device|stream"}
```

Compared between targets: the ordered sequence of `kind`/`name`/`grid`/`block`/`bytes`, and
hashes of kernel outputs where they are deterministic. **Not** compared: timestamps and
durations (the simulator's are meaningless), and anything that depends on scheduling order
between independent streams.

Open questions: how to name fused or JIT-compiled kernels stably across driver versions; how
to hash outputs of non-deterministic kernels (atomics, split-K); which counters (CUPTI / rocprof)
are worth comparing at all.
