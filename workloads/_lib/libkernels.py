#!/usr/bin/env python3
"""Framework of the lib-* workloads: library-kernel checks on seeded inputs, no model, no download.

    libkernels.py <group>                  run a group's ops (the functional workloads)
    libkernels.py <group> --bench          the throughput numbers of the group (lib-kernels-bench, gpu only)
    libkernels.py <group> --calibrate      measure float32-vs-float64 and run-to-run differences and print
                                           suggested tolerances (how the manifests' bounds were derived)

An op is a function of a context `c`: `c(x)` turns a seeded float32 CPU tensor into what this run computes
with (device + dtype), `c.raw(x)` moves a non-float tensor (indices) to the device, `c.module(m)` does the
same for a module. The same function is run twice: on the target device in the op's dtype, and on the CPU in
float64 from the same inputs (rounded to the op's dtype first, so the reference sees the same numbers).

Per op the run

  * fails if the device result is further from the float64 result than the op's bound (max |error| as a
    fraction of the reference's rms; integer results must be equal), which catches a backend that is wrong
    the same way every time, as a recorded reference could not;
  * reports `"unsupported"` instead of numbers when the backend has no kernel for it (a NotImplementedError or a
    RuntimeError saying so), so a limited backend is described op by op instead of crashing, and the op is never
    silently dropped; the manifest's reference comparison then fails on exactly that key;
  * reports a checksum: for float results [mean |y|, rms, y at three fixed positions], for integer results the
    count, the sum, a position-weighted checksum and the first 8 values.

The last stdout line is the contract's JSON (docs/workload-contract.md). Exit 77: no usable device.
Inputs come from CPU generators with fixed seeds (a GPU's RNG would give other numbers); TF32 is off unless
an op turns it on for itself.
"""
import argparse
import copy
import json
import math
import os
import re
import statistics
import sys
import time
import warnings

import torch

SKIP = 77
DEVICE = os.environ.get("PW_DEVICE", "cpu")
REAL = os.environ.get("PW_TARGET", "cpu") in ("cpu", "gpu")      # metrics only on real targets
UNSUPPORTED = re.compile(
    r"not implemented|not supported|unsupported|not available|not compiled|no available kernel|not enabled|"
    r"does not support|only supports?|could not create|no kernel|not built|not been compiled|cannot be used with|no viable backend", re.I)


class Fail(Exception):
    """An op's own consistency check failed (not a missing kernel)."""


def skip(why):
    print(f"SKIP: {why}", flush=True)
    sys.exit(SKIP)


def setup():
    """Full-precision settings (no TF32, no autotune), deterministic seeds."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        skip(f"PW_DEVICE={DEVICE} but torch.cuda is not available in this build/target")
    torch.manual_seed(0)
    warnings.filterwarnings("ignore")      # beta-feature notices of sparse tensors etc.


def sync():
    if DEVICE != "cpu":
        torch.cuda.synchronize()


def device_name():
    return torch.cuda.get_device_name(0) if DEVICE != "cpu" else "cpu"


# ---------------------------------------------------------------------------------------------- seeded data

class Data:
    """Seeded CPU data. Every group makes one (seed per group) and draws in a fixed order."""

    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.g)

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g)

    def randint(self, low, high, shape):
        return torch.randint(low, high, shape, generator=self.g)

    def perm(self, n):
        return torch.randperm(n, generator=self.g)


def seeded(seed, make):
    """make() with torch's global CPU RNG seeded (module constructors draw from it); the state is not restored."""
    torch.manual_seed(seed)
    return make()


class Ctx:
    """What an op function computes with: a device, a dtype, and (for the reference) the dtype to round inputs to."""

    def __init__(self, dev, dtype, round_to=None):
        self.dev, self.dtype, self.round_to = dev, dtype, round_to

    def __call__(self, x):
        if self.round_to is not None and x.is_floating_point():
            x = x.to(self.round_to)
        return x.to(self.dev, self.dtype) if x.is_floating_point() else x.to(self.dev)

    def leaf(self, x):
        """A fresh tensor that requires grad (never aliases the shared seeded input)."""
        return self(x).detach().clone().requires_grad_()

    def raw(self, x):
        return x.to(self.dev)

    def module(self, m):
        """A copy of m with parameters and buffers converted like inputs are."""
        m = copy.deepcopy(m)
        for t in list(m.parameters()) + [b for b in m.buffers() if b.is_floating_point()]:
            t.data = self(t.data)
        return m.to(self.dev)

    @property
    def is_ref(self):
        return self.round_to is not None


# ---------------------------------------------------------------------------------------------- results

def flat(y):
    parts = []
    for t in (y if isinstance(y, (tuple, list)) else (y,)):
        t = t.detach().cpu()
        if t.is_complex():
            t = torch.view_as_real(t)
        parts.append(t.double().flatten() if t.is_floating_point() else t.long().flatten())
    if any(p.is_floating_point() for p in parts):
        parts = [p.double() for p in parts]
    return torch.cat(parts)


def stats(y):
    """[mean |y|, rms, y at three fixed positions]: sign-safe aggregates plus a few raw elements."""
    v = flat(y)
    n = v.numel()
    out = [float(v.abs().mean()), float(v.pow(2).mean().sqrt()), float(v[0]), float(v[n // 3]), float(v[-1])]
    return [float(f"{x:.9g}") for x in out]


def icheck(y):
    """Exact checksum of integer results."""
    v = flat(y).long()
    n = v.numel()
    w = torch.arange(1, n + 1) % 1000003
    chk = int(((v % 1000003) * w).sum() % 2147483647)
    return f"n={n} sum={int(v.sum())} chk={chk} head={' '.join(map(str, v[:8].tolist()))}"


class Op:
    def __init__(self, name, fn, dtype, bound, kind, ref, note, prec):
        self.name, self.fn, self.dtype, self.bound, self.kind, self.ref, self.note = name, fn, dtype, bound, kind, ref, note
        self.prec = prec or dtype      # the precision class that sets the checksum tolerance (autocast, TF32: not the input dtype)


class Suite:
    """A group of ops. Register with @suite.op(name, ...)."""

    def __init__(self, group):
        self.group, self.ops = group, {}

    def op(self, name, dtype=torch.float32, bound=1e-4, kind="float", ref=None, note="", prec=None):
        def deco(fn):
            assert name not in self.ops, name
            self.ops[name] = Op(name, fn, dtype, bound, kind, ref, note, prec)
            return fn
        return deco


def is_unsupported(e):
    return isinstance(e, NotImplementedError) or (isinstance(e, (RuntimeError, TypeError, ValueError, AttributeError))
                                                  and bool(UNSUPPORTED.search(str(e))))


def evaluate(op, dev):
    """(device result, float64 CPU result) of one op."""
    y = op.fn(Ctx(dev, op.dtype))
    sync()
    rc = Ctx("cpu", torch.float64, round_to=op.dtype)
    r = (op.ref or op.fn)(rc)
    return y, r


def error(op, y, r):
    """max |y - r| as a fraction of rms(r) (floats); 0 / inf for integer results (equal or not)."""
    a, b = flat(y), flat(r)
    if a.shape != b.shape:
        raise Fail(f"shape {tuple(a.shape)} differs from the reference's {tuple(b.shape)}")
    if op.kind == "int":
        return 0.0 if torch.equal(a.long(), b.long()) else math.inf
    if not bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()):
        return math.inf
    rms = float(b.pow(2).mean().sqrt()) or 1.0
    return float((a - b).abs().max()) / rms


def run_suite(suite):
    """Run every op. Returns (output dict, detail, failures)."""
    output, failures, worst, unsupported = {}, [], 0.0, []
    for name, op in suite.ops.items():
        try:
            y, r = evaluate(op, DEVICE)
            err = error(op, y, r)
            if err > op.bound:
                failures.append(f"{name}: differs from the float64 CPU result by {err:.3g} of rms, allowed {op.bound:g}")
                output[name] = "wrong"
                continue
            worst = max(worst, err if err != math.inf else 0.0)
            output[name] = icheck(y) if op.kind == "int" else stats(y)
        except Exception as e:                                   # noqa: BLE001: classified just below
            if is_unsupported(e):
                output[name] = "unsupported"
                unsupported.append(name)
                if DEVICE != "cpu":
                    torch.cuda.empty_cache()
                print(f"unsupported {name}: {str(e).splitlines()[0][:160]}", file=sys.stderr, flush=True)
            else:
                output[name] = "error"
                failures.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    n = len(suite.ops)
    detail = (f"{n} {suite.group} ops on {device_name()}: {n - len(unsupported) - len(failures)} checked, "
              f"{len(unsupported)} unsupported, worst error {worst:.2e} of rms vs the float64 CPU result, torch {torch.__version__}")
    return output, detail, failures


def finish(output, detail, metrics=None):
    rec = {"output": output, "detail": detail, "metrics": (metrics or {}) if REAL else {}}
    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------------------------- calibration

def need(a, b, rel):
    """The abs bound that makes |a-b| <= abs + rel*|b| hold for every statistic."""
    return max([max(abs(x - y) - rel * abs(y), 0.0) for x, y in zip(a, b)] + [0.0])


def calibrate(suite):
    """Per float op: the checksum from the dtype run, from the float64 run, from a 1-thread run and from a
    repeat, plus the max error against float64 (as a fraction of rms). Rows for `suggest`."""
    rows = []
    for name, op in suite.ops.items():
        if op.kind == "int":
            continue
        try:
            y, r = evaluate(op, DEVICE)
        except Exception as e:                                   # noqa: BLE001
            print(f"# {name}: {'unsupported' if is_unsupported(e) else 'error ' + str(e)[:80]}", file=sys.stderr)
            continue
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        y1 = stats(op.fn(Ctx(DEVICE, op.dtype)))
        torch.set_num_threads(threads)
        y2 = stats(op.fn(Ctx(DEVICE, op.dtype)))
        rows.append((name, op, stats(y), stats(r), y1, y2, error(op, y, r)))
    return rows


def suggest(rows, base_abs=1e-4, base_rel=1e-3, headroom=4.0):
    """YAML for `tolerance.fields`, one line per op that needs more than the manifest's default {abs, rel}.

    rel: base_rel, raised to 4 eps of the dtype for fp16/bf16 (an output element can land one or two
         ulp from another backend's).
    abs: headroom x the largest amount by which any checksum entry of the dtype run is outside rel of
         the float64 run, of the 1-thread run, or of a repeat; at least one eps of the checksum's rms.
    Rounded up to 1/2/5 x 10^k. Also prints the table of measurements to stdout (as '#' comments)."""
    def up(x):
        e = 10 ** math.floor(math.log10(x))
        for m in (1, 2, 5, 10):
            if m * e >= x:
                return m * e
    lines = [f"# {'op':36s} abs-needed: dtype-vs-f64  1-thread  repeat | max-error/rms (bound), share of the default tolerance used by dtype-vs-f64"]
    fields = []
    for name, op, s, s64, y1, y2, err in rows:
        half = op.prec in (torch.float16, torch.bfloat16)
        eps = torch.finfo(op.prec).eps if op.prec.is_floating_point else 0
        rel = max(base_rel, up(4 * eps)) if half else base_rel
        needs = [need(s, s64, rel), need(y1, s, rel), need(y2, s, rel)]
        ab = max(max(needs) * headroom, eps * s64[1] if half else 0.0)
        used = max(abs(x - y) / (base_abs + rel * abs(y)) for x, y in zip(s, s64))   # share of the default tolerance used
        lines.append(f"#   {name:36s} {needs[0]:9.2e} {needs[1]:9.2e} {needs[2]:9.2e} | {err:9.2e} ({op.bound:g}) used {used:6.1%}")
        if ab > base_abs or rel > base_rel:
            fields.append(f"    {name}: {{abs: {up(max(ab, base_abs)):.1e}, rel: {rel:.1e}}}")
    return "\n".join(lines + fields)


# ---------------------------------------------------------------------------------------------- benchmarks

def timeit(fn, iters=10, rounds=5, warmup=2):
    """Median seconds per call over `rounds` rounds of `iters` calls."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        times.append((time.perf_counter() - t) / iters)
    return statistics.median(times)


class Bench:
    """Collects metrics; an op that is unsupported or does not fit in memory is left out, not faked."""

    def __init__(self, tiny):
        self.out, self.tiny = {}, tiny

    def size(self, big, small):
        return small if self.tiny else big

    def put(self, key, amount, scale, fn, iters=10):
        """metric = amount / seconds / scale  (e.g. flops/1e12 for TFLOPS, bytes/1e9 for GB/s, 1 for ops/s)."""
        try:
            self.out[key] = float(f"{amount / timeit(fn, iters=iters) / scale:.4g}")
        except Exception as e:                                   # noqa: BLE001
            # best effort: a metric that cannot be measured (no kernel, no memory, a driver error) is left out
            # and named on stderr; the keys present in a record say what ran
            print(f"bench {key}: left out ({type(e).__name__}: {str(e).splitlines()[0][:100] if str(e) else ''})", file=sys.stderr, flush=True)
            if DEVICE != "cpu":
                torch.cuda.empty_cache()


def load_group(name):
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    return importlib.import_module("grp_" + name.replace("-", "_"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("group", help="a group name; with --bench a comma-separated list, metrics merged")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args(argv)
    setup()
    if a.bench:
        b = Bench(tiny=os.environ.get("PW_LIB_SIZE", "large") == "tiny")
        for g in a.group.split(","):
            try:
                load_group(g).bench(b)
            except Exception as e:                               # noqa: BLE001: e.g. a buffer that does not fit; keep the other groups
                print(f"bench {g}: stopped early ({type(e).__name__}: {str(e).splitlines()[0][:100] if str(e) else ''})", file=sys.stderr, flush=True)
        finish(f"{a.group} benchmark", f"{len(b.out)} metrics on {device_name()} ({'tiny' if b.tiny else 'large'} sizes), torch {torch.__version__}", b.out)
        return 0
    suite = load_group(a.group).SUITE
    if a.calibrate:
        print(suggest(calibrate(suite)))
        return 0
    output, detail, failures = run_suite(suite)
    if failures:
        print("\n".join("FAIL " + f for f in failures), file=sys.stderr)
        sys.exit(f"{len(failures)} op(s) failed: " + "; ".join(f.split(":")[0] for f in failures))
    finish(output, detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
