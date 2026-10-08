# fp8 cast semantics: why `cast_roundtrip_fp8_e4m3fn` failed on the A10G

Date of the probe: 2026-10-08. Probe: `tools/fp8_probe.py` (torch only). Raw JSON of every run: `docs/fp8-probe-results/`.

## The question

`lib-attention-precision` failed on a real A10G (sm_86) in `cast_roundtrip_fp8_e4m3fn`: the inputs include 448.0, 449.0 and 57344.0, the CPU
reference holds 448.0 for the out-of-range ones (saturated) and the card gave NaN. The simulated A100 and T4 passed against that
reference. Candidates: (a) the architecture (hardware saturating convert on sm_89+, software NaN-on-overflow before); (b) the PyTorch
version (the reference was recorded with conda-forge torch 2.13.0; the workloads' pin is `torch==2.10.0`); (c) plain PyTorch behaviour,
CUDA NaN and CPU saturating, whatever the card; (d) a simulator conversion bug.

## Answer: (b), the PyTorch version. Not the card, not CPU vs CUDA, not the simulator.

PyTorch 2.10.0 converts a float above e4m3fn's range to NaN; PyTorch 2.13.0 and 2.14.1 saturate to +-448. This holds on the CPU and on
CUDA alike, for float32, float64, float16 and bfloat16 sources, on an A10G (sm_86) and an L4 (sm_89), and in the simulator for sm_75,
sm_80 and sm_86. Candidate (a) is ruled out by the two real cards behaving identically per torch version. (c) is ruled out because the CPU
gives NaN too under 2.10.0 and saturates too under 2.13.0+. (d) is ruled out because every simulated GPU gave results identical to
the real card under the same torch (all 24 result rows, for torch 2.10.0+cu130 and 2.14.1+cu130).

So the failure was a mismatch between the torch of the first GPU run (the workloads' pin, 2.10.0, inferred from the pin; the run's torch version was
not recorded) and the torch of the CPU reference (2.13.0). `docs/sim-validation.md` runs used torch 2.14.1 (PyPI's default wheel),
which saturates, so they agree with the reference and the earlier note in `docs/first-gpu-run-a10g.md` ("CUDA semantics ... the
simulator must produce NaN to match hardware") was wrong: the simulator is right under both versions.

## Probe results (float32 -> float8_e4m3fn -> float64; identical rows for the other three source dtypes)

Inputs: `[448, 449, 480, 500, 57344, -449, inf, nan, 2^-10, 1e-30]`.

| Card | sm | torch | device | 448 | 449 | 480 | 500 | 57344 | -449 | inf | nan | 2^-10 | 1e-30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A10G, L4 | 86, 89 | 2.10.0 (+cu128, and on the L4 also +cu130) | CUDA and CPU | 448 | 448 | NaN | NaN | NaN | -448 | NaN | NaN | 0 | 0 |
| A10G, L4 | 86, 89 | 2.13.0+cu130 | CUDA and CPU | 448 | 448 | 448 | 448 | 448 | -448 | 448 | NaN | 0 | 0 |
| A10G, L4 | 86, 89 | 2.14.1+cu130 | CUDA and CPU | 448 | 448 | 448 | 448 | 448 | -448 | 448 | NaN | 0 | 0 |
| sim A10, A100, RTX 3060, T4 | 86, 80, 86, 75 | 2.10.0+cu130 | CUDA (simulated) | 448 | 448 | NaN | NaN | NaN | -448 | NaN | NaN | 0 | 0 |
| sim A10, A100, RTX 3060, T4 | 86, 80, 86, 75 | 2.14.1+cu130 | CUDA (simulated) | 448 | 448 | 448 | 448 | 448 | -448 | 448 | NaN | 0 | 0 |

Notes:

- 449 is not an overflow: it rounds to 448 (e4m3fn's top binade has step 32 and the next code, 480, is the NaN encoding, so
  the largest finite value is 448 and rounding to nearest sends everything up to 464 to 448). The first overflowing input is above 464.
- Under 2.13.0+, +inf converts to 448 too (saturating), and nan stays nan; 2^-10 underflows to 0 in e4m3fn (smallest subnormal 2^-9).
- float8_e5m2 is the same in every run: 448 -> 448, 480 -> 512, 500 -> 512, 57344 -> 57344 (its maximum), -449 -> -448, inf -> inf, nan -> nan,
  2^-10 -> 2^-10, 1e-30 -> 0. Its overflow (65536 and above) gives `inf` in all versions, see the workload's informational op below.
- The same table row applies to float64, float16 and bfloat16 sources (the files in `docs/fp8-probe-results/` hold all 24 rows per run).
  float16 sources cannot hold anything above 65504, so 57344 is still exact there.
- A10G == L4 for every torch (all 24 rows). Simulated == real for every simulated card and both tested torch versions. The simulated
  runs used the nightly's simulator build (the `sim-build` artifact of run 37787475661, pantheonsim built by the nightly) on the L4 host.

Hosts: one on-demand g5.xlarge (A10G, driver 595.x DLAMI Ubuntu 22.04) and one on-demand g6.xlarge (L4, DLAMI Ubuntu 24.04), us-east-1, Python 3.12
from uv; torch wheels from PyPI (`+cu130` for 2.13.0 and 2.14.1; PyPI's 2.10.0 is the `+cu128` build; the `+cu130` 2.10.0 from
download.pytorch.org was probed on the L4 and the simulator). Both instances were terminated afterwards.

## What the workload does about it

`lib-attention-precision` (workload change in this commit):

- `cast_roundtrip_fp8_e4m3fn` sees only in-range inputs: `CASTV` capped at +-448. Its numerics are identical on every torch and card tested
  (checksum equal on CPU and CUDA, torch 2.10.0 and 2.13.0, A10G and L4), so its recorded reference did not change. `cast_roundtrip_fp8_e5m2`
  keeps the full `CASTV`: 57344 is e5m2's maximum, so nothing overflows.
- Two new `kind="info"` ops, `cast_overflow_class_fp8_e4m3fn` and `cast_overflow_class_fp8_e5m2`, cast inputs above the range (e4m3fn: 480, 500,
  57344, -480, -57344, 1e30, +-inf, nan; e5m2: 61440, 65536, 1e6, -1e6, 1e30, +-inf, nan) and record the class of each result:
  `saturates` (+-max), `nan`, `inf` or `other`, on the device and on the CPU. The class goes to the run record's `info` field and to stderr;
  `output` holds the constant string `"informational"`, so the op cannot change a verdict. See `docs/lib-coverage.md`.
- The reference gained the two `"informational"` keys; the e4m3fn key is unchanged (the old reference's 449 and 57344 became 448 anyway).
  The new keys come from the torch 2.13.0 CPU run of the changed workload; the A10G run with torch 2.13.0 and 2.10.0, and the L4 run
  with 2.10.0+cu130, 2.13.0 and 2.14.1, produced the same `output` for these three keys. No other key and no bound was touched.

Observed classes (`info` of the run): torch 2.10.0, A10G and L4, CUDA and CPU: e4m3fn all `nan` (nan input: `nan`); e5m2 all `inf`
(nan input `nan`). Torch 2.13.0 and newer: e4m3fn all `saturates` (nan input `nan`); e5m2 all `inf`.

## Simulator versus hardware

No disagreement found: the simulated sm_75, sm_80 and sm_86 GPUs give the same fp8 conversion results as the A10G and L4 under torch 2.10.0+cu130
and 2.14.1+cu130. Not tested: Hopper/Blackwell (no real card here to compare with), and torch 2.13.0 in the simulator (expected equal).
Nothing to report to pantheonsim.

## Reproduce

```bash
python3 tools/fp8_probe.py > probe.json                 # on any machine with torch; CUDA rows only when a GPU is visible
env -u LD_LIBRARY_PATH python3 tools/fp8_probe.py        # DLAMI: see docs/first-gpu-run-a10g.md
vgpu run --gpu nvidia/a100 --preload python3 tools/fp8_probe.py   # simulated
```
