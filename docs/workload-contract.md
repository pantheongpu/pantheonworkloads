# Workload contract

`workloads/<name>/run.sh` is executed by `bin/pw run` from the repository root with:

| Variable | Value |
| --- | --- |
| `PW_TARGET` | `cpu`, `gpu` or `sim:<vendor>/<profile>` |
| `PW_WORKLOAD_DIR` | the workload's directory |
| `PW_OUT` | a scratch directory the workload may write to |
| `PW_SIM_PROFILE` | the `<vendor>/<profile>` part, for `sim:` targets |
| `PANTHEONSIM_DIR` | pantheonsim checkout, for `sim:` targets |

Exit status: `0` the workload ran, `77` it cannot run here (missing tool or model: the runner
records SKIP), anything else is a failure.

The **last line of stdout** must be one JSON object:

```json
{"output": "<what is compared to reference.json>", "detail": "<short human text>", "metrics": {}}
```

`metrics` is for benchmark numbers (real targets only; the runner drops it, and fails the run,
for `sim:` targets). `output` is compared with `reference.json` according to the manifest's `compare`.

`bin/pw record <workload> --target T` runs the workload once and writes the result as
`reference.json`. Record references only on a target you trust (a real GPU or the CPU backend),
and say which in the commit message.
