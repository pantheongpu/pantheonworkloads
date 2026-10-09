#!/usr/bin/env bash
# The selftest workload. Environment knobs let tests/test_runner.py make it
# misbehave on purpose: PW_SELFTEST_OUTPUT, PW_SELFTEST_SKIP, PW_SELFTEST_FAIL,
# PW_SELFTEST_SLEEP, PW_SELFTEST_METRICS, PW_SELFTEST_BADJSON, PW_SELFTEST_VERSIONS (a JSON object, reported as "versions").
set -eu
[[ -z "${PW_SELFTEST_SKIP:-}" ]] || { echo "SKIP: asked to skip"; exit 77; }
[[ -z "${PW_SELFTEST_FAIL:-}" ]] || { echo "boom" >&2; exit 3; }
[[ -z "${PW_SELFTEST_SLEEP:-}" ]] || sleep "$PW_SELFTEST_SLEEP"
[[ -z "${PW_SELFTEST_BADJSON:-}" ]] || { echo "not json"; exit 0; }
metrics='{}'
[[ -z "${PW_SELFTEST_METRICS:-}" ]] || metrics='{"tokens_per_s": 1.0}'
versions=''
[[ -z "${PW_SELFTEST_VERSIONS:-}" ]] || versions=", \"versions\": $PW_SELFTEST_VERSIONS"
printf '{"output": "%s", "detail": "target %s", "metrics": %s%s}\n' \
  "${PW_SELFTEST_OUTPUT:-pw-selftest-ok}" "$PW_TARGET" "$metrics" "$versions"
