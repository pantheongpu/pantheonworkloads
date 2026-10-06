# Sourced by the llamacpp-bench-* run.sh after lc_setup and lc_fetch_model: runs llama-bench
# on $LC_MODEL and prints the contract's JSON line (metrics from bench_parse.py).
lc_bench() {
  if [[ "$PW_TARGET" == sim:* ]]; then
    lc_skip "benchmarks are for real targets only; simulated GPU time is not GPU time"
  fi
  local pp="${PW_BENCH_PP:-512}" tg="${PW_BENCH_TG:-128}"
  # shellcheck disable=SC2086
  env "${LC_ENV[@]}" "$LC_BIN/llama-bench" -m "$LC_MODEL" -p "$pp" -n "$tg" -r "${PW_BENCH_REPS:-3}" \
    -ngl "$LC_NGL" -t "${PW_THREADS:-4}" -o json ${PW_BENCH_EXTRA:-} \
    >"$PW_OUT/bench.json" 2>"$PW_OUT/bench.err" \
    || { echo "llama-bench failed" >&2; tail -20 "$PW_OUT/bench.err" >&2; exit 1; }
  PW_BENCHJSON="$PW_OUT/bench.json" PW_DETAIL="llama.cpp $LLAMACPP_TAG $LC_BACKEND, $(basename "$LC_MODEL"), pp$pp tg$tg, ngl $LC_NGL" \
  python3 -I - "$LC_ROOT" <<'PY'
import json, os, sys
sys.path.insert(0, sys.argv[1])
import bench_parse
m = bench_parse.parse(open(os.environ["PW_BENCHJSON"], encoding="utf-8").read())
print(json.dumps({"output": "llama-bench ok", "detail": os.environ["PW_DETAIL"], "metrics": m}))
PY
}
