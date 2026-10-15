#!/usr/bin/env bash
# Keep tools/rig/stage_weights.py running on a shared box without root or systemd system units (docs/rigs.md,
# "Staging from a shared box"). Single instance (flock), restarts on failure with backoff, exits when the plan is complete.
#
#   stage_supervisor.sh run          [stage_weights args]   foreground loop (what cron runs)
#   stage_supervisor.sh start        [stage_weights args]   same, detached; prints the pid
#   stage_supervisor.sh stop                                SIGTERM the supervisor (partial files are kept)
#   stage_supervisor.sh status                              stage_weights.py --status plus whether the supervisor runs
#   stage_supervisor.sh install-cron [stage_weights args]   add an @reboot crontab entry for `run` with these args
#   stage_supervisor.sh remove-cron                         remove it again (only the line this script added)
#
# Example args: --tier 2b --mirror s3://BUCKET/pw-weights   (--scratch, --state-dir and --log-file default to PW_STAGE_HOME).
# Environment: PW_STAGE_HOME (default ~/.pw-stage: logs/, status.json, scratch/, credentials), PW_STAGE_NO_SYSTEMD=1 (skip
# the systemd-run limits), PW_STAGE_BACKOFF_S (fixed restart delay, for tests), PW_STAGE_PY (stage_weights.py path).
# If PW_STAGE_HOME/credentials exists it is used as the AWS credentials file with profile pw-stage (unless already set).
# Exit codes of `run`: 0 plan complete, 2 usage/plan error (not retried), 4 another stage_weights holds the run lock,
# 9 another supervisor is running, 143 stopped.
set -u
HOME_DIR="${PW_STAGE_HOME:-$HOME/.pw-stage}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PW_STAGE_PY:-$HERE/stage_weights.py}"
LOGS="$HOME_DIR/logs"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
MARK="# pw-stage-supervisor"
mkdir -p "$LOGS"
export PATH="$PATH:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
CRONTAB="${PW_CRONTAB:-crontab}"

rotate() {   # rotate FILE MAXBYTES: keep 5 old copies
    local f="$1" max="$2" i
    [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt "$max" ] || return 0
    for i in 4 3 2 1; do [ -f "$f.$i" ] && mv -f "$f.$i" "$f.$((i + 1))"; done
    mv -f "$f" "$f.1"
}
slog() { rotate "$LOGS/supervisor.log" 5000000; printf '%s supervisor[%s]: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$$" "$*" >>"$LOGS/supervisor.log"; }

cmd="${1:-}"; shift || true

case "$cmd" in
status)
    if flock -n "$HOME_DIR/supervisor.lock" true 2>/dev/null; then echo "supervisor: not running"
    else echo "supervisor: running (pid $(cat "$HOME_DIR/supervisor.pid" 2>/dev/null || echo '?'))"; fi
    exec python3 "$PY" --status --state-dir "$HOME_DIR"
    ;;
stop)
    pid="$(cat "$HOME_DIR/supervisor.pid" 2>/dev/null || true)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "supervisor: not running"; exit 0; fi
    kill -TERM "$pid"
    for _ in $(seq 1 120); do kill -0 "$pid" 2>/dev/null || { echo "stopped"; exit 0; }; sleep 1; done
    echo "supervisor $pid did not exit within 120 s" >&2; exit 1
    ;;
start)
    if ! flock -n "$HOME_DIR/supervisor.lock" true 2>/dev/null; then echo "already running" >&2; exit 9; fi
    setsid nohup "$SELF" run "$@" >>"$LOGS/supervisor.out" 2>&1 </dev/null &
    echo "started supervisor pid $! (logs: $LOGS, status: $SELF status)"
    exit 0
    ;;
install-cron|remove-cron)
    cur="$("$CRONTAB" -l 2>/dev/null | grep -vF "$MARK" || true)"
    if [ "$cmd" = install-cron ]; then
        line="@reboot PW_STAGE_HOME=$(printf %q "$HOME_DIR") $(printf %q "$SELF") run"
        for a in "$@"; do line="$line $(printf %q "$a")"; done
        printf '%s\n%s\n' "$cur" "$line >>$(printf %q "$LOGS")/cron.log 2>&1 $MARK" | sed '/^$/d' | "$CRONTAB" -
        echo "installed: $line"
        echo "remove with: $SELF remove-cron   (or: crontab -e and delete the line ending in '$MARK')"
    else
        printf '%s\n' "$cur" | sed '/^$/d' | "$CRONTAB" -
        echo "removed the pw-stage @reboot entry"
    fi
    exit 0
    ;;
run) ;;
*) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 64 ;;
esac

# ---- run ----
exec 9>"$HOME_DIR/supervisor.lock"
if ! flock -n 9; then echo "supervisor already running" >&2; exit 9; fi
echo $$ >"$HOME_DIR/supervisor.pid"

if [ -f "$HOME_DIR/credentials" ] && [ -z "${AWS_SHARED_CREDENTIALS_FILE:-}" ]; then
    export AWS_SHARED_CREDENTIALS_FILE="$HOME_DIR/credentials"
    export AWS_PROFILE="${AWS_PROFILE:-pw-stage}"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
fi
if ! command -v aws >/dev/null 2>&1 && [ -x "$HOME/.local/bin/aws" ]; then export PW_AWS="$HOME/.local/bin/aws"; fi

args=("$@")
has() { local x; for x in "${args[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
has --scratch   || args+=(--scratch "$HOME_DIR/scratch")
has --state-dir || args+=(--state-dir "$HOME_DIR")
has --log-file  || args+=(--log-file "$LOGS/stage.log")

wrap=()
if [ -n "${PW_STAGE_NO_SYSTEMD:-}" ]; then
    why="disabled by PW_STAGE_NO_SYSTEMD"
else
    [ -n "${XDG_RUNTIME_DIR:-}" ] || { [ -d "/run/user/$(id -u)" ] && export XDG_RUNTIME_DIR="/run/user/$(id -u)"; }
    lim=(-p CPUQuota=400% -p MemoryMax=6G -p MemorySwapMax=0)
    if why="$(systemd-run --user --scope -q "${lim[@]}" true 2>&1)"; then
        wrap=(systemd-run --user --scope -q "${lim[@]}")
        why=""
    fi
fi
if [ ${#wrap[@]} -gt 0 ]; then slog "limits: systemd-run --user --scope CPUQuota=400% MemoryMax=6G MemorySwapMax=0, plus nice/ionice"
else slog "limits: nice/ionice only (systemd-run scope unavailable: ${why:-unknown})"; fi

child=""; stopping=""
trap 'stopping=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM INT
delay="${PW_STAGE_BACKOFF_S:-30}"
while :; do
    rotate "$LOGS/stage.stderr.log" 5000000
    slog "starting: python3 $(basename "$PY") ${args[*]}"
    t0=$(date +%s)
    "${wrap[@]}" python3 "$PY" "${args[@]}" >/dev/null 2>>"$LOGS/stage.stderr.log" &
    child=$!
    wait "$child"; rc=$?
    [ -n "$stopping" ] && wait "$child" 2>/dev/null
    child=""
    ran=$(( $(date +%s) - t0 ))
    if [ -n "$stopping" ]; then slog "stopped by signal"; exit 143; fi
    case "$rc" in
        0) slog "plan complete (exit 0 after ${ran}s); nothing left to stage"; exit 0 ;;
        2|4) slog "exit $rc is not retried (usage/plan error or run lock held); see $LOGS/stage.stderr.log"; exit "$rc" ;;
    esac
    if [ -z "${PW_STAGE_BACKOFF_S:-}" ]; then
        if [ "$ran" -gt 600 ]; then delay=30; else delay=$(( delay * 2 > 1800 ? 1800 : delay * 2 )); fi
    fi
    slog "exit $rc after ${ran}s; restarting in ${delay}s"
    sleep "$delay" & child=$!; wait "$child"; child=""
    [ -n "$stopping" ] && { slog "stopped by signal"; exit 143; }
done
