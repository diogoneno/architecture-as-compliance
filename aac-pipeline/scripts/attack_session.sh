#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ $# -ge 2 ] || { echo "usage: scripts/attack_session.sh CONFIG SEED [--timeout SECONDS] [--device cpu|cuda] [--force]" >&2; exit 2; }
CONFIG=$1
SEED=$2
shift 2
TIMEOUT=""
FORCE=""
DEVICE=cpu
while [ $# -gt 0 ]; do
 case "$1" in
 --timeout) [ $# -ge 2 ] || { echo "attack_session.sh: --timeout needs a value" >&2; exit 2; }; TIMEOUT=$2; shift 2 ;;
 --device) [ $# -ge 2 ] || { echo "attack_session.sh: --device needs a value" >&2; exit 2; }; DEVICE=$2; shift 2 ;;
 --force) FORCE="--force"; shift ;;
 *) echo "attack_session.sh: unknown argument $1" >&2; exit 2 ;;
 esac
done
case "$SEED" in ''|*[!0-9]*) echo "attack_session.sh: SEED must be a non-negative integer" >&2; exit 2 ;; esac
case "$TIMEOUT" in ''|*[!0-9]*) [ -z "$TIMEOUT" ] || { echo "attack_session.sh: --timeout must be whole seconds" >&2; exit 2; } ;; esac
[ -f "$CONFIG" ] || { echo "attack_session.sh: no config $CONFIG" >&2; exit 2; }
case "$DEVICE" in
 cpu) export AAC_GPU= ;;
 cuda)
 case "${AAC_GPU:-}" in
 ""|*[!0-9]*) echo "attack_session.sh: --device cuda needs AAC_GPU=<GPU index> (got '${AAC_GPU:-}')" >&2; exit 2 ;;
 esac
 export AAC_GPU ;;
 *) echo "attack_session.sh: --device must be cpu or cuda (got '$DEVICE')" >&2; exit 2 ;;
esac
PY="$ROOT/.venv-main/bin/python"
[ -x "$PY" ] || { echo "attack_session.sh: $ROOT/.venv-main missing" >&2; exit 2; }
if [ -z "$TIMEOUT" ]; then # Session ceiling from config when not provided
 TIMEOUT=$("$PY" -c 'import sys; from aac.run.config import load_raw; t = (load_raw(sys.argv[1]).get("attack") or {}).get("session_timeout_s"); print(int(t) if t else "")' "$CONFIG") \
 || { echo "attack_session.sh: cannot read attack.session_timeout_s from $CONFIG" >&2; exit 2; }
fi
MIN_RETRY_S=${AAC_ATTACK_MIN_RETRY_S:-60}
case "$MIN_RETRY_S" in ''|*[!0-9]*) echo "attack_session.sh: AAC_ATTACK_MIN_RETRY_S must be whole seconds" >&2; exit 2 ;; esac
DEADLINE=""
[ -z "$TIMEOUT" ] || DEADLINE=$(( $(date +%s) + TIMEOUT )) # one deadline for every attempt of this session
left { echo $(( DEADLINE - $(date +%s) )); }

umask 077
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/aac-attack-XXXXXX") || exit 2
chmod 700 "$STAGE"
SOCKDIR="$STAGE/sock"
SERVER=""
RUNNER=""
cleanup_on_signal {
 [ -n "$RUNNER" ] && kill -TERM "$RUNNER" 2>/dev/null
 [ -n "$SERVER" ] && kill -TERM "$SERVER" 2>/dev/null
 echo "attack-session: interrupted; staging kept at $STAGE" >&2
 exit 130
}
trap cleanup_on_signal INT TERM

fail { # fail <code> <message>
 echo "attack-session: $2; staging kept at $STAGE" >&2
 exit "$1"
}

start_server { # start_server [--force]; returns once /v1/ping answers
 "$PY" -m aac.serve.victim_server serve --config "$CONFIG" --seed "$SEED" --socket-dir "$SOCKDIR" "$@" \
 >>"$STAGE/server.out" 2>>"$STAGE/server.err" &
 SERVER=$!
 local waited=0 rc
 until "$PY" -m aac.serve.victim_server ping --socket-dir "$SOCKDIR" >/dev/null 2>&1; do
 if ! kill -0 "$SERVER" 2>/dev/null; then
 wait "$SERVER"; rc=$?
 grep -E '^(refusing|serving)' "$STAGE/server.out" | tail -3 >&2
 fail "$(( rc == 0 ? 1 : rc ))" "the server exited before answering /v1/ping (exit $rc)"
 fi
 if [ "$waited" -ge 600 ]; then
 kill -TERM "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null
 fail 1 "no /v1/ping answer within 600 s"
 fi
 if [ -n "$DEADLINE" ] && [ "$(left)" -le 0 ]; then
 kill -TERM "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null
 fail 124 "no /v1/ping answer before the session deadline (${TIMEOUT} s)"
 fi
 sleep 1
 waited=$((waited + 1))
 done
}

if [ -n "${AAC_ATTACK_RUNNER:-}" ]; then read -r -a RUNNER_CMD <<<"$AAC_ATTACK_RUNNER"
else RUNNER_CMD=(scripts/attack_env.sh .venv-attack/bin/python attacks/runner/openattack_runner.py); fi
RUNNER_CMD+=(--socket "$SOCKDIR/victim.sock" --device "$DEVICE")

START_ARGS=($FORCE)
FAULT_RETRY=0
TIMEOUT_RETRY=0
while :; do
 start_server "${START_ARGS[@]}"
 if [ -n "$TIMEOUT" ]; then
 window=$(left); [ "$window" -ge 1 ] || window=1
 timeout --signal=INT --kill-after=60 "$window" "${RUNNER_CMD[@]}" >>"$STAGE/runner.out" 2>>"$STAGE/runner.err" &
 else
 "${RUNNER_CMD[@]}" >>"$STAGE/runner.out" 2>>"$STAGE/runner.err" &
 fi
 RUNNER=$!
 wait "$RUNNER"; RUNNER_RC=$?
 RUNNER=""
 if [ "$RUNNER_RC" -eq 3 ] && [ "$FAULT_RETRY" -eq 0 ]; then why="pipeline fault"
 elif [ "$RUNNER_RC" -eq 124 ] && [ "$TIMEOUT_RETRY" -eq 0 ]; then why="timeout"
 else break; fi
 if [ -n "$DEADLINE" ] && [ "$(left)" -lt "$MIN_RETRY_S" ]; then
 rest=$(left); [ "$rest" -ge 0 ] || rest=0
 echo "attack-session: $why (runner exit $RUNNER_RC); no retry: ${rest} s left before the session deadline (minimum ${MIN_RETRY_S} s)"
 break
 fi
 if [ "$why" = "pipeline fault" ]; then FAULT_RETRY=1; else TIMEOUT_RETRY=1; fi
 echo "attack-session: $why (runner exit $RUNNER_RC); one retry: server restart, resume from the finished items"
 kill -TERM "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null
 SERVER=""
 START_ARGS=($FORCE --resume) # Resume from finished items
done
ITEMS=$(grep -c '^item=' "$STAGE/runner.out" 2>/dev/null)
echo "attack-session: ${ITEMS:-0} items attacked"

if [ "$RUNNER_RC" -eq 0 ] || [ "$RUNNER_RC" -eq 124 ]; then
 "$PY" -m aac.serve.victim_server shutdown --socket-dir "$SOCKDIR" >"$STAGE/shutdown.out" 2>&1
 SHUT_RC=$?
 if [ "$SHUT_RC" -ne 0 ]; then
 kill -TERM "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null
 fail "$SHUT_RC" "POST /v1/shutdown failed; the run stays unsealed and resumable"
 fi
 wait "$SERVER"; SERVER_RC=$?
 SERVER=""
 grep -E '^(served|refusing|L2:)' "$STAGE/server.out" | tail -2
 if [ "$RUNNER_RC" -ne 0 ]; then
 [ "$SERVER_RC" -eq 0 ] || fail "$RUNNER_RC" "overall timeout ${TIMEOUT} s expired; the server did not seal, the run stays unsealed (exit $SERVER_RC)"
 fail "$RUNNER_RC" "overall timeout ${TIMEOUT} s expired; the server sealed the finished items (exit $SERVER_RC)"
 fi
 [ "$SERVER_RC" -eq 0 ] || fail "$SERVER_RC" "the server did not seal the run"
 rm -rf "$STAGE"
 exit 0
fi

grep -E '^(PipelineFault|runner error|KeyboardInterrupt)' "$STAGE/runner.err" | head -1 | cut -c1-200 >&2
kill -TERM "$SERVER" 2>/dev/null
wait "$SERVER" 2>/dev/null
fail "$RUNNER_RC" "the runner failed (exit $RUNNER_RC); the server was stopped and the run stays unsealed and resumable"
