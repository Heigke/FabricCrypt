#!/usr/bin/env bash
# Self-restarting supervisor for telemetry_logger.py + thermal_guard.py.
# Checks every 60 s; respawns either if its pid is dead. Logs to
# /tmp/sentinel.log. Stop via `pkill -f scripts/sentinel.sh`.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
LOG="/tmp/sentinel.log"

is_alive() {
    pgrep -f "$1" >/dev/null 2>&1
}

ensure_telemetry() {
    if ! is_alive "scripts/telemetry_logger.py"; then
        echo "$(date -Iseconds) [sentinel] starting telemetry_logger" >> "$LOG"
        nohup "$PY" "$ROOT/scripts/telemetry_logger.py" > /tmp/telemetry_stdout.log 2>&1 &
        disown
    fi
}

ensure_guard() {
    if ! is_alive "scripts/thermal_guard.py --watch"; then
        echo "$(date -Iseconds) [sentinel] starting thermal_guard" >> "$LOG"
        nohup "$PY" "$ROOT/scripts/thermal_guard.py" --watch --interval 5 \
            --filter 'python.*scripts/(z[0-9]+|demo_local|nsram_)|node.*bootstrap-fork.*fileWatcher' \
            > /tmp/thermal_guard.log 2>&1 &
        disown
    fi
}

echo "$(date -Iseconds) [sentinel] startup pid=$$" >> "$LOG"
while true; do
    ensure_telemetry
    ensure_guard
    sleep 60
done
