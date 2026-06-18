#!/usr/bin/env bash
# Clean shutdown of voice bridge (cloudflared, voice_server, voice_bridge).
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/scripts/voice/logs"

for name in voice_bridge voice_server cloudflared; do
    pidfile="$LOG_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile" 2>/dev/null || true)
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[stop_bridge] kill $name pid=$pid"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done

# fallback: any stragglers by name
pkill -f "voice_bridge.py" 2>/dev/null || true
pkill -f "voice_server.py" 2>/dev/null || true
pkill -f "cloudflared tunnel --url http://localhost:" 2>/dev/null || true

echo "[stop_bridge] done"
