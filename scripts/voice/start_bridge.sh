#!/usr/bin/env bash
# Start the voice bridge: cloudflared tunnel + voice_server + voice_bridge poller.
#
# Usage:
#   bash scripts/voice/start_bridge.sh
#
# After it prints the trycloudflare URL, register the URL in the Vonage
# dashboard:
#   Voice → Application → Capabilities
#     Answer URL: <URL>/answer
#     Event URL:  <URL>/event

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VOICE_DIR="$REPO_ROOT/scripts/voice"
VENV_PY="$REPO_ROOT/venv/bin/python"
LOG_DIR="$REPO_ROOT/scripts/voice/logs"
mkdir -p "$LOG_DIR"

CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$HOME/.local/bin/cloudflared}"
PORT="${VOICE_SERVER_PORT:-5050}"

# 1. ensure cloudflared exists
if [ ! -x "$CLOUDFLARED_BIN" ]; then
    echo "[start_bridge] installing cloudflared to $CLOUDFLARED_BIN"
    mkdir -p "$(dirname "$CLOUDFLARED_BIN")"
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -O "$CLOUDFLARED_BIN"
    chmod +x "$CLOUDFLARED_BIN"
fi
echo "[start_bridge] cloudflared: $($CLOUDFLARED_BIN --version 2>&1 | head -1)"

# 2. ensure private key is 600
chmod 600 "$REPO_ROOT/scripts/private-2.key" || true

# 3. start cloudflared tunnel
TUNNEL_LOG="$LOG_DIR/cloudflared.log"
: > "$TUNNEL_LOG"
echo "[start_bridge] starting cloudflared tunnel on port $PORT"
nohup "$CLOUDFLARED_BIN" tunnel --url "http://localhost:$PORT" \
    > "$TUNNEL_LOG" 2>&1 &
CLOUDFLARED_PID=$!
echo "$CLOUDFLARED_PID" > "$LOG_DIR/cloudflared.pid"

# 4. wait for URL (up to 60s)
URL=""
for i in $(seq 1 60); do
    URL=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)
    if [ -n "$URL" ]; then break; fi
    sleep 1
done
if [ -z "$URL" ]; then
    echo "[start_bridge] ERROR: cloudflared did not produce URL in 60s. See $TUNNEL_LOG"
    exit 1
fi
echo "[start_bridge] tunnel URL: $URL"
export WEBHOOK_BASE_URL="$URL"
echo "$URL" > "$LOG_DIR/webhook_base_url.txt"

# 5. start voice server
SERVER_LOG="$LOG_DIR/voice_server.log"
echo "[start_bridge] starting voice_server on :$PORT"
cd "$VOICE_DIR"
WEBHOOK_BASE_URL="$URL" VOICE_SERVER_PORT="$PORT" \
    nohup "$VENV_PY" voice_server.py > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$LOG_DIR/voice_server.pid"

# 6. wait for server health
for i in $(seq 1 30); do
    if curl -fsS "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[start_bridge] voice_server healthy"
        break
    fi
    sleep 1
done

# 7. start poller
BRIDGE_LOG="$LOG_DIR/voice_bridge.log"
echo "[start_bridge] starting voice_bridge poller"
WEBHOOK_BASE_URL="$URL" \
    nohup "$VENV_PY" voice_bridge.py > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
echo "$BRIDGE_PID" > "$LOG_DIR/voice_bridge.pid"

echo
echo "================================================================"
echo " Voice bridge running."
echo "   tunnel URL: $URL"
echo "   PIDs:       cloudflared=$CLOUDFLARED_PID server=$SERVER_PID bridge=$BRIDGE_PID"
echo "   logs:       $LOG_DIR"
echo
echo " Register in Vonage dashboard:"
echo "   Voice → Application → Capabilities"
echo "     Answer URL: $URL/answer   (GET)"
echo "     Event URL:  $URL/event    (POST)"
echo
echo " Test a call:    $VENV_PY $VOICE_DIR/voice_bridge.py --test"
echo " Stop:           bash $VOICE_DIR/stop_bridge.sh"
echo "================================================================"
