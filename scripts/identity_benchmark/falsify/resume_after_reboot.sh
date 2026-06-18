#!/bin/bash
# Post-reboot resume script — auto-runs the falsification pipeline and
# re-launches the Claude session in tmux.
# Idempotent: safe to call multiple times (will not double-launch tmux session).

set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
LOG=$ROOT/logs/falsify/reboot_resume.log
STATE=$ROOT/state/falsify_state.json
SESSION_FILE=$ROOT/state/claude_session.txt
PY=$ROOT/venv/bin/python
TMUX=/usr/bin/tmux
CLAUDE_BIN=/home/ikaros/.local/bin/claude
RESUME_TMUX=claude-resume

# Force tmux socket to /tmp (not Claude's harness-provided $TMPDIR which may be gone post-reboot)
export TMUX_TMPDIR=/tmp
unset TMPDIR

mkdir -p $(dirname "$LOG") $(dirname "$STATE")

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== resume_after_reboot.sh start ==="
log "uptime: $(cut -d. -f1 /proc/uptime)s, APU=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)mC"

# 1) Wait for thermal equilibrium
log "waiting for APU <= 45C..."
for i in $(seq 1 60); do
    T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 99000)
    if [ "$T" -le 45000 ]; then
        log "APU cool: ${T}mC after ${i}*5s"
        break
    fi
    sleep 5
done

# 2) Wait for network
log "waiting for network..."
for i in $(seq 1 30); do
    if ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then
        log "network up after ${i}*2s"
        break
    fi
    sleep 2
done

# 3) Read state
if [ ! -f "$STATE" ]; then
    log "no state file at $STATE — nothing to resume"
else
    NEXT=$($PY -c "import json; print(json.load(open('$STATE')).get('next') or 'NONE')")
    log "state.next=$NEXT"
    if [ "$NEXT" != "NONE" ] && [ "$NEXT" != "None" ]; then
        log "running falsify pipeline (next-only)..."
        cd "$ROOT"
        HSA_OVERRIDE_GFX_VERSION=11.0.0 $PY scripts/identity_benchmark/falsify/run_pipeline.py --next-only \
            >> "$LOG" 2>&1 || log "pipeline returned non-zero"
    fi
fi

# 4) Launch Claude in tmux (idempotent)
SESSION_UUID=""
if [ -f "$SESSION_FILE" ]; then
    SESSION_UUID=$(cat "$SESSION_FILE")
fi
if [ -z "$SESSION_UUID" ]; then
    SESSION_UUID="83353dd8-59e1-4c51-a16d-0c4cceb1d1b4"
fi

TMUX_SOCKET="-L $RESUME_TMUX"
if $TMUX $TMUX_SOCKET has-session -t $RESUME_TMUX 2>/dev/null; then
    log "tmux session $RESUME_TMUX already exists, not re-launching"
else
    log "launching tmux ($RESUME_TMUX socket) with claude --resume $SESSION_UUID"
    $TMUX $TMUX_SOCKET new-session -d -s $RESUME_TMUX -c "$ROOT" \
        "$CLAUDE_BIN --resume $SESSION_UUID --dangerously-skip-permissions 2>&1 | tee -a $LOG"
    sleep 10
    log "sending /remote-control to tmux session"
    $TMUX $TMUX_SOCKET send-keys -t $RESUME_TMUX "/remote-control" Enter
fi

# 5) If F4 reboot still needed, schedule another reboot in 5 min
if [ -f "$ROOT/state/needs_reboot.flag" ]; then
    log "needs_reboot.flag present — scheduling reboot in 300s (use sudo)"
    # We don't auto-reboot; require human or pre-authorised sudoers
    log "TODO: human must run 'sudo reboot' or enable passwordless sudo for reboot"
fi

log "=== resume_after_reboot.sh done ==="
