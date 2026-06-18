#!/bin/bash
# B4 — dry-run autoresume for embodiment3 (no reboot).
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
TMUX=/usr/bin/tmux
CLAUDE_BIN=/home/ikaros/.local/bin/claude
DRY_TMUX=claude-embodiment3-dryrun
LOG=$ROOT/logs/embodiment3/b4_dryrun.log

export TMUX_TMPDIR=/tmp
unset TMPDIR
mkdir -p $(dirname "$LOG")
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== embodiment3 B4 DRY-RUN START ==="
[ -x "$CLAUDE_BIN" ] || { log "FATAL: $CLAUDE_BIN missing"; exit 1; }
[ -x "$TMUX" ] || { log "FATAL: tmux missing"; exit 1; }

if $TMUX -L $DRY_TMUX has-session -t $DRY_TMUX 2>/dev/null; then
    log "tearing down stale dry-run session"
    $TMUX -L $DRY_TMUX kill-session -t $DRY_TMUX 2>/dev/null || true
fi

log "launching dry-run tmux on socket $DRY_TMUX"
DRY_PROMPT="embodiment3 B4 DRY RUN — reply with AUTORESUME3_OK"
$TMUX -L $DRY_TMUX new-session -d -s $DRY_TMUX -c "$ROOT" \
    "$CLAUDE_BIN --dangerously-skip-permissions '$DRY_PROMPT'"

log "waiting 20s for claude..."
sleep 20

if $TMUX -L $DRY_TMUX has-session -t $DRY_TMUX 2>/dev/null; then
    log "PASS: tmux $DRY_TMUX alive"
else
    log "FAIL: tmux $DRY_TMUX died"
    exit 2
fi

PANE_OUT=$($TMUX -L $DRY_TMUX capture-pane -p -t $DRY_TMUX 2>/dev/null | tail -10)
log "pane tail: $PANE_OUT"

log "sending /remote-control"
$TMUX -L $DRY_TMUX send-keys -t $DRY_TMUX "/remote-control" Enter
sleep 6
PANE_OUT2=$($TMUX -L $DRY_TMUX capture-pane -p -t $DRY_TMUX 2>/dev/null | tail -15)
log "pane after /remote-control:"
log "$PANE_OUT2"

# Extract claude.ai URL if present
URL=$(echo "$PANE_OUT2" | grep -oE "https://claude\\.ai/[^ ]+" | head -1)
if [ -n "$URL" ]; then
    log "URL_FOUND: $URL"
    echo "$URL" > $ROOT/state/embodiment3_dryrun_url.txt
fi

log "cleanup: killing dry-run session"
$TMUX -L $DRY_TMUX kill-session -t $DRY_TMUX 2>/dev/null || true
log "=== embodiment3 B4 DRY-RUN DONE ==="
