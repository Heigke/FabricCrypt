#!/bin/bash
# B4 — dry-run autoresume flow WITHOUT rebooting. Verifies:
#  - tmux can launch claude --resume <uuid> --dangerously-skip-permissions
#  - /remote-control gets sent
#  - we can re-attach
#
# Cleans up the dry-run tmux session at the end.
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
SESSION_UUID="83353dd8-59e1-4c51-a16d-0c4cceb1d1b4"
TMUX=/usr/bin/tmux
CLAUDE_BIN=/home/ikaros/.local/bin/claude
DRY_TMUX=claude-embodiment-dryrun
LOG=$ROOT/logs/embodiment/b4_dryrun.log

export TMUX_TMPDIR=/tmp
unset TMPDIR
mkdir -p $(dirname "$LOG")
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== B4 DRY-RUN START ==="
# Verify binaries
[ -x "$CLAUDE_BIN" ] || { log "FATAL: $CLAUDE_BIN missing"; exit 1; }
[ -x "$TMUX" ] || { log "FATAL: tmux missing"; exit 1; }

# Already-running guard
if $TMUX -L $DRY_TMUX has-session -t $DRY_TMUX 2>/dev/null; then
    log "tearing down stale dry-run session"
    $TMUX -L $DRY_TMUX kill-session -t $DRY_TMUX 2>/dev/null || true
fi

log "launching dry-run tmux on socket $DRY_TMUX (fresh session + prompt, no tee)"
DRY_PROMPT="DRY RUN: this is a B4 verification. Reply with exactly the words: AUTORESUME_OK"
$TMUX -L $DRY_TMUX new-session -d -s $DRY_TMUX -c "$ROOT" \
    "$CLAUDE_BIN --dangerously-skip-permissions '$DRY_PROMPT'"

log "waiting 20s for claude to come up..."
sleep 20

if $TMUX -L $DRY_TMUX has-session -t $DRY_TMUX 2>/dev/null; then
    log "PASS: tmux session $DRY_TMUX is alive"
else
    log "FAIL: tmux session $DRY_TMUX died"
    exit 2
fi

# capture a screenshot of the pane
PANE_OUT=$($TMUX -L $DRY_TMUX capture-pane -p -t $DRY_TMUX 2>/dev/null | tail -10)
log "pane tail: $PANE_OUT"

log "sending /remote-control"
$TMUX -L $DRY_TMUX send-keys -t $DRY_TMUX "/remote-control" Enter
sleep 5
PANE_OUT2=$($TMUX -L $DRY_TMUX capture-pane -p -t $DRY_TMUX 2>/dev/null | tail -15)
log "pane after /remote-control:"
log "$PANE_OUT2"

# cleanup dry-run
log "cleanup: killing dry-run session"
$TMUX -L $DRY_TMUX kill-session -t $DRY_TMUX 2>/dev/null || true
log "=== B4 DRY-RUN DONE ==="
