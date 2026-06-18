#!/bin/bash
# Phase A4 post-reboot: wait equilibrium, re-collect signature, compute D_reboot,
# decide GREENLIGHT/DOWNGRADE, launch Claude in tmux to continue Phase C.
# Designed to be idempotent — safe to re-run.

set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
LOG=$ROOT/logs/embodiment/post_reboot.log
STATE=$ROOT/state/embodiment_state.json
SESSION_FILE=$ROOT/state/embodiment_claude_session.txt
PY=$ROOT/venv/bin/python
TMUX=/usr/bin/tmux
CLAUDE_BIN=/home/ikaros/.local/bin/claude
RESUME_TMUX=claude-embodiment-resume

export TMUX_TMPDIR=/tmp
unset TMPDIR

mkdir -p $(dirname "$LOG") $(dirname "$STATE")
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== post_reboot.sh START ==="
log "uptime=$(cut -d. -f1 /proc/uptime)s apu=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)mC"

# 1) refuse to do anything if not actually a fresh boot (uptime < 600s)
UPT=$(cut -d. -f1 /proc/uptime)
if [ "$UPT" -gt 1800 ]; then
    log "uptime=${UPT}s > 1800s, this is not a fresh boot — exiting"
    exit 0
fi

# 2) abort if A4 not armed (avoid running on every reboot)
ARMED=$($PY -c "import json,pathlib; p=pathlib.Path('$STATE'); print(json.loads(p.read_text()).get('phase_a',{}).get('A4',{}).get('armed',False) if p.exists() else False)" 2>/dev/null)
if [ "$ARMED" != "True" ]; then
    log "A4 not armed (armed=$ARMED) — exiting"
    exit 0
fi

# 3) wait thermal equilibrium ≤45C (up to 5 min)
log "waiting for APU ≤45C..."
for i in $(seq 1 60); do
    T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 99000)
    if [ "$T" -le 45000 ]; then log "APU cool: ${T}mC (i=$i)"; break; fi
    sleep 5
done

# 4) wait network
log "waiting for network..."
for i in $(seq 1 30); do
    if ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then log "network up i=$i"; break; fi
    sleep 2
done

# 5) re-collect post-reboot envelope
log "collecting post-reboot envelope..."
$PY $ROOT/scripts/identity_benchmark/embodiment/envelope_fast.py \
    --out $ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a/A4_post.json \
    --label A4_post 2>&1 | tee -a "$LOG"

# 6) compute distances + decision
log "computing A4 distances..."
$PY $ROOT/scripts/identity_benchmark/embodiment/a4_decide.py 2>&1 | tee -a "$LOG"

# 7) disarm so we don't re-run on next reboot
$PY -c "
import json,pathlib,time
p=pathlib.Path('$STATE')
d=json.loads(p.read_text())
d['phase_a']['A4']['armed']=False
d['phase_a']['A4']['post_reboot_done_at']=time.strftime('%Y-%m-%dT%H:%M:%S')
p.write_text(json.dumps(d,indent=2,default=str))
"

# 8) launch Claude in tmux for Phase C continuation
SESSION_UUID="83353dd8-59e1-4c51-a16d-0c4cceb1d1b4"
[ -f "$SESSION_FILE" ] && SESSION_UUID=$(cat "$SESSION_FILE")

if $TMUX -L $RESUME_TMUX has-session -t $RESUME_TMUX 2>/dev/null; then
    log "tmux $RESUME_TMUX already exists"
else
    # NOTE: --resume <uuid> fails with "no deferred tool marker" when session
    # was not interrupted at a deferred-tool boundary. Use a fresh interactive
    # session with continuation prompt that points at state/embodiment_state.json.
    log "launching tmux $RESUME_TMUX with fresh claude + continuation prompt"
    CONT_PROMPT="Autoresume embodiment work after ikaros reboot. Read state/embodiment_state.json to see Phase A status, then read results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a/A4_result.json for the A4 verdict. If verdict is GREENLIGHT or MARGINAL, run: venv/bin/python scripts/identity_benchmark/embodiment/phase_c_run.py --steps C1,C2,C3,C4,C5 --seeds 10 . Then write the final deliverable summarizing all results (A1-A4, B autoresume, C1-C5 with gates) to results/IDENTITY_BENCHMARK_2026-05-30/embodiment/DELIVERABLE.md ."
    # IMPORTANT: do NOT pipe to tee — claude needs TTY or it exits in --print mode.
    $TMUX -L $RESUME_TMUX new-session -d -s $RESUME_TMUX -c "$ROOT" \
        "$CLAUDE_BIN --dangerously-skip-permissions '$CONT_PROMPT'"
    sleep 15
    log "sending /remote-control"
    $TMUX -L $RESUME_TMUX send-keys -t $RESUME_TMUX "/remote-control" Enter
fi

log "=== post_reboot.sh DONE ==="
