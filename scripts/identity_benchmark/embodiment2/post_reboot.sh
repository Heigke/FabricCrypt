#!/bin/bash
# embodiment2 post-reboot script.
# Wait cool, re-collect envelope, compute G4 (both v1 and v2 axes), then
# launch fresh Claude tmux for continuation.
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
LOG=$ROOT/logs/embodiment2/post_reboot.log
STATE=$ROOT/state/embodiment2_state.json
SESSION_FILE=$ROOT/state/embodiment_claude_session.txt
PY=$ROOT/venv/bin/python
TMUX=/usr/bin/tmux
CLAUDE_BIN=/home/ikaros/.local/bin/claude
RESUME_TMUX=claude-embodiment-resume

export TMUX_TMPDIR=/tmp
unset TMPDIR
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONUNBUFFERED=1

mkdir -p $(dirname "$LOG") $(dirname "$STATE")
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== embodiment2 post_reboot START ==="
log "uptime=$(cut -d. -f1 /proc/uptime)s apu=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)mC"

# Refuse to run unless fresh boot (uptime < 1800s) AND armed
UPT=$(cut -d. -f1 /proc/uptime)
if [ "$UPT" -gt 1800 ]; then
    log "uptime=${UPT}s > 1800s, not a fresh boot — exiting"
    exit 0
fi
ARMED=$($PY -c "import json,pathlib; p=pathlib.Path('$STATE'); print(json.loads(p.read_text()).get('A4_armed',False) if p.exists() else False)" 2>/dev/null)
if [ "$ARMED" != "True" ]; then
    log "A4 not armed (armed=$ARMED) — exiting"
    exit 0
fi

# Wait thermal equilibrium
log "waiting for APU ≤45C..."
for i in $(seq 1 60); do
    T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 99000)
    if [ "$T" -le 45000 ]; then log "APU cool: ${T}mC (i=$i)"; break; fi
    sleep 5
done

# Wait network
for i in $(seq 1 30); do
    if ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then log "network up i=$i"; break; fi
    sleep 2
done

# Re-collect post-reboot envelope (OVERWRITE any stale A4_post.json)
log "collecting post-reboot envelope..."
$PY $ROOT/scripts/identity_benchmark/embodiment/envelope_fast.py \
    --out $ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a/A4_post.json \
    --label A4_post --quick 2>&1 | tee -a "$LOG"

# Compute G4 for both v1 and v2 structures
log "computing G4..."
$PY $ROOT/scripts/identity_benchmark/embodiment2/a4_g4_compute.py 2>&1 | tee -a "$LOG"

# Disarm
$PY -c "
import json,pathlib,time
p=pathlib.Path('$STATE')
d=json.loads(p.read_text()) if p.exists() else {}
d['A4_armed']=False
d['A4_post_reboot_done_at']=time.strftime('%Y-%m-%dT%H:%M:%S')
p.write_text(json.dumps(d,indent=2,default=str))
"
log "disarmed"

# Launch Claude tmux for continuation
SESSION_UUID="83353dd8-59e1-4c51-a16d-0c4cceb1d1b4"
[ -f "$SESSION_FILE" ] && SESSION_UUID=$(cat "$SESSION_FILE")

if $TMUX -L $RESUME_TMUX has-session -t $RESUME_TMUX 2>/dev/null; then
    log "tmux $RESUME_TMUX already exists"
else
    log "launching tmux $RESUME_TMUX"
    CONT_PROMPT="Autoresume embodiment2 work after ikaros reboot for G4 closure. Read state/embodiment2_state.json and results/IDENTITY_BENCHMARK_2026-05-30/embodiment2/A4_reboot_result.json . A4 reboot test has been run already — that file contains G1, G4, G4_ratio, PASS/FAIL for both v1 (3-axis) and v2 (5-axis) structures. Update results/IDENTITY_BENCHMARK_2026-05-30/embodiment2/DELIVERABLE.md with the actual G4 values (currently placeholders). All other phases (D1-D3, E1-E2, F1, F2, oracle O106, lit scan) are already complete and documented. Use /remote-control to give the user a URL. Then stop."
    $TMUX -L $RESUME_TMUX new-session -d -s $RESUME_TMUX -c "$ROOT" \
        "$CLAUDE_BIN --dangerously-skip-permissions '$CONT_PROMPT'"
    sleep 15
    log "sending /remote-control"
    $TMUX -L $RESUME_TMUX send-keys -t $RESUME_TMUX "/remote-control" Enter
fi

log "=== embodiment2 post_reboot DONE ==="
