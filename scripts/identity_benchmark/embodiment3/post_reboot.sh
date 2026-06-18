#!/bin/bash
# Embodiment3 post-reboot: wait equilibrium, re-collect robust signature,
# run V3 Phase C with G4 closed, launch Claude in tmux for autoresume.
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
LOG=$ROOT/logs/embodiment3/post_reboot.log
STATE=$ROOT/state/embodiment3_state.json
PY=$ROOT/venv/bin/python
TMUX=/usr/bin/tmux
CLAUDE_BIN=/home/ikaros/.local/bin/claude
RESUME_TMUX=claude-embodiment3
SESSION_UUID="83353dd8-59e1-4c51-a16d-0c4cceb1d1b4"

export TMUX_TMPDIR=/tmp
unset TMPDIR

mkdir -p $(dirname "$LOG")
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== embodiment3 post_reboot START ==="
log "uptime=$(cut -d. -f1 /proc/uptime)s apu=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)mC"

# 1) Refuse if not a fresh boot (uptime < 1800s)
UPT=$(cut -d. -f1 /proc/uptime)
if [ "$UPT" -gt 1800 ]; then
    log "uptime=${UPT}s > 1800s, not a fresh boot — exiting"
    exit 0
fi

# 2) Check armed flag
ARMED=$($PY -c "
import json, pathlib
p=pathlib.Path('$STATE')
print(json.loads(p.read_text()).get('armed', False) if p.exists() else False)
" 2>/dev/null)
if [ "$ARMED" != "True" ]; then
    log "not armed (armed=$ARMED) — exiting"
    exit 0
fi

# 3) Wait APU cool ≤45C (up to 5 min)
log "waiting for APU ≤45C..."
for i in $(seq 1 60); do
    T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 99000)
    if [ "$T" -le 45000 ]; then log "APU cool: ${T}mC (i=$i)"; break; fi
    sleep 5
done

# 4) Wait network
log "waiting for network..."
for i in $(seq 1 30); do
    if ping -c1 -W2 1.1.1.1 >/dev/null 2>&1; then log "network up i=$i"; break; fi
    sleep 2
done

# 5) Collect post-reboot robust signature
log "collecting post-reboot robust signature (N=60)..."
$PY $ROOT/scripts/identity_benchmark/embodiment3/robust_signature.py \
    --N 60 --interval 0.5 --label post_reboot \
    --out $ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/signatures/ikaros_post_reboot.json \
    2>&1 | tee -a "$LOG"

# 6) Compute bit-distance pre vs post
log "computing reboot bit-distance..."
$PY -c "
import json, sys
sys.path.insert(0, '$ROOT/scripts/identity_benchmark/embodiment3')
from robust_signature import load_signature, quantize_robust, quantized_to_bitstring, bit_distance
sig_pre = load_signature('$ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/signatures/ikaros_v2a_t0.json')
sig_post = load_signature('$ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/signatures/ikaros_post_reboot.json')
qa = quantize_robust(sig_pre); qb = quantize_robust(sig_post)
d, n = bit_distance(qa, qb)
print(f'BIT_DRIFT_REBOOT={d}/{n} ({100*d/max(1,n):.2f}%)')
open('$ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/reboot_drift.json','w').write(json.dumps({'hamming': d, 'total_bits': n, 'pct': 100*d/max(1,n)}, indent=2))
" 2>&1 | tee -a "$LOG"

# 7) Run V3 Phase C with G4 active
log "running V3 Phase C with G4..."
$PY $ROOT/scripts/identity_benchmark/embodiment3/v3_phase_c.py --seeds 10 2>&1 | tee -a "$LOG"

# 8) Disarm
$PY -c "
import json, pathlib, time
p=pathlib.Path('$STATE')
d=json.loads(p.read_text()) if p.exists() else {}
d['armed']=False
d['post_reboot_done_at']=time.strftime('%Y-%m-%dT%H:%M:%S')
p.write_text(json.dumps(d, indent=2, default=str))
"

# 9) Launch Claude in tmux for continuation
if $TMUX -L $RESUME_TMUX has-session -t $RESUME_TMUX 2>/dev/null; then
    log "tmux $RESUME_TMUX already exists"
else
    log "launching tmux $RESUME_TMUX (fresh claude session)"
    CONT_PROMPT="embodiment3 post-reboot autoresume. Read results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/reboot_drift.json and results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/phase_c/v3_phase_c_result.json for the G4 verdict. If G3 and G4 pass, run scripts/identity_benchmark/embodiment3/v4_advantage.py . Then write the final deliverable to results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/DELIVERABLE.md ."
    $TMUX -L $RESUME_TMUX new-session -d -s $RESUME_TMUX -c "$ROOT" \
        "$CLAUDE_BIN --dangerously-skip-permissions '$CONT_PROMPT'"
    sleep 15
    log "sending /remote-control"
    $TMUX -L $RESUME_TMUX send-keys -t $RESUME_TMUX "/remote-control" Enter
fi

log "=== embodiment3 post_reboot DONE ==="
