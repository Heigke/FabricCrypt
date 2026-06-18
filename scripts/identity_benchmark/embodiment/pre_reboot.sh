#!/bin/bash
# Phase A4 pre-reboot: snapshot state, verify autoresume, then reboot.
# Idempotent: refuses to reboot unless autoresume hooks are verified.
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
SESSION_UUID="83353dd8-59e1-4c51-a16d-0c4cceb1d1b4"
STATE=$ROOT/state/embodiment_state.json
LOG=$ROOT/logs/embodiment/pre_reboot.log
RESUME_SH=$ROOT/scripts/identity_benchmark/embodiment/post_reboot.sh

mkdir -p $(dirname "$LOG") $(dirname "$STATE")
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== pre_reboot.sh START ==="
log "uptime=$(cut -d. -f1 /proc/uptime)s apu=$(cat /sys/class/thermal/thermal_zone0/temp)mC"

# 1) verify post_reboot.sh exists + executable
if [ ! -x "$RESUME_SH" ]; then
    log "FATAL: $RESUME_SH not executable — aborting reboot"
    exit 1
fi

# 2) verify cron @reboot hook exists for the embodiment resume
if ! crontab -l 2>/dev/null | grep -q "embodiment/post_reboot.sh"; then
    log "FATAL: no @reboot cron line for embodiment/post_reboot.sh — aborting"
    exit 1
fi

# 3) verify session uuid file
echo "$SESSION_UUID" > $ROOT/state/embodiment_claude_session.txt
log "session uuid pinned: $SESSION_UUID"

# 4) annotate state with pre-reboot snapshot
PRE_VEC=$(ls $ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a/A1_ikaros.json 2>/dev/null || echo MISSING)
log "pre-reboot ikaros baseline: $PRE_VEC"
$ROOT/venv/bin/python -c "
import json,time,pathlib
p=pathlib.Path('$STATE')
d=json.loads(p.read_text()) if p.exists() else {}
d.setdefault('phase_a',{}).setdefault('A4',{})
d['phase_a']['A4']['pre_reboot_ts']=time.strftime('%Y-%m-%dT%H:%M:%S')
d['phase_a']['A4']['pre_reboot_uptime_s']=float(open('/proc/uptime').read().split()[0])
d['phase_a']['A4']['armed']=True
p.write_text(json.dumps(d,indent=2,default=str))
"
log "state annotated for A4"

# 5) capture pre-reboot signature as canonical 'pre' to compare against
$ROOT/venv/bin/python $ROOT/scripts/identity_benchmark/embodiment/envelope_fast.py \
    --out $ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment/phase_a/A4_pre.json \
    --label A4_pre 2>&1 | tee -a "$LOG"

# 6) actually trigger reboot
log "rebooting NOW via sudo reboot..."
sudo -n /sbin/reboot 2>&1 | tee -a "$LOG" || { log "FATAL: sudo reboot failed (no NOPASSWD?)"; exit 2; }
