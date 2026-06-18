#!/bin/bash
# External thermal-throttle wrapper for Phase 21b training on daedalus.
# Runs OUTSIDE the training process and enforces SIGSTOP/SIGCONT based on APU temp.
#
# Strategy:
#  - Every TICK seconds reads /sys/class/thermal/thermal_zone0/temp
#  - If APU >= ABORT (75): SIGKILL all tracked PIDs and exit
#  - If APU >= PAUSE (65): SIGSTOP all tracked PIDs
#  - If APU <= RESUME (50) AND all PIDs currently in T-state: SIGCONT
#  - Otherwise no-op
#
# PID discovery is DYNAMIC: it rescans for train.py / generate.py / run_phase21b.sh
# children on every tick, so newly spawned sessions are picked up automatically.
#
# Excluded: s9_jacobian_dynamics.py (old experiment, must stay STOPPED) and any
# srgeo / whisper-cli / llama-server processes.

ABORT=80      # °C — kill switch (was 72; spurious aborts on transient spikes killed user's gen)
PAUSE=60      # °C — SIGSTOP
RESUME=50     # °C — SIGCONT (only if all are currently stopped)
TICK=2        # seconds between checks
ABORT_CONSECUTIVE=3  # require N consecutive ticks at >=ABORT before SIGKILL (avoid transient spikes)
LOG=/tmp/thermal_wrap.log

# Patterns of processes to govern (match against full cmdline)
INCLUDE_REGEX='(embodiment21b/(train|generate)\.py|run_phase21b\.sh)'
# Patterns to NEVER touch
EXCLUDE_REGEX='(s9_jacobian_dynamics|srgeo|whisper-cli|llama-server)'

get_temp() { echo $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )); }

discover_pids() {
    # Returns space-separated PID list
    ps -eo pid=,args= | \
        grep -E "$INCLUDE_REGEX" | \
        grep -Ev "$EXCLUDE_REGEX" | \
        grep -v thermal_wrap | \
        awk '{print $1}' | \
        tr '\n' ' '
}

pid_state() {
    awk '/^State/{print $2}' /proc/$1/status 2>/dev/null
}

all_stopped() {
    local any=0
    for p in $1; do
        local s=$(pid_state $p)
        [ -z "$s" ] && continue
        any=1
        [ "$s" = "T" ] || return 1
    done
    [ $any -eq 1 ] || return 1
    return 0
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "==== thermal_wrapper start: ABORT=${ABORT} PAUSE=${PAUSE} RESUME=${RESUME} TICK=${TICK}s ===="

aborted=0
abort_streak=0
no_pid_streak=0
while true; do
    pids=$(discover_pids)
    if [ -z "$pids" ]; then
        # Allow transient gaps between orchestrator-spawned generate.py runs
        no_pid_streak=$((no_pid_streak+1))
        if [ "$no_pid_streak" -ge 30 ]; then  # 60s with no pids -> exit
            log "no Phase21b PIDs for ${no_pid_streak} ticks (~$((no_pid_streak*TICK))s) — exit"
            break
        fi
        sleep $TICK
        continue
    fi
    no_pid_streak=0

    t=$(get_temp)

    if [ "$t" -ge "$ABORT" ]; then
        abort_streak=$((abort_streak+1))
        # Always force STOP at ABORT threshold to immediately curb heating
        for p in $pids; do kill -STOP $p 2>/dev/null; done
        log "[ABORT-WARN] APU=${t}°C >= ${ABORT}°C streak=${abort_streak}/${ABORT_CONSECUTIVE} — SIGSTOP pids: $pids"
        if [ "$abort_streak" -ge "$ABORT_CONSECUTIVE" ]; then
            log "[ABORT-KILL] APU=${t}°C sustained — SIGKILL pids: $pids"
            for p in $pids; do kill -KILL $p 2>/dev/null; done
            aborted=1
            break
        fi
    elif [ "$t" -ge "$PAUSE" ]; then
        abort_streak=0
        # Only act if at least one is currently NOT stopped
        need_stop=0
        for p in $pids; do
            s=$(pid_state $p); [ -n "$s" ] && [ "$s" != "T" ] && need_stop=1
        done
        if [ "$need_stop" = "1" ]; then
            log "[PAUSE] APU=${t}°C >= ${PAUSE}°C — SIGSTOP pids: $pids"
            for p in $pids; do kill -STOP $p 2>/dev/null; done
        else
            log "[PAUSE-hold] APU=${t}°C pids already stopped: $pids"
        fi
    elif [ "$t" -le "$RESUME" ]; then
        abort_streak=0
        if all_stopped "$pids"; then
            log "[RESUME] APU=${t}°C <= ${RESUME}°C — SIGCONT pids: $pids"
            for p in $pids; do kill -CONT $p 2>/dev/null; done
        else
            log "[OK] APU=${t}°C pids: $pids (not all stopped, no action)"
        fi
    else
        abort_streak=0
        log "[OK] APU=${t}°C pids: $pids"
    fi

    sleep $TICK
done

log "==== thermal_wrapper exit (aborted=${aborted}) ===="
