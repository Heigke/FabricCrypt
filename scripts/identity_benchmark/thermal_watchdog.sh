#!/bin/bash
# Thermal safety watchdog: SIGSTOP a training process when the APU gets too hot,
# SIGCONT it once cooled. Prevents the 99C ACPI trip during long unattended runs.
#
# Usage: thermal_watchdog.sh <PID>
# Signals a SPECIFIC pid (not a pattern) so the watchdog can NEVER stop itself
# (the earlier pattern version self-matched on its own argv and froze before CONT).
PID="${1:?need target PID}"
HOT=${HOT:-90000}      # mC: pause above this (lowered for margin vs 99C ACPI trip)
COOL=${COOL:-78000}    # mC: resume below this
ZONE=/sys/class/thermal/thermal_zone0/temp
LOG=/tmp/thermal_watchdog_${PID}.log
echo "watchdog start pid=$PID HOT=$HOT COOL=$COOL $(date)" >> "$LOG"

# alive() is TRUE only if the target exists AND is not a zombie. `kill -0` succeeds on a
# zombie (finished-but-unreaped) process, which made the old loop spin forever SIGSTOP/CONT-ing
# a dead PID and leave its parent stopped (the v13/v14 freeze bug). Read state from /proc.
alive() {
  local st
  st=$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null) || return 1
  [ -n "$st" ] && [ "$st" != "Z" ]
}

while alive "$PID"; do
  T=$(cat "$ZONE" 2>/dev/null || echo 0)
  if [ "$T" -gt "$HOT" ]; then
    echo "$(date) HOT ${T}mC -> STOP $PID" >> "$LOG"
    kill -STOP "$PID"
    while alive "$PID"; do
      sleep 1
      C=$(cat "$ZONE" 2>/dev/null || echo 0)
      [ "$C" -lt "$COOL" ] && break
    done
    echo "$(date) COOL $(cat $ZONE)mC -> CONT $PID" >> "$LOG"
    kill -CONT "$PID" 2>/dev/null
  fi
  sleep 0.4   # APU heats ~13C/s under GPU load — sub-second poll keeps overshoot < ~6C
done
kill -CONT "$PID" 2>/dev/null   # NEVER leave the target (or its waiting parent) stopped on exit
echo "watchdog exit (pid gone/zombie) $(date)" >> "$LOG"
