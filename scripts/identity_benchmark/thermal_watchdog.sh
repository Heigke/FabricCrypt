#!/bin/bash
# Thermal safety watchdog: SIGSTOP a training process when the APU gets too hot,
# SIGCONT it once cooled. Prevents the 99C ACPI trip during long unattended runs.
#
# Usage: thermal_watchdog.sh <PID>
# Signals a SPECIFIC pid (not a pattern) so the watchdog can NEVER stop itself
# (the earlier pattern version self-matched on its own argv and froze before CONT).
PID="${1:?need target PID}"
HOT=${HOT:-95000}      # mC: pause above this
COOL=${COOL:-80000}    # mC: resume below this
ZONE=/sys/class/thermal/thermal_zone0/temp
LOG=/tmp/thermal_watchdog_${PID}.log
echo "watchdog start pid=$PID HOT=$HOT COOL=$COOL $(date)" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  T=$(cat "$ZONE" 2>/dev/null || echo 0)
  if [ "$T" -gt "$HOT" ]; then
    echo "$(date) HOT ${T}mC -> STOP $PID" >> "$LOG"
    kill -STOP "$PID"
    while kill -0 "$PID" 2>/dev/null; do
      sleep 3
      C=$(cat "$ZONE" 2>/dev/null || echo 0)
      [ "$C" -lt "$COOL" ] && break
    done
    echo "$(date) COOL $(cat $ZONE)mC -> CONT $PID" >> "$LOG"
    kill -CONT "$PID"
  fi
  sleep 4
done
echo "watchdog exit (pid gone) $(date)" >> "$LOG"
