#!/bin/bash
# Adaptive GPU clock governor — keeps APU thermal_zone0 below trip
# Uses amdgpu sysfs power_dpm_force_performance_level
# States: low (low_apu_C), auto (mid), high (cool)
# Run as: scripts/cooling/gpu_thermal_governor.sh start &
# Stop:   pkill -f gpu_thermal_governor

LOW_TRIP=80      # >this => force low
MID_TRIP=75      # hysteresis mid
COOL_TRIP=70     # <this => allow auto
CARD_DEV=/sys/class/drm/card0/device
PWD_FILE=daedalus_sudo_pass  # ikaros sudo pass = "daedalus"
LOG=/tmp/gpu_thermal_governor.log

current_level() {
    cat $CARD_DEV/power_dpm_force_performance_level 2>/dev/null
}
set_level() {
    local L=$1
    [ "$(current_level)" = "$L" ] && return
    echo "$(date +%H:%M:%S) APU=${APU}C state=$STATE level=$(current_level) -> $L" >> $LOG
    echo daedalus | sudo -S bash -c "echo $L > $CARD_DEV/power_dpm_force_performance_level" 2>/dev/null
}

STATE=auto
while true; do
    APU=$(awk '{printf "%.0f", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
    [ -z "$APU" ] && sleep 5 && continue
    if [ "$APU" -ge "$LOW_TRIP" ]; then
        set_level low; STATE=hot
    elif [ "$APU" -le "$COOL_TRIP" ]; then
        set_level auto; STATE=cool
    elif [ "$APU" -le "$MID_TRIP" ] && [ "$STATE" = "hot" ]; then
        set_level auto; STATE=auto
    fi
    sleep 10
done
