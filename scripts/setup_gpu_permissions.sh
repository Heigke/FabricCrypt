#!/bin/bash
# Setup GPU permissions for non-root actuator daemon
# Run this script ONCE with sudo: sudo ./setup_gpu_permissions.sh

set -e

echo "=== GPU Permission Setup for FEEL Actuator ==="

# Create gpu-actuator group if it doesn't exist
if ! getent group gpu-actuator > /dev/null 2>&1; then
    echo "Creating gpu-actuator group..."
    groupadd gpu-actuator
fi

# Add current user to the group
CURRENT_USER=${SUDO_USER:-$USER}
echo "Adding $CURRENT_USER to gpu-actuator group..."
usermod -a -G gpu-actuator "$CURRENT_USER"

# Create udev rules for AMD GPU
echo "Creating udev rules for AMD GPU..."
cat > /etc/udev/rules.d/99-amd-gpu-actuator.rules << 'EOF'
# FEEL Actuator - AMD GPU permissions
# Power management
SUBSYSTEM=="drm", KERNEL=="card*", ATTR{device/power_dpm_force_performance_level}=="*", MODE="0664", GROUP="gpu-actuator"

# DPM levels
SUBSYSTEM=="drm", KERNEL=="card*", ATTR{device/pp_dpm_sclk}=="*", MODE="0664", GROUP="gpu-actuator"
SUBSYSTEM=="drm", KERNEL=="card*", ATTR{device/pp_dpm_mclk}=="*", MODE="0664", GROUP="gpu-actuator"

# Power cap via hwmon
SUBSYSTEM=="hwmon", ATTR{name}=="amdgpu", MODE="0664", GROUP="gpu-actuator"
KERNEL=="hwmon*", SUBSYSTEM=="hwmon", ATTR{name}=="amdgpu", RUN+="/bin/chmod g+w /sys/class/hwmon/%k/power1_cap"

# Alternative: specific power files
ACTION=="add", SUBSYSTEM=="hwmon", ATTR{name}=="amdgpu", RUN+="/bin/sh -c 'chgrp gpu-actuator /sys/class/hwmon/%k/power1_cap 2>/dev/null; chmod g+w /sys/class/hwmon/%k/power1_cap 2>/dev/null || true'"
EOF

# Create udev rules for NVIDIA GPU (if present)
echo "Creating udev rules for NVIDIA GPU..."
cat > /etc/udev/rules.d/99-nvidia-gpu-actuator.rules << 'EOF'
# FEEL Actuator - NVIDIA GPU permissions
# nvidia-smi doesn't need special permissions, but nvidia device files do
KERNEL=="nvidia*", MODE="0666", GROUP="gpu-actuator"
KERNEL=="nvidia-uvm*", MODE="0666", GROUP="gpu-actuator"
EOF

# Reload udev rules
echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

# Set permissions immediately for current session (without reboot)
echo "Setting immediate permissions..."

# AMD GPU paths
for hwmon in /sys/class/hwmon/hwmon*; do
    if [ -f "$hwmon/name" ] && grep -q "amdgpu" "$hwmon/name" 2>/dev/null; then
        echo "  Found AMD GPU hwmon: $hwmon"
        chgrp gpu-actuator "$hwmon/power1_cap" 2>/dev/null || true
        chmod g+w "$hwmon/power1_cap" 2>/dev/null || true
    fi
done

for card in /sys/class/drm/card*; do
    if [ -d "$card/device" ]; then
        for f in power_dpm_force_performance_level pp_dpm_sclk pp_dpm_mclk; do
            if [ -f "$card/device/$f" ]; then
                echo "  Setting permissions on $card/device/$f"
                chgrp gpu-actuator "$card/device/$f" 2>/dev/null || true
                chmod g+w "$card/device/$f" 2>/dev/null || true
            fi
        done
    fi
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: You need to log out and log back in for group membership to take effect."
echo "Or run: newgrp gpu-actuator"
echo ""
echo "Then you can start the daemon without sudo:"
echo "  HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 src/actuator/privileged_daemon.py --port 8770 --vendor AMD"
echo ""
echo "To verify permissions:"
echo "  ls -la /sys/class/hwmon/hwmon*/power1_cap"
echo "  groups"
