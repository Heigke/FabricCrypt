#!/bin/bash
# post_reboot_fw_direct_test.sh — Run after reboot with amdgpu.fw_load_type=0
# Captures all diagnostics needed to determine if DIRECT loading worked

OUT="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts/fw_direct_results.txt"
echo "=== fw_load_type=0 POST-REBOOT DIAGNOSTICS ===" > "$OUT"
echo "Date: $(date)" >> "$OUT"
echo "Kernel: $(uname -r)" >> "$OUT"
echo "" >> "$OUT"

# 1. Check boot params
echo "=== BOOT PARAMS ===" >> "$OUT"
cat /proc/cmdline >> "$OUT"
echo "" >> "$OUT"

# 2. Check if amdgpu loaded at all
echo "=== AMDGPU MODULE ===" >> "$OUT"
lsmod | grep amdgpu >> "$OUT" 2>&1
echo "" >> "$OUT"

# 3. Full dmesg amdgpu output
echo "=== DMESG AMDGPU (full) ===" >> "$OUT"
dmesg | grep -i amdgpu >> "$OUT" 2>&1
echo "" >> "$OUT"

# 4. Check for firmware loading messages
echo "=== FIRMWARE LOADING ===" >> "$OUT"
dmesg | grep -iE 'firmware|fw_load|direct|psp|mes|mec|ucode' >> "$OUT" 2>&1
echo "" >> "$OUT"

# 5. Check GPU device
echo "=== GPU DEVICE ===" >> "$OUT"
lspci -vnn | grep -A20 "VGA\|Display" >> "$OUT" 2>&1
echo "" >> "$OUT"

# 6. Check for errors
echo "=== ERRORS ===" >> "$OUT"
dmesg | grep -iE 'error|fail|timeout|fault' | grep -i amdgpu >> "$OUT" 2>&1
echo "" >> "$OUT"

# 7. Check debugfs for firmware info
echo "=== DEBUGFS FW INFO ===" >> "$OUT"
if [ -f /sys/kernel/debug/dri/1/amdgpu_firmware_info ]; then
    sudo cat /sys/kernel/debug/dri/1/amdgpu_firmware_info >> "$OUT" 2>&1
elif [ -f /sys/kernel/debug/dri/0/amdgpu_firmware_info ]; then
    sudo cat /sys/kernel/debug/dri/0/amdgpu_firmware_info >> "$OUT" 2>&1
fi
echo "" >> "$OUT"

# 8. Check if DRM device exists
echo "=== DRM DEVICES ===" >> "$OUT"
ls -la /dev/dri/ >> "$OUT" 2>&1
echo "" >> "$OUT"

# 9. Check amdgpu parameters
echo "=== AMDGPU PARAMS ===" >> "$OUT"
if [ -d /sys/module/amdgpu/parameters ]; then
    for f in /sys/module/amdgpu/parameters/fw_load_type /sys/module/amdgpu/parameters/ppfeaturemask /sys/module/amdgpu/parameters/cg_mask; do
        echo "$f = $(cat $f 2>/dev/null)" >> "$OUT"
    done
fi
echo "" >> "$OUT"

# 10. Check PSP status in dmesg
echo "=== PSP STATUS ===" >> "$OUT"
dmesg | grep -iE 'psp|tmr|trusted' >> "$OUT" 2>&1
echo "" >> "$OUT"

# 11. Quick HIP test
echo "=== HIP DEVICE CHECK ===" >> "$OUT"
HSA_OVERRIDE_GFX_VERSION=11.0.0 /opt/rocm/bin/rocminfo 2>&1 | head -60 >> "$OUT"
echo "" >> "$OUT"

echo "=== DONE ===" >> "$OUT"
echo "Results saved to $OUT"
cat "$OUT"
