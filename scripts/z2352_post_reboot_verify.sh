#!/bin/bash
# z2352_post_reboot_verify.sh — Run after reboot to verify MEC firmware injection
# Usage: sudo bash scripts/z2352_post_reboot_verify.sh

echo "=== z2352 Post-Reboot Firmware Injection Verification ==="
echo "Date: $(date)"
echo ""

echo "1. Kernel cmdline:"
cat /proc/cmdline
echo ""

echo "2. fw_load_type parameter:"
cat /sys/module/amdgpu/parameters/fw_load_type 2>/dev/null || echo "  NOT AVAILABLE"
echo ""

echo "3. MEC firmware version (expect 0x0000FEE1 for FEEL injection):"
sudo cat /sys/kernel/debug/dri/128/amdgpu_firmware_info 2>/dev/null | grep -i mec
echo ""

echo "4. dmesg MEC/firmware lines:"
dmesg | grep -iE "mec|firmware.*version|fw_load|direct.*load" | tail -20
echo ""

echo "5. GPU status:"
lspci | grep 1586
echo ""

echo "6. amdgpu module loaded:"
lsmod | grep amdgpu | head -3
echo ""

echo "7. Installed firmware file check:"
python3 -c "
import struct, subprocess
fw = subprocess.run(['zstd', '-d', '-c', '/lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst'], capture_output=True).stdout
ver = struct.unpack_from('<I', fw, 16)[0]
marker = struct.unpack_from('<I', fw, 256 + 18871*4)[0]
print(f'  ucode_version: 0x{ver:08X} ({\"FEEL\" if ver == 0xFEE1 else \"ORIGINAL\"})')
print(f'  marker:        0x{marker:08X} ({\"FEEL\" if marker == 0xFEE10001 else \"NONE\"})')
"

echo ""
echo "=== VERDICT ==="
FW_VER=$(sudo cat /sys/kernel/debug/dri/128/amdgpu_firmware_info 2>/dev/null | grep "MEC.*firmware version" | grep -o "0x[0-9a-fA-F]*")
FW_LOAD=$(cat /sys/module/amdgpu/parameters/fw_load_type 2>/dev/null)

if [ "$FW_VER" = "0x0000fee1" ] || [ "$FW_VER" = "0x0000FEE1" ]; then
    echo "*** INJECTION SUCCESSFUL — MEC running FEEL firmware version 0xFEE1 ***"
elif [ "$FW_LOAD" = "0" ]; then
    echo "fw_load_type=0 (DIRECT) active but MEC version is $FW_VER"
    echo "Check dmesg for firmware loading errors"
else
    echo "fw_load_type=$FW_LOAD — NOT in DIRECT mode"
    echo "MEC version: $FW_VER"
fi
