#!/bin/bash
# z2352_restore_original.sh — Emergency restore of original MEC firmware
# Usage: sudo bash scripts/z2352_restore_original.sh

echo "=== Restoring Original MEC Firmware ==="

if [ ! -f /lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst.orig ]; then
    echo "ERROR: No backup found at /lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst.orig"
    exit 1
fi

cp /lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst.orig /lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst
echo "Restored original firmware"

# Remove fw_load_type=0 from grub
sed -i 's/ amdgpu.fw_load_type=0//' /etc/default/grub
update-grub 2>/dev/null
echo "Removed fw_load_type=0 from GRUB"

echo ""
echo "Reboot to apply: sudo reboot"
