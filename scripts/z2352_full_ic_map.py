#!/usr/bin/env python3
"""z2352e: Full IC register map and deep writability probe.

Map every register from 0x5800 to 0x58FF. For each non-zero or writable
register, record value and writability. Also scan extended ranges.

Then try the real test: redirect CPC IC_BASE and trigger GPU reset
to see if firmware reloads from new address.
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_full_ic_map.json'
    with open(p, 'w') as f:
        json.dump(results, f, indent=2)
    if tag:
        log(f"  [saved: {tag}]")

class MMIO:
    def __init__(self):
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 1024*1024, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=MMIO_BASE)
    def r(self, off):
        b = off * 4
        return struct.unpack('<I', self.mm[b:b+4])[0]
    def w(self, off, val):
        b = off * 4
        self.mm[b:b+4] = struct.pack('<I', val)
    def close(self):
        self.mm.close()
        os.close(self.fd)

mm = MMIO()
log("=== z2352e: Full IC Register Map ===")

# === STEP 1: Complete map 0x5800-0x58FF ===
log("\n--- Step 1: Full register map 0x5800-0x58FF ---")
full_map = {}
for off in range(0x5800, 0x5900):
    val = mm.r(off)
    # Quick writability: try writing 0xFFFFFFFF
    mm.w(off, 0xFFFFFFFF)
    post_ff = mm.r(off)
    # try writing 0x00000000
    mm.w(off, 0x00000000)
    post_00 = mm.r(off)
    # restore
    mm.w(off, val)

    writable = (post_ff != val) or (post_00 != val)
    if val != 0 or writable:
        w_mask = post_ff  # bits that can be set
        log(f"  0x{off:04X}: val=0x{val:08X}  W(FF)=0x{post_ff:08X}  W(00)=0x{post_00:08X}  {'W' if writable else 'R'}")
        full_map[f"0x{off:04X}"] = {
            'val': f"0x{val:08X}",
            'w_ff': f"0x{post_ff:08X}",
            'w_00': f"0x{post_00:08X}",
            'writable': writable,
        }

results['full_map'] = full_map
save("full_map")

# === STEP 2: Try wider scan for RLC-related registers ===
# GFX11 moved many registers. Search for non-zero in 0x4000-0x5FFF
log("\n--- Step 2: Non-zero register scan 0x4000-0x5FFF ---")
nonzero_4000 = {}
total_nz = 0
for off in range(0x4000, 0x6000):
    val = mm.r(off)
    if val != 0:
        total_nz += 1
        if total_nz <= 50:
            nonzero_4000[f"0x{off:04X}"] = f"0x{val:08X}"
            if total_nz <= 20:
                log(f"  0x{off:04X} = 0x{val:08X}")

log(f"  Total non-zero in 0x4000-0x5FFF: {total_nz}")
results['nonzero_4000_5FFF'] = nonzero_4000
results['total_nonzero_4000_5FFF'] = total_nz
save("wide_scan")

# === STEP 3: Try wider scan for GC registers (0x1000-0x3FFF) ===
log("\n--- Step 3: Non-zero register scan 0x1000-0x3FFF ---")
nonzero_gc = {}
total_gc = 0
for off in range(0x1000, 0x4000):
    val = mm.r(off)
    if val != 0:
        total_gc += 1
        if total_gc <= 50:
            nonzero_gc[f"0x{off:04X}"] = f"0x{val:08X}"
            if total_gc <= 20:
                log(f"  0x{off:04X} = 0x{val:08X}")

log(f"  Total non-zero in 0x1000-0x3FFF: {total_gc}")
results['nonzero_1000_3FFF'] = nonzero_gc
results['total_nonzero_1000_3FFF'] = total_gc
save("gc_scan")

# === STEP 4: Check GRBM_GFX_INDEX for GFX11 ===
# On GFX11, you may need to set GRBM_GFX_INDEX before reading many registers
# GRBM_GFX_INDEX is usually at 0x2200 on older gens
log("\n--- Step 4: GRBM_GFX_INDEX probe ---")
# Try known GRBM_GFX_INDEX locations
grbm_candidates = [0x2200, 0xD000, 0xD800, 0x5000, 0x5A00]
for off in grbm_candidates:
    val = mm.r(off)
    if val != 0:
        log(f"  0x{off:04X} = 0x{val:08X} (non-zero, potential GRBM_GFX_INDEX)")

# Try setting GRBM_GFX_INDEX to select SE=0, SH=0, instance=0
# and re-read GRBM_STATUS
grbm_idx_off = 0x2200
orig_idx = mm.r(grbm_idx_off)
mm.w(grbm_idx_off, 0xE0000000)  # broadcast: SE=all, SH=all, INSTANCE=0
time.sleep(0.01)
grbm_status = mm.r(0x2004)
log(f"  After GRBM_GFX_INDEX=0xE0000000: GRBM_STATUS=0x{grbm_status:08X}")
mm.w(grbm_idx_off, orig_idx)  # restore

results['grbm_after_index'] = f"0x{grbm_status:08X}"
save("grbm")

# === STEP 5: Scan VERY high offsets — some GFX11 registers at 0xD000+ ===
log("\n--- Step 5: Non-zero scan 0xD000-0xDFFF and 0x3E000-0x3FFFF ---")
nz_d = 0
for off in range(0xD000, 0xE000):
    val = mm.r(off)
    if val != 0:
        nz_d += 1
        if nz_d <= 10:
            log(f"  0x{off:04X} = 0x{val:08X}")
log(f"  Total non-zero 0xD000-0xDFFF: {nz_d}")

# Check really high offsets (within 1MB aperture = 256K DWORDs = 0x40000 max)
nz_3e = 0
max_off = min(0x40000, 1024*1024 // 4)
for off in range(0x3E000, max_off):
    val = mm.r(off)
    if val != 0:
        nz_3e += 1
        if nz_3e <= 10:
            log(f"  0x{off:05X} = 0x{val:08X}")
log(f"  Total non-zero 0x3E000-0x{max_off:05X}: {nz_3e}")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
