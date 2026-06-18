#!/usr/bin/env python3
"""z2352: Probe BAR0 for GPU register aperture via /dev/mem.

BAR0 at 0x6800000000 (256MB). The amdgpu driver maps a register
aperture somewhere within BAR0 (adev->rmmio). We need to find where
registers live to do direct MMIO writes bypassing debugfs shadows.

Also checks BAR2 (doorbell, 0xb4000000) and BAR5 (0xb4400000, ISP).
"""
import mmap, struct, os, json, time, sys

BAR0_PHYS = 0x6800000000   # 256MB VRAM + possibly register aperture
BAR2_PHYS = 0xb4000000     # 2MB doorbell
BAR5_PHYS = 0xb4400000     # 1MB (ISP, not GPU regs)

# Known register values from debugfs for validation
KNOWN_REGS = {
    'ME_IC_BASE_CNTL': (0x5846, 0x0000531F),
    'PFP_IC_BASE_CNTL': (0x5842, 0x20020000),
}

results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save():
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_bar0_reg_probe.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)

def read_dword_devmem(fd, phys_addr):
    """Read a single DWORD from physical address via /dev/mem."""
    page = phys_addr & ~0xFFF
    offset = phys_addr & 0xFFF
    mm = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=page)
    val = struct.unpack('<I', mm[offset:offset+4])[0]
    mm.close()
    return val

def write_dword_devmem(fd, phys_addr, value):
    """Write a single DWORD to physical address via /dev/mem."""
    page = phys_addr & ~0xFFF
    offset = phys_addr & 0xFFF
    mm = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=page)
    mm[offset:offset+4] = struct.pack('<I', value)
    mm.close()

log("=== z2352: BAR0 Register Aperture Probe ===")

fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)

# === STEP 1: Scan BAR0 for non-VRAM data (register aperture) ===
log("\n--- Step 1: Scan BAR0 at 1MB intervals for register-like data ---")
bar0_scan = {}

# Sample at 1MB intervals across 256MB BAR0
for mb in range(0, 256, 1):
    base = BAR0_PHYS + mb * 1024 * 1024
    try:
        mm = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
        vals = struct.unpack('<16I', mm[:64])
        mm.close()

        # Check for register-like patterns (not all-zero, not all-FF, not encrypted noise)
        unique = len(set(vals))
        all_zero = all(v == 0 for v in vals)
        all_ff = all(v == 0xFFFFFFFF for v in vals)

        # High entropy = encrypted VRAM, low unique = zeroed, registers = mixed
        if not all_zero and not all_ff and unique < 14:
            log(f"  BAR0+{mb:3d}MB: {unique:2d} unique vals - {' '.join(f'0x{v:08X}' for v in vals[:4])} ... POTENTIAL REGS")
            bar0_scan[f"{mb}MB"] = [f"0x{v:08X}" for v in vals[:8]]
        elif mb % 32 == 0:
            # Print every 32MB for progress
            tag = "ZERO" if all_zero else ("FF" if all_ff else f"{unique}uniq")
            log(f"  BAR0+{mb:3d}MB: {tag}")
    except Exception as e:
        log(f"  BAR0+{mb:3d}MB: ERROR {e}")
        bar0_scan[f"{mb}MB"] = f"ERROR: {e}"

results['bar0_scan'] = bar0_scan
save()

# === STEP 2: Check if register offsets are at BAR0 + reg_offset*4 ===
log("\n--- Step 2: Try direct register read at BAR0 + offset*4 ---")

# On some AMD GPUs, registers are at the very start of BAR0 or at a fixed offset
# Try common register aperture bases
test_bases = [
    ("BAR0+0", BAR0_PHYS),
    ("BAR0+252MB", BAR0_PHYS + 252 * 1024 * 1024),
    ("BAR0+254MB", BAR0_PHYS + 254 * 1024 * 1024),
    ("BAR0+255MB", BAR0_PHYS + 255 * 1024 * 1024),
    ("BAR2", BAR2_PHYS),
]

reg_search = {}
for name, base in test_bases:
    for reg_name, (reg_off, expected) in KNOWN_REGS.items():
        phys = base + reg_off * 4
        try:
            val = read_dword_devmem(fd, phys)
            match = (val == expected)
            tag = "MATCH!" if match else ""
            log(f"  {name} + 0x{reg_off:04X}*4 = 0x{val:08X} (expect 0x{expected:08X}) {tag}")
            reg_search[f"{name}_{reg_name}"] = {
                'phys': f"0x{phys:012X}",
                'value': f"0x{val:08X}",
                'expected': f"0x{expected:08X}",
                'match': match,
            }
            if match:
                log(f"  *** FOUND REGISTER APERTURE at {name}! ***")
        except Exception as e:
            log(f"  {name} + 0x{reg_off:04X}*4 = ERROR: {e}")
            reg_search[f"{name}_{reg_name}"] = f"ERROR: {e}"

results['reg_search'] = reg_search
save()

# === STEP 3: Try reading the amdgpu driver's rmmio base from /proc ===
log("\n--- Step 3: Find amdgpu rmmio base from /proc/iomem ---")
try:
    with open('/proc/iomem', 'r') as f:
        iomem = f.read()
    # Find all c3:00.0 regions
    for line in iomem.splitlines():
        if 'c3:00.0' in line or 'amdgpu' in line.lower():
            log(f"  {line.strip()}")
except Exception as e:
    log(f"  ERROR reading /proc/iomem: {e}")

# === STEP 4: Scan for known register values in BAR0 at finer granularity ===
# ME_IC_BASE_CNTL = 0x0000531F is distinctive enough to search for
log("\n--- Step 4: Search for 0x0000531F pattern in BAR0 ---")
target = 0x0000531F
found_locs = []

for mb in range(0, 256, 1):
    base = BAR0_PHYS + mb * 1024 * 1024
    try:
        mm = mmap.mmap(fd, 1024 * 1024, mmap.MAP_SHARED, mmap.PROT_READ, offset=base)
        data = mm[:]
        mm.close()

        # Search for target value
        target_bytes = struct.pack('<I', target)
        pos = 0
        while True:
            idx = data.find(target_bytes, pos)
            if idx == -1:
                break
            abs_off = mb * 1024 * 1024 + idx
            # Check if this could be at the right register offset
            # ME_IC_BASE_CNTL is at 0x5846 * 4 = 0x16118
            possible_base = abs_off - 0x5846 * 4
            log(f"  Found 0x{target:08X} at BAR0+0x{abs_off:08X} (implies reg base at BAR0+0x{possible_base:08X})")
            found_locs.append({
                'offset': f"0x{abs_off:08X}",
                'implied_base': f"0x{possible_base:08X}",
            })
            pos = idx + 4
            if len(found_locs) > 20:
                break
    except Exception as e:
        if mb % 64 == 0:
            log(f"  BAR0+{mb}MB: {e}")

    if len(found_locs) > 20:
        log("  (stopped after 20 matches)")
        break

results['pattern_search'] = found_locs
save()

# === STEP 5: Check if MMIO is at a completely different physical address ===
log("\n--- Step 5: Check extended physical address ranges ---")
# Some AMD APU/dGPU hybrids put MMIO at high physical addresses
# Check /proc/iomem more carefully
extended_ranges = []
try:
    with open('/proc/iomem', 'r') as f:
        for line in f:
            if 'c3:00' in line:
                parts = line.strip().split(':')
                addr_range = parts[0].strip()
                desc = ':'.join(parts[1:]).strip()
                start, end = addr_range.split('-')
                start_addr = int(start.strip(), 16)
                end_addr = int(end.strip(), 16)
                size = end_addr - start_addr + 1
                log(f"  0x{start_addr:012X}-0x{end_addr:012X} ({size//1024}KB) : {desc}")
                extended_ranges.append({
                    'start': f"0x{start_addr:012X}",
                    'end': f"0x{end_addr:012X}",
                    'size_kb': size // 1024,
                    'desc': desc,
                })
except Exception as e:
    log(f"  ERROR: {e}")

results['pci_regions'] = extended_ranges

# For each non-VRAM region, try reading register values
for region in extended_ranges:
    start = int(region['start'], 16)
    size_kb = region['size_kb']
    if size_kb <= 4096 and size_kb > 0 and start != BAR0_PHYS:
        log(f"\n  Probing region 0x{start:012X} ({size_kb}KB):")
        for reg_name, (reg_off, expected) in KNOWN_REGS.items():
            phys = start + reg_off * 4
            if phys < start + size_kb * 1024:
                try:
                    val = read_dword_devmem(fd, phys)
                    match = (val == expected)
                    tag = " *** MATCH ***" if match else ""
                    log(f"    +0x{reg_off*4:06X} ({reg_name}) = 0x{val:08X} (expect 0x{expected:08X}){tag}")
                    if match:
                        results['REGISTER_APERTURE_FOUND'] = {
                            'base_phys': f"0x{start:012X}",
                            'region': region['desc'],
                        }
                except Exception as e:
                    log(f"    +0x{reg_off*4:06X} ({reg_name}) = ERROR: {e}")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save()
os.close(fd)
log("\nDone. Results saved.")
