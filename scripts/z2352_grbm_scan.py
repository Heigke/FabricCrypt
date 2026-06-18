#!/usr/bin/env python3
"""z2352c: Scan for GRBM_STATUS and other critical registers via /dev/mem.

GFX11 may have shifted register offsets. Scan common ranges to find
non-zero status registers. Also decode ME_IC_BASE_CNTL properly and
test the full writability of IC_BASE_CNTL to understand what bits
we actually control.

Key question: Can we clear bit 29 of PFP_IC_BASE_CNTL (0x20020000)?
That might be an encryption/protection bit.
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000

results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_grbm_scan.json'
    with open(outpath, 'w') as f:
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

log("=== z2352c: GRBM/Status Register Scan ===")
mm = MMIO()

# === STEP 1: Scan for non-zero registers in key ranges ===
log("\n--- Step 1: Scan register ranges for non-zero values ---")

ranges = [
    ("GC/GRBM 0x2000-0x20FF", 0x2000, 0x2100),
    ("GC/CP   0x2100-0x21FF", 0x2100, 0x2200),
    ("GC/CP   0x2200-0x22FF", 0x2200, 0x2300),
    ("GC/CP   0x2300-0x23FF", 0x2300, 0x2400),
    ("GC/CP   0x2400-0x24FF", 0x2400, 0x2500),
    ("GC/SPI  0x2C00-0x2CFF", 0x2C00, 0x2D00),
    ("RLC     0x4C00-0x4CFF", 0x4C00, 0x4D00),
    ("CP/IC   0x5800-0x5900", 0x5800, 0x5900),
    ("SDMA    0x4E00-0x4EFF", 0x4E00, 0x4F00),
]

nonzero_regs = {}
for desc, start, end in ranges:
    count = 0
    for off in range(start, end):
        val = mm.r(off)
        if val != 0:
            count += 1
            if count <= 10:  # First 10 non-zero per range
                log(f"  0x{off:04X} = 0x{val:08X}  ({desc})")
                nonzero_regs[f"0x{off:04X}"] = f"0x{val:08X}"
    log(f"  {desc}: {count}/{end-start} non-zero")

results['nonzero_regs'] = nonzero_regs
save("scan")

# === STEP 2: Decode IC_BASE_CNTL values ===
log("\n--- Step 2: IC_BASE_CNTL bit-level analysis ---")

for name, off in [('PFP', 0x5842), ('ME', 0x5846), ('CPC', 0x584E), ('MES', 0x5852)]:
    val = mm.r(off)
    vmid = val & 0xF
    addr_clamp = (val >> 4) & 1
    exe_disable = (val >> 23) & 1
    cache_policy = (val >> 24) & 3
    bit29 = (val >> 29) & 1
    log(f"  {name}_IC_BASE_CNTL (0x{off:04X}) = 0x{val:08X}")
    log(f"    VMID={vmid} ADDR_CLAMP={addr_clamp} EXE_DISABLE={exe_disable} CACHE_POLICY={cache_policy} bit29={bit29}")
    log(f"    Binary: {val:032b}")
    results[f'{name}_cntl_decode'] = {
        'value': f"0x{val:08X}",
        'binary': f"{val:032b}",
        'vmid': vmid, 'addr_clamp': addr_clamp,
        'exe_disable': exe_disable, 'cache_policy': cache_policy,
        'bit29': bit29,
    }

save("decode")

# === STEP 3: Bit-by-bit writability test on ME_IC_BASE_CNTL ===
log("\n--- Step 3: Bit-by-bit writability of ME_IC_BASE_CNTL ---")
orig = mm.r(0x5846)
log(f"  Original: 0x{orig:08X} = {orig:032b}")

writable_bits = 0
readonly_bits = 0
for bit in range(32):
    mask = 1 << bit
    # Try setting the bit
    mm.w(0x5846, orig | mask)
    time.sleep(0.001)
    val_set = mm.r(0x5846)
    # Try clearing the bit
    mm.w(0x5846, orig & ~mask)
    time.sleep(0.001)
    val_clr = mm.r(0x5846)
    # Restore
    mm.w(0x5846, orig)

    can_set = bool(val_set & mask)
    can_clr = not bool(val_clr & mask)
    orig_bit = bool(orig & mask)

    if can_set and can_clr:
        status = "R/W"
        writable_bits |= mask
    elif can_set and not can_clr:
        status = "SET-only"
        writable_bits |= mask
    elif not can_set and can_clr:
        status = "CLR-only"
        writable_bits |= mask
    else:
        status = f"R/O (stuck {'1' if orig_bit else '0'})"
        readonly_bits |= mask

    if status != f"R/O (stuck {'1' if orig_bit else '0'})" or orig_bit:
        log(f"  bit {bit:2d}: orig={int(orig_bit)} set→{int(can_set)} clr→{int(can_clr)}  [{status}]")

mm.w(0x5846, orig)  # Final restore
results['me_cntl_writability'] = {
    'writable_mask': f"0x{writable_bits:08X}",
    'readonly_mask': f"0x{readonly_bits:08X}",
}
save("writability")

# === STEP 4: Check IC_BASE_LO/HI for ALL engines ===
log("\n--- Step 4: IC_BASE_LO/HI writability for all engines ---")
base_regs = [
    ('PFP_IC_BASE_LO', 0x5840), ('PFP_IC_BASE_HI', 0x5841),
    ('ME_IC_BASE_LO',  0x5844), ('ME_IC_BASE_HI',  0x5845),
    ('CPC_IC_BASE_LO', 0x584C), ('CPC_IC_BASE_HI', 0x584D),
    ('MES_IC_BASE_LO', 0x5850), ('MES_IC_BASE_HI', 0x5851),
]

base_rw = {}
for name, off in base_regs:
    orig_val = mm.r(off)
    test = 0xAAAAAAAA
    mm.w(off, test)
    time.sleep(0.001)
    post = mm.r(off)
    mm.w(off, orig_val)  # restore
    changed = (post != orig_val)
    log(f"  {name:20s} orig=0x{orig_val:08X}  write=0x{test:08X}  read=0x{post:08X}  {'WRITABLE' if changed else 'READ-ONLY'}")
    base_rw[name] = {
        'original': f"0x{orig_val:08X}",
        'wrote': f"0x{test:08X}",
        'readback': f"0x{post:08X}",
        'writable': changed,
    }

results['base_reg_writability'] = base_rw
save("base_regs")

# === STEP 5: Scan for firmware loading registers (CP_*_UCODE) ===
log("\n--- Step 5: Scan for CP firmware/ucode registers ---")
ucode_scan = {}
# GFX11 CP ucode registers are typically around 0x2150-0x2160
ucode_ranges = [
    ("CP_PFP_UCODE_ADDR", 0x2150),
    ("CP_PFP_UCODE_DATA", 0x2151),
    ("CP_ME_RAM_WADDR", 0x2152),
    ("CP_ME_RAM_DATA", 0x2153),
    ("CP_CE_UCODE_ADDR", 0x2154),
    ("CP_CE_UCODE_DATA", 0x2155),
    ("CP_MEC_ME1_UCODE_ADDR", 0x2158),
    ("CP_MEC_ME1_UCODE_DATA", 0x2159),
]
for name, off in ucode_ranges:
    val = mm.r(off)
    log(f"  {name:35s} (0x{off:04X}) = 0x{val:08X}")
    ucode_scan[name] = f"0x{val:08X}"

results['ucode_registers'] = ucode_scan
save("ucode")

# Check if ucode RAM registers are writable (this would be the key finding)
log("\n  Testing CP_ME_RAM_WADDR writability...")
orig_addr = mm.r(0x2152)
mm.w(0x2152, 0x00000000)
time.sleep(0.001)
post_addr = mm.r(0x2152)
mm.w(0x2152, orig_addr)
log(f"  CP_ME_RAM_WADDR: orig=0x{orig_addr:08X}  write=0x00000000  read=0x{post_addr:08X}  {'WRITABLE' if post_addr==0 else 'READ-ONLY'}")
results['me_ram_waddr_writable'] = (post_addr == 0)

log("\n  Testing CP_ME_RAM_DATA writability (write RISC-V NOP)...")
orig_data = mm.r(0x2153)
mm.w(0x2153, 0x00000013)  # RISC-V NOP
time.sleep(0.001)
post_data = mm.r(0x2153)
mm.w(0x2153, orig_data)
log(f"  CP_ME_RAM_DATA: orig=0x{orig_data:08X}  write=0x00000013  read=0x{post_data:08X}  {'WRITABLE' if post_data==0x13 else 'READ-ONLY or side-effect'}")
results['me_ram_data_writable'] = (post_data == 0x00000013)

save("ucode_write")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
