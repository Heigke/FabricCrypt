#!/usr/bin/env python3
"""z2352m: Correct IC_OP_CNTL offset test!

BREAKTHROUGH: Kernel source reveals CP_CPC_IC_OP_CNTL is at 0x297A,
NOT 0x584F. All previous invalidate attempts used the WRONG register.

Also found:
  CP_PFP_IC_OP_CNTL = 0x5843  (we had this right)
  CP_ME_IC_OP_CNTL  = 0x5847  (we had this right)
  CP_CPC_IC_OP_CNTL = 0x297A  (we had 0x584F — WRONG!)

And RS64 registers:
  CP_MEC_RS64_PRGRM_CNTR_START = 0x2900
  CP_MEC_DC_BASE_LO = 0x5870
  CP_MEC_DC_BASE_HI = 0x5871

The driver sequence for compute loading:
  1. Write IC_BASE_LO/HI with firmware address
  2. Set IC_BASE_CNTL (VMID=0, ADDRESS_CLAMP=1)
  3. Write IC_OP_CNTL with INVALIDATE_CACHE=1
  4. Poll for INVALIDATE_CACHE_COMPLETE (bit 1)
  5. Write IC_OP_CNTL with PRIME_ICACHE=1
  6. Poll for ICACHE_PRIMED (bit 5)
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_correct_ic_op.json'
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
log("=== z2352m: Correct IC_OP_CNTL Offset Test ===")

# Correct register offsets from kernel source gc_11_0_0_offset.h
CORRECT_REGS = {
    'CP_PFP_IC_OP_CNTL':  0x5843,  # We had right
    'CP_ME_IC_OP_CNTL':   0x5847,  # We had right
    'CP_CPC_IC_OP_CNTL':  0x297A,  # WE HAD WRONG (was 0x584F)
    # MEC doesn't have a separate IC_OP_CNTL — uses CPC's
}

# Also check these newly discovered registers
RS64_REGS = {
    'CP_MEC_RS64_PRGRM_CNTR_START':    0x2900,
    'CP_MEC_RS64_PRGRM_CNTR_START_HI': 0x2938,
    'CP_MEC_DC_BASE_LO':               0x5870,
    'CP_MEC_DC_BASE_HI':               0x5871,
    'CP_MEC_DC_BASE_CNTL':             0x5872,  # assumed
}

# === STEP 1: Read ALL IC_OP_CNTL registers ===
log("\n--- Step 1: Read IC_OP_CNTL registers (correct offsets) ---")
op_cntl_state = {}
for name, off in CORRECT_REGS.items():
    val = mm.r(off)
    log(f"  {name:25s} (0x{off:04X}) = 0x{val:08X}")
    op_cntl_state[name] = f"0x{val:08X}"

# Also check the WRONG offset we were using
wrong = mm.r(0x584F)
log(f"  {'WRONG_0x584F':25s} (0x584F) = 0x{wrong:08X}")
op_cntl_state['WRONG_0x584F'] = f"0x{wrong:08X}"

results['ic_op_cntl_state'] = op_cntl_state
save("op_cntl_read")

# === STEP 2: Read RS64 registers ===
log("\n--- Step 2: RS64 MEC registers ---")
rs64_state = {}
for name, off in RS64_REGS.items():
    val = mm.r(off)
    # Writability test
    mm.w(off, 0xAAAAAAAA)
    time.sleep(0.001)
    post = mm.r(off)
    mm.w(off, val)
    writable = (post != val)
    log(f"  {name:40s} (0x{off:04X}) = 0x{val:08X}  {'W' if writable else 'R'}")
    rs64_state[name] = {'value': f"0x{val:08X}", 'writable': writable}

results['rs64_state'] = rs64_state
save("rs64")

# === STEP 3: Writability of CP_CPC_IC_OP_CNTL (correct offset!) ===
log("\n--- Step 3: CP_CPC_IC_OP_CNTL (0x297A) writability ---")
orig_op = mm.r(0x297A)
writable_mask = 0
for bit in range(32):
    mask = 1 << bit
    mm.w(0x297A, orig_op | mask)
    time.sleep(0.001)
    val_set = mm.r(0x297A)
    mm.w(0x297A, orig_op & ~mask)
    time.sleep(0.001)
    val_clr = mm.r(0x297A)
    mm.w(0x297A, orig_op)

    can_set = bool(val_set & mask)
    can_clr = not bool(val_clr & mask)

    if can_set and can_clr:
        status = "R/W"
        writable_mask |= mask
    elif can_set:
        status = "SET-only"
        writable_mask |= mask
    elif can_clr:
        status = "CLR-only"
        writable_mask |= mask
    else:
        status = "R/O"

    if status != "R/O" or (orig_op & mask):
        log(f"  bit {bit:2d}: [{status}]")

results['cpc_ic_op_cntl_writable'] = f"0x{writable_mask:08X}"
log(f"  CPC IC_OP_CNTL writable mask: 0x{writable_mask:08X}")
save("op_cntl_bits")

# === STEP 4: THE BIG TEST — IC Invalidate via correct register! ===
log("\n--- Step 4: IC Invalidate via CP_CPC_IC_OP_CNTL (0x297A) ---")

# First, read current CPC IC_BASE to know what we're working with
cpc_lo = mm.r(0x584C)
cpc_hi = mm.r(0x584D)
cpc_cntl = mm.r(0x584E)
log(f"  CPC IC_BASE_LO   = 0x{cpc_lo:08X}")
log(f"  CPC IC_BASE_HI   = 0x{cpc_hi:08X}")
log(f"  CPC IC_BASE_CNTL = 0x{cpc_cntl:08X}")

# Clear first
mm.w(0x297A, 0x00)
time.sleep(0.01)

# Write INVALIDATE_CACHE (bit 0)
log("  Writing INVALIDATE_CACHE (bit 0) to 0x297A...")
mm.w(0x297A, 0x01)
time.sleep(0.1)

# Read back — check INVALIDATE_CACHE_COMPLETE (bit 1)
post = mm.r(0x297A)
inv_complete = bool(post & 0x02)
log(f"  CP_CPC_IC_OP_CNTL = 0x{post:08X}  INV_COMPLETE={inv_complete}")

results['cpc_invalidate'] = {
    'register': '0x297A',
    'wrote': '0x00000001',
    'readback': f"0x{post:08X}",
    'inv_complete': inv_complete,
}
save("invalidate_test")

if inv_complete:
    log("  *** INV_COMPLETE IS HIGH! Invalidation succeeded! ***")

    # Try PRIME
    mm.w(0x297A, 0x00)
    time.sleep(0.01)
    mm.w(0x297A, 0x10)  # PRIME_ICACHE (bit 4)
    time.sleep(0.2)
    post_prime = mm.r(0x297A)
    primed = bool(post_prime & 0x20)
    log(f"  After PRIME: 0x{post_prime:08X}  PRIMED={primed}")
    results['cpc_prime'] = {
        'readback': f"0x{post_prime:08X}",
        'primed': primed,
    }
    save("prime_test")
else:
    log("  INV_COMPLETE not set at 0x297A either.")

# === STEP 5: Try ME and PFP IC invalidate (these were at correct offsets already) ===
log("\n--- Step 5: Verify ME/PFP IC invalidate (known offsets) ---")
for name, off in [('ME', 0x5847), ('PFP', 0x5843)]:
    mm.w(off, 0x00)
    time.sleep(0.01)
    mm.w(off, 0x01)
    time.sleep(0.1)
    post = mm.r(off)
    inv_c = bool(post & 0x02)
    log(f"  {name} IC_OP_CNTL = 0x{post:08X}  INV_COMPLETE={inv_c}")
    results[f'{name}_inv_verify'] = {'value': f"0x{post:08X}", 'inv_complete': inv_c}

save("me_pfp_verify")

# === STEP 6: Look for MEC-specific IC_OP_CNTL in kernel headers ===
# The kernel shows CP_CPC_IC_OP_CNTL at 0x297A. But MEC shares CPC's IC?
# Let's scan the area around 0x297A for other related registers
log("\n--- Step 6: Scan 0x2970-0x2990 for related CP IC registers ---")
area_297x = {}
for off in range(0x2970, 0x2990):
    val = mm.r(off)
    if val != 0:
        mm.w(off, 0xFFFFFFFF)
        time.sleep(0.001)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        log(f"  0x{off:04X} = 0x{val:08X}  {'W' if writable else 'R'}")
        area_297x[f"0x{off:04X}"] = {'value': f"0x{val:08X}", 'writable': writable}

results['area_297x'] = area_297x
save("area_297x")

# === STEP 7: Deep IC_OP_CNTL polling — maybe it takes more time ===
log("\n--- Step 7: Extended IC invalidate polling ---")
# Try CPC invalidate with extended polling (50ms like the driver)
mm.w(0x297A, 0x00)
time.sleep(0.01)
mm.w(0x297A, 0x01)

polls = []
for i in range(500):
    val = mm.r(0x297A)
    if i < 10 or i % 50 == 0:
        log(f"  poll {i}: 0x{val:08X}")
    polls.append(f"0x{val:08X}")
    if val & 0x02:
        log(f"  *** INV_COMPLETE at poll {i}! ***")
        break
    time.sleep(0.0001)  # 100µs like driver

unique_vals = set(polls)
log(f"  Unique poll values: {unique_vals}")
results['extended_poll'] = {
    'total_polls': len(polls),
    'unique_values': list(unique_vals),
    'completed': bool(int(polls[-1], 16) & 0x02),
}
save("extended_poll")

# === STEP 8: Try the FULL driver sequence ===
log("\n--- Step 8: Full driver firmware loading sequence emulation ---")
# The driver does:
# 1. Halt MEC (CP_MEC_CNTL |= MEC_ME1_HALT)
# 2. Set IC_BASE_LO/HI
# 3. Set IC_BASE_CNTL (VMID=0, ADDRESS_CLAMP=1, EXE_DISABLE=0)
# 4. Set DC_BASE_LO/HI (for RS64)
# 5. Invalidate IC
# 6. Prime IC

# Save original state
orig_cpc_lo = mm.r(0x584C)
orig_cpc_hi = mm.r(0x584D)
orig_cpc_cntl = mm.r(0x584E)
orig_mec_cntl = mm.r(0x2188)
orig_dc_lo = mm.r(0x5870)
orig_dc_hi = mm.r(0x5871)

log(f"  Saving state...")
log(f"  CPC_IC_BASE_LO = 0x{orig_cpc_lo:08X}")
log(f"  CPC_IC_BASE_HI = 0x{orig_cpc_hi:08X}")
log(f"  CPC_IC_BASE_CNTL = 0x{orig_cpc_cntl:08X}")
log(f"  CP_MEC_CNTL = 0x{orig_mec_cntl:08X}")
log(f"  MEC_DC_BASE_LO = 0x{orig_dc_lo:08X}")
log(f"  MEC_DC_BASE_HI = 0x{orig_dc_hi:08X}")

save("pre_full_sequence")

# Step 8.1: Halt MEC
log("  [8.1] Halting MEC (bit 28 of CP_MEC_CNTL)...")
mm.w(0x2188, orig_mec_cntl | (1 << 28))
time.sleep(0.05)
log(f"  CP_MEC_CNTL = 0x{mm.r(0x2188):08X}")

# Step 8.2: Set IC_BASE to our NOP sled in VRAM
nop_addr = 0x00F00000
log(f"  [8.2] Setting CPC IC_BASE_LO = 0x{nop_addr:08X} (NOP sled)...")
mm.w(0x584C, nop_addr & 0xFFFFF000)  # 4KB aligned per driver
time.sleep(0.01)
mm.w(0x584D, 0x0000)
time.sleep(0.01)
log(f"  CPC IC_BASE_LO = 0x{mm.r(0x584C):08X}")
log(f"  CPC IC_BASE_HI = 0x{mm.r(0x584D):08X}")

# Step 8.3: Set IC_BASE_CNTL (VMID=0, ADDRESS_CLAMP=1, EXE_DISABLE=0, CACHE_POLICY=0)
cntl_val = (0 & 0xF) | (1 << 4)  # VMID=0, ADDRESS_CLAMP=1
log(f"  [8.3] Setting CPC IC_BASE_CNTL = 0x{cntl_val:08X}...")
mm.w(0x584E, cntl_val)
time.sleep(0.01)
log(f"  CPC IC_BASE_CNTL = 0x{mm.r(0x584E):08X}")

# Step 8.4: Set DC_BASE (data cache — not strictly needed for IC test)
log(f"  [8.4] Setting MEC DC_BASE...")
mm.w(0x5870, nop_addr & 0xFFFFF000)
time.sleep(0.01)
mm.w(0x5871, 0x0000)
time.sleep(0.01)
log(f"  MEC_DC_BASE_LO = 0x{mm.r(0x5870):08X}")

# Step 8.5: Invalidate IC (using CORRECT register 0x297A)
log(f"  [8.5] Invalidating IC at 0x297A...")
op = mm.r(0x297A)
op |= 0x01  # INVALIDATE_CACHE
mm.w(0x297A, op)

# Poll for completion (up to 50ms, 100µs intervals)
completed = False
for i in range(500):
    val = mm.r(0x297A)
    if val & 0x02:  # INVALIDATE_CACHE_COMPLETE
        completed = True
        log(f"  *** INV_COMPLETE at poll {i} (t={i*0.1:.1f}ms)! val=0x{val:08X} ***")
        break
    time.sleep(0.0001)

if not completed:
    final_op = mm.r(0x297A)
    log(f"  INV_COMPLETE NOT set after 50ms. Final: 0x{final_op:08X}")

results['full_sequence'] = {
    'nop_addr': f"0x{nop_addr:08X}",
    'ic_base_lo_set': f"0x{mm.r(0x584C):08X}",
    'ic_base_cntl_set': f"0x{mm.r(0x584E):08X}",
    'inv_complete': completed,
}
save("full_sequence")

# Step 8.6: If invalidate completed, try prime
if completed:
    log(f"  [8.6] Priming IC...")
    mm.w(0x297A, 0x10)  # PRIME_ICACHE
    primed = False
    for i in range(500):
        val = mm.r(0x297A)
        if val & 0x20:  # ICACHE_PRIMED
            primed = True
            log(f"  *** PRIMED at poll {i}! val=0x{val:08X} ***")
            break
        time.sleep(0.0001)

    if not primed:
        log(f"  PRIME failed. Final: 0x{mm.r(0x297A):08X}")

    results['full_sequence']['primed'] = primed

    # Step 8.7: Unhalt MEC
    log(f"  [8.7] Unhalting MEC...")
    mm.w(0x2188, orig_mec_cntl & ~(1 << 28))
    time.sleep(0.1)

    # Check if GPU still alive
    grbm = mm.r(0x2004)
    log(f"  GRBM_STATUS = 0x{grbm:08X}")
    results['full_sequence']['gpu_alive'] = True
    results['full_sequence']['post_grbm'] = f"0x{grbm:08X}"
else:
    # Unhalt and restore
    log("  Unhalting and restoring...")
    mm.w(0x2188, orig_mec_cntl)

# Restore everything
log("\n  Restoring original state...")
mm.w(0x584C, orig_cpc_lo)
mm.w(0x584D, orig_cpc_hi)
mm.w(0x584E, orig_cpc_cntl)
mm.w(0x5870, orig_dc_lo)
mm.w(0x5871, orig_dc_hi)
mm.w(0x2188, orig_mec_cntl)
time.sleep(0.1)

# Verify restore
for name, off in [('CPC_IC_BASE_LO', 0x584C), ('CPC_IC_BASE_CNTL', 0x584E),
                   ('CP_MEC_CNTL', 0x2188)]:
    val = mm.r(off)
    log(f"  {name} = 0x{val:08X}")

save("restored")

# === STEP 9: Try MEC IC_BASE registers at the CORRECT locations ===
# Maybe the kernel maps MEC IC differently from CPC IC
log("\n--- Step 9: Search for MEC-specific IC_OP in extended ranges ---")
# Scan 0x2900-0x2950 for MEC RS64 registers
mec_rs64_scan = {}
for off in range(0x2900, 0x2950):
    val = mm.r(off)
    if val != 0:
        mm.w(off, 0xAAAAAAAA)
        time.sleep(0.001)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        log(f"  0x{off:04X} = 0x{val:08X}  {'W' if writable else 'R'}")
        mec_rs64_scan[f"0x{off:04X}"] = {'value': f"0x{val:08X}", 'writable': writable}

# Also scan 0x5848-0x584B specifically with the understanding that
# MEC IC registers ARE at 0x5848-0x584B based on the kernel headers
# CP_MEC_ME1_IC_OP_CNTL might be at a different offset
log("\n  Checking 0x5848-0x584B as MEC IC (not CPC IC)...")
for off in range(0x5848, 0x584C):
    val = mm.r(off)
    log(f"  0x{off:04X} = 0x{val:08X}")

results['mec_rs64_scan'] = mec_rs64_scan
save("mec_rs64")

# === STEP 10: Final summary of ALL IC_OP_CNTL possibilities ===
log("\n--- Step 10: Exhaustive IC_OP_CNTL invalidate test ---")
# Try EVERY plausible offset with IC invalidate
all_candidates = [
    ('PFP_IC_OP_CNTL', 0x5843),
    ('ME_IC_OP_CNTL', 0x5847),
    ('MEC_0x584B', 0x584B),
    ('CPC_IC_OP_CNTL_CORRECT', 0x297A),
    ('WRONG_0x584F', 0x584F),
    ('MES_0x5853', 0x5853),
]

exhaustive = {}
for name, off in all_candidates:
    mm.w(off, 0x00)
    time.sleep(0.01)
    mm.w(off, 0x01)
    time.sleep(0.1)
    post = mm.r(off)
    inv_c = bool(post & 0x02)
    log(f"  {name:30s} (0x{off:04X}) = 0x{post:08X}  INV_COMPLETE={inv_c}")
    exhaustive[name] = {'offset': f"0x{off:04X}", 'result': f"0x{post:08X}", 'inv_complete': inv_c}

results['exhaustive_inv'] = exhaustive
save("exhaustive")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
