#!/usr/bin/env python3
"""z2352g: ME Halt → IC_BASE redirect → unhalt test.

FINDING: 0x586C accepts bit 28 (ME_HALT pattern on older gens).
FINDING: MEC 0x5848 IC_BASE_LO is FULLY writable.

Strategy:
1. Halt ME via 0x586C bit 28
2. While halted, try writing ME_IC_BASE_LO (was read-only before)
3. Also try MEC redirect (already writable)
4. Test IC invalidate while halted
5. Unhalt and check behavior

Also: Deep PSP mailbox analysis.
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_halt_redirect.json'
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
log("=== z2352g: ME Halt + IC_BASE Redirect Test ===")

# === STEP 1: Characterize 0x586C fully ===
log("\n--- Step 1: Full characterization of 0x586C ---")
off = 0x586C
orig = mm.r(off)
log(f"  0x586C original: 0x{orig:08X} = {orig:032b}")

# Bit-by-bit writability
writable_mask = 0
for bit in range(32):
    mask = 1 << bit
    mm.w(off, orig | mask)
    val_set = mm.r(off)
    mm.w(off, orig & ~mask)
    val_clr = mm.r(off)
    mm.w(off, orig)
    
    can_set = bool(val_set & mask)
    can_clr = not bool(val_clr & mask)
    
    if can_set and can_clr:
        writable_mask |= mask
        status = "R/W"
    elif can_set:
        writable_mask |= mask
        status = "SET"
    elif can_clr:
        writable_mask |= mask
        status = "CLR"
    else:
        status = "R/O"
    
    if status != "R/O":
        log(f"  bit {bit:2d}: [{status}]")

log(f"  Writable mask: 0x{writable_mask:08X} = {writable_mask:032b}")
results['reg_586C'] = {
    'original': f"0x{orig:08X}",
    'writable_mask': f"0x{writable_mask:08X}",
}
save("586C_characterize")

# === STEP 2: Save pre-halt state of ALL IC registers ===
log("\n--- Step 2: Pre-halt IC state ---")
pre_halt = {}
ic_offsets = {
    'PFP_IC_BASE_LO': 0x5840, 'PFP_IC_BASE_HI': 0x5841,
    'PFP_IC_BASE_CNTL': 0x5842, 'PFP_IC_OP_CNTL': 0x5843,
    'ME_IC_BASE_LO': 0x5844, 'ME_IC_BASE_HI': 0x5845,
    'ME_IC_BASE_CNTL': 0x5846, 'ME_IC_OP_CNTL': 0x5847,
    'MEC_IC_BASE_LO': 0x5848, 'MEC_IC_BASE_HI': 0x5849,
    'MEC_IC_BASE_CNTL': 0x584A, 'MEC_IC_OP_CNTL': 0x584B,
    'CPC_IC_BASE_LO': 0x584C, 'CPC_IC_BASE_HI': 0x584D,
    'CPC_IC_BASE_CNTL': 0x584E, 'CPC_IC_OP_CNTL': 0x584F,
    'MES_IC_BASE_LO': 0x5850, 'MES_IC_BASE_HI': 0x5851,
    'MES_IC_BASE_CNTL': 0x5852, 'MES_IC_OP_CNTL': 0x5853,
}
for name, off in ic_offsets.items():
    val = mm.r(off)
    pre_halt[name] = f"0x{val:08X}"
    if val != 0:
        log(f"  {name:20s} = 0x{val:08X}")

results['pre_halt'] = pre_halt
save("pre_halt")

# === STEP 3: Set ME_HALT via 0x586C ===
log("\n--- Step 3: Setting ME_HALT (bit 28 of 0x586C) ---")
log("  WARNING: This may freeze graphics engine. Saving state first.")
save("pre_halt_write")

# Set bit 28 (ME_HALT) and bit 26 (PFP_HALT)
halt_val = (1 << 28) | (1 << 26)
log(f"  Writing 0x{halt_val:08X} to 0x586C (ME_HALT + PFP_HALT)")
mm.w(0x586C, halt_val)
time.sleep(0.1)
post_halt = mm.r(0x586C)
log(f"  0x586C after halt write: 0x{post_halt:08X}")
results['halt_write'] = {
    'wrote': f"0x{halt_val:08X}",
    'readback': f"0x{post_halt:08X}",
    'bit28_set': bool(post_halt & (1 << 28)),
    'bit26_set': bool(post_halt & (1 << 26)),
}
save("halt_write")

# === STEP 4: While halted, try writing ME_IC_BASE_LO ===
log("\n--- Step 4: Writing ME_IC_BASE_LO while ME halted ---")
orig_me_lo = int(pre_halt['ME_IC_BASE_LO'], 16)
test_val = 0xBEEF0000

log(f"  ME_IC_BASE_LO before: 0x{mm.r(0x5844):08X}")
mm.w(0x5844, test_val)
time.sleep(0.1)
post_write = mm.r(0x5844)
changed = (post_write != orig_me_lo)
log(f"  ME_IC_BASE_LO after write 0x{test_val:08X}: 0x{post_write:08X}  {'*** CHANGED ***' if changed else 'still locked'}")

results['me_lo_while_halted'] = {
    'original': f"0x{orig_me_lo:08X}",
    'wrote': f"0x{test_val:08X}",
    'readback': f"0x{post_write:08X}",
    'changed': changed,
}
save("me_halted_write")

# Also test ME_IC_BASE_HI
orig_me_hi = int(pre_halt['ME_IC_BASE_HI'], 16)
mm.w(0x5845, test_val)
time.sleep(0.1)
post_hi = mm.r(0x5845)
hi_changed = (post_hi != orig_me_hi)
log(f"  ME_IC_BASE_HI after write 0x{test_val:08X}: 0x{post_hi:08X}  {'*** CHANGED ***' if hi_changed else 'still locked'}")
results['me_hi_while_halted'] = {
    'original': f"0x{orig_me_hi:08X}",
    'wrote': f"0x{test_val:08X}",
    'readback': f"0x{post_hi:08X}",
    'changed': hi_changed,
}

# Try PFP too
orig_pfp_lo = int(pre_halt['PFP_IC_BASE_LO'], 16)
mm.w(0x5840, test_val)
time.sleep(0.1)
post_pfp = mm.r(0x5840)
pfp_changed = (post_pfp != orig_pfp_lo)
log(f"  PFP_IC_BASE_LO after write 0x{test_val:08X}: 0x{post_pfp:08X}  {'*** CHANGED ***' if pfp_changed else 'still locked'}")
results['pfp_lo_while_halted'] = {
    'original': f"0x{orig_pfp_lo:08X}",
    'wrote': f"0x{test_val:08X}",
    'readback': f"0x{post_pfp:08X}",
    'changed': pfp_changed,
}
save("halted_writes")

# === STEP 5: Try IC invalidate while halted ===
log("\n--- Step 5: IC invalidate while ME halted ---")
orig_op = mm.r(0x5847)
mm.w(0x5847, 0x01)  # INVALIDATE
time.sleep(0.2)
post_inv = mm.r(0x5847)
inv_complete = bool(post_inv & 0x02)
log(f"  ME_IC_OP_CNTL: orig=0x{orig_op:08X}  after inv=0x{post_inv:08X}  INV_COMPLETE={inv_complete}")

results['inv_while_halted'] = {
    'orig': f"0x{orig_op:08X}",
    'post': f"0x{post_inv:08X}",
    'inv_complete': inv_complete,
}
save("inv_halted")

# === STEP 6: Unhalt ===
log("\n--- Step 6: Unhalting ME ---")
mm.w(0x586C, 0)  # Clear halt bits
time.sleep(0.2)
post_unhalt = mm.r(0x586C)
log(f"  0x586C after clear: 0x{post_unhalt:08X}")

# Restore ME_IC_BASE_LO if we changed it
if changed:
    log(f"  Restoring ME_IC_BASE_LO to 0x{orig_me_lo:08X}")
    mm.w(0x5844, orig_me_lo)
if hi_changed:
    mm.w(0x5845, orig_me_hi)
if pfp_changed:
    mm.w(0x5840, orig_pfp_lo)

# Check all IC state after unhalt
log("\n--- Post-unhalt IC state ---")
post_unhalt_state = {}
for name, off in ic_offsets.items():
    val = mm.r(off)
    orig_val = pre_halt[name]
    changed_flag = (f"0x{val:08X}" != orig_val)
    if val != 0 or changed_flag:
        tag = " *** CHANGED ***" if changed_flag else ""
        log(f"  {name:20s} = 0x{val:08X}  (was {orig_val}){tag}")
    post_unhalt_state[name] = f"0x{val:08X}"

results['post_unhalt'] = post_unhalt_state
save("post_unhalt")

# === STEP 7: PSP mailbox deep analysis ===
log("\n--- Step 7: PSP mailbox 0x16000-0x160FF ---")
psp_map = {}
for off in range(0x16000, 0x16100):
    val = mm.r(off)
    if val != 0:
        # Quick writability
        mm.w(off, 0xFFFFFFFF)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        psp_map[f"0x{off:05X}"] = {
            'val': f"0x{val:08X}",
            'writable': writable,
        }
        if len(psp_map) <= 25:
            log(f"  0x{off:05X} = 0x{val:08X}  {'W' if writable else 'R/O'}")

results['psp_mailbox'] = psp_map
save("psp_detail")

# === STEP 8: Check MP0 C2PMSG registers ===
# C2PMSG (CPU-to-PSP messages) on GFX11 typically at 0x16000+
# The key registers for PSP command interface:
# MP0_C2PMSG_64 through MP0_C2PMSG_103
log("\n--- Step 8: MP0 C2PMSG register scan ---")
c2pmsg = {}
# MP0 offset base varies by IP version
# On RDNA4/GFX11 with IP 14.x, check MP0_BASE = 0x16000
# C2PMSG registers start at offset 0x50 from MP0_BASE
# So MP0_C2PMSG_64 = 0x16000 + 64 = 0x16040
# MP0_C2PMSG_66 = 0x16042 (NEVER write! SMU mailbox)
for i in range(40):
    off = 0x16040 + i
    val = mm.r(off)
    if val != 0:
        log(f"  C2PMSG_{64+i} (0x{off:05X}) = 0x{val:08X}")
        c2pmsg[f"C2PMSG_{64+i}"] = f"0x{val:08X}"

results['c2pmsg'] = c2pmsg
save("c2pmsg")

# === STEP 9: Check GRBM_STATUS and CP status ===
log("\n--- Step 9: GRBM/CP status check ---")
status_regs = {
    'GRBM_STATUS': 0x2004,
    'GRBM_STATUS2': 0x2002,
    'CP_STALLED_STAT1': 0x219C,
    'CP_BUSY_STAT': 0x219E,
}
for name, off in status_regs.items():
    val = mm.r(off)
    log(f"  {name:25s} = 0x{val:08X}")
    results[f'status_{name}'] = f"0x{val:08X}"

# Also check if the GPU is actually still alive after our halt/unhalt
log("\n--- Step 10: GPU liveness check ---")
try:
    # Read a known register that should have a non-zero value
    me_cntl = mm.r(0x5846)  # ME_IC_BASE_CNTL
    cpc_cntl = mm.r(0x584E)  # CPC_IC_BASE_CNTL
    log(f"  ME_IC_BASE_CNTL = 0x{me_cntl:08X}")
    log(f"  CPC_IC_BASE_CNTL = 0x{cpc_cntl:08X}")
    results['gpu_alive'] = True
    log(f"  GPU is alive!")
except Exception as e:
    log(f"  GPU access failed: {e}")
    results['gpu_alive'] = False

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
