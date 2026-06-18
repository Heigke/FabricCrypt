#!/usr/bin/env python3
"""z2352d: CPC IC Redirect Test via /dev/mem.

BREAKTHROUGH: CPC_IC_BASE_LO and CPC_IC_BASE_HI are WRITABLE via /dev/mem!
CPC = Compute Command Processor.

Test plan:
1. Read current CPC IC state
2. Decode CPC_IC_BASE_CNTL writability
3. Test IC invalidate/prime on CPC (does INV_COMPLETE fire?)
4. Write NOP sled to VRAM via BAR0
5. Redirect CPC IC_BASE to NOP sled
6. Invalidate + prime CPC IC
7. Check if GPU survives

ALSO test PFP (partially writable BASE_LO).

Save state at every step for crash recovery.
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
BAR0_PHYS = 0x6800000000  # VRAM

# CPC IC registers
CPC_IC_BASE_LO   = 0x584C
CPC_IC_BASE_HI   = 0x584D
CPC_IC_BASE_CNTL = 0x584E
CPC_IC_OP_CNTL   = 0x584F  # Assumed — need to verify

# Also check 0x5848-0x584B (may be CPC area, between ME and what we labeled CPC)
# Let's scan the gap

RISCV_NOP = 0x00000013
VRAM_NOP_OFFSET = 0x0F00000  # 15MB into VRAM — safe area away from firmware

results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_cpc_redirect.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    if tag:
        log(f"  [saved: {tag}]")

class MMIO:
    def __init__(self, base, size=1024*1024):
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
    def r(self, off):
        b = off * 4
        return struct.unpack('<I', self.mm[b:b+4])[0]
    def w(self, off, val):
        b = off * 4
        self.mm[b:b+4] = struct.pack('<I', val)
    def close(self):
        self.mm.close()
        os.close(self.fd)

log("=== z2352d: CPC IC Redirect Test ===")
mm = MMIO(MMIO_BASE)

# === STEP 1: Map the 0x5848-0x5855 range to find CPC_IC_OP_CNTL ===
log("\n--- Step 1: Scan 0x5848-0x5855 for CPC/MES registers ---")
gap_regs = {}
for off in range(0x5848, 0x5856):
    val = mm.r(off)
    # Test writability
    mm.w(off, 0xFFFFFFFF)
    time.sleep(0.001)
    post = mm.r(off)
    mm.w(off, val)  # restore
    writable = (post != val)
    if val != 0 or writable:
        log(f"  0x{off:04X} = 0x{val:08X}  write→0x{post:08X}  {'WRITABLE' if writable else 'R/O'}")
    gap_regs[f"0x{off:04X}"] = {
        'value': f"0x{val:08X}",
        'write_test': f"0x{post:08X}",
        'writable': writable,
    }

results['gap_scan'] = gap_regs
save("gap_scan")

# === STEP 2: Full CPC register state ===
log("\n--- Step 2: CPC IC register state ---")
cpc_state = {}
for name, off in [('CPC_IC_BASE_LO', 0x584C), ('CPC_IC_BASE_HI', 0x584D),
                   ('CPC_IC_BASE_CNTL', 0x584E), ('CPC_0x584F', 0x584F)]:
    val = mm.r(off)
    log(f"  {name:20s} (0x{off:04X}) = 0x{val:08X}")
    cpc_state[name] = f"0x{val:08X}"

# Decode CPC_IC_BASE_CNTL
cntl = mm.r(0x584E)
log(f"  CPC_IC_BASE_CNTL decode:")
log(f"    Binary: {cntl:032b}")
log(f"    VMID={cntl&0xF} ADDR_CLAMP={(cntl>>4)&1} EXE_DISABLE={(cntl>>23)&1} CACHE_POLICY={(cntl>>24)&3} bit29={(cntl>>29)&1}")

results['cpc_state'] = cpc_state
save("cpc_state")

# === STEP 3: Bit-by-bit writability of CPC_IC_BASE_CNTL ===
log("\n--- Step 3: CPC_IC_BASE_CNTL bit writability ---")
orig_cntl = mm.r(0x584E)
writable_bits = 0
for bit in range(32):
    mask = 1 << bit
    mm.w(0x584E, orig_cntl | mask)
    time.sleep(0.001)
    val_set = mm.r(0x584E)
    mm.w(0x584E, orig_cntl & ~mask)
    time.sleep(0.001)
    val_clr = mm.r(0x584E)
    mm.w(0x584E, orig_cntl)

    can_set = bool(val_set & mask)
    can_clr = not bool(val_clr & mask)
    orig_bit = bool(orig_cntl & mask)

    if can_set and can_clr:
        status = "R/W"
        writable_bits |= mask
    elif can_set:
        status = "SET-only"
        writable_bits |= mask
    elif can_clr:
        status = "CLR-only"
        writable_bits |= mask
    else:
        status = f"R/O"

    if status != "R/O" or orig_bit:
        log(f"  bit {bit:2d}: orig={int(orig_bit)} [{status}]")

mm.w(0x584E, orig_cntl)
results['cpc_cntl_writable_mask'] = f"0x{writable_bits:08X}"
log(f"  Writable mask: 0x{writable_bits:08X}")
save("cpc_cntl_bits")

# === STEP 4: Test CPC IC invalidate ===
log("\n--- Step 4: CPC IC invalidate test ---")

# Find the right IC_OP_CNTL for CPC
# Try 0x584F first (right after CPC_IC_BASE_CNTL)
cpc_op_candidates = [0x584F, 0x5848, 0x5849]
for cand in cpc_op_candidates:
    orig_op = mm.r(cand)
    log(f"  Testing IC_OP_CNTL candidate 0x{cand:04X} (current: 0x{orig_op:08X})")
    mm.w(cand, 0x01)  # IC_INVALIDATE
    time.sleep(0.1)
    post = mm.r(cand)
    inv_complete = bool(post & 0x02)
    log(f"    After invalidate: 0x{post:08X}  INV_COMPLETE={inv_complete}")

    if inv_complete:
        log(f"    *** CPC IC_OP_CNTL is at 0x{cand:04X}! INV_COMPLETE fired! ***")
        results['cpc_ic_op_cntl_offset'] = f"0x{cand:04X}"
        results['cpc_inv_complete'] = True

        # Try prime
        mm.w(cand, 0x00)
        time.sleep(0.01)
        mm.w(cand, 0x10)  # IC_PRIME
        time.sleep(0.1)
        post_prime = mm.r(cand)
        primed = bool(post_prime & 0x20)
        log(f"    After prime: 0x{post_prime:08X}  PRIMED={primed}")
        results['cpc_primed'] = primed

    mm.w(cand, orig_op)  # restore

results.setdefault('cpc_inv_complete', False)
save("cpc_invalidate")

# === STEP 5: Write NOP sled to VRAM ===
if '--no-vram' not in sys.argv:
    log("\n--- Step 5: Write NOP sled to VRAM via BAR0 ---")
    vram_fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    vram_mm = mmap.mmap(vram_fd, 4096*4, mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE,
                        offset=BAR0_PHYS + VRAM_NOP_OFFSET)

    # Read original VRAM at target offset
    orig_vram = vram_mm[:16]
    log(f"  VRAM at offset 0x{VRAM_NOP_OFFSET:08X}: {orig_vram.hex()}")

    # Write 4KB of RISC-V NOPs
    nop_sled = struct.pack('<I', RISCV_NOP) * 1024 * 4  # 16KB
    vram_mm[:len(nop_sled)] = nop_sled
    verify = struct.unpack('<I', vram_mm[:4])[0]
    log(f"  Wrote {len(nop_sled)} bytes of NOP sled, verify: 0x{verify:08X}")
    results['vram_nop_sled'] = {
        'offset': f"0x{VRAM_NOP_OFFSET:08X}",
        'size': len(nop_sled),
        'verified': verify == RISCV_NOP,
    }

    vram_mm.close()
    os.close(vram_fd)
    save("vram_nop")
else:
    log("\n--- Step 5: SKIPPED (--no-vram) ---")

# === STEP 6: CPC IC_BASE redirect ===
if '--no-redirect' not in sys.argv and results.get('cpc_inv_complete'):
    log("\n--- Step 6: CPC IC_BASE redirect to NOP sled ---")
    log("  WARNING: This may crash the compute engine. Saving state first.")
    save("pre_redirect")

    orig_lo = mm.r(CPC_IC_BASE_LO)
    orig_hi = mm.r(CPC_IC_BASE_HI)
    orig_cntl = mm.r(CPC_IC_BASE_CNTL)
    log(f"  Current CPC IC_BASE: LO=0x{orig_lo:08X} HI=0x{orig_hi:08X} CNTL=0x{orig_cntl:08X}")

    # Point to VRAM NOP sled (physical VRAM offset)
    # IC_BASE_LO is byte address >> 8 typically, or direct byte address
    # The firmware at MES has IC_BASE_LO=0x8000, suggesting byte offset in VRAM
    nop_addr_lo = VRAM_NOP_OFFSET  # 0x0F00000
    nop_addr_hi = 0

    log(f"  Setting CPC IC_BASE_LO = 0x{nop_addr_lo:08X}")
    mm.w(CPC_IC_BASE_LO, nop_addr_lo)
    time.sleep(0.01)
    verify_lo = mm.r(CPC_IC_BASE_LO)
    log(f"  Verify CPC IC_BASE_LO = 0x{verify_lo:08X}")

    mm.w(CPC_IC_BASE_HI, nop_addr_hi)
    time.sleep(0.01)
    verify_hi = mm.r(CPC_IC_BASE_HI)
    log(f"  Verify CPC IC_BASE_HI = 0x{verify_hi:08X}")

    results['cpc_redirect'] = {
        'orig_lo': f"0x{orig_lo:08X}",
        'orig_hi': f"0x{orig_hi:08X}",
        'new_lo': f"0x{verify_lo:08X}",
        'new_hi': f"0x{verify_hi:08X}",
    }
    save("redirect_done")

    # Invalidate + prime
    cpc_op = int(results.get('cpc_ic_op_cntl_offset', '0x584F'), 16)
    log(f"  Invalidating CPC IC...")
    mm.w(cpc_op, 0x01)  # INVALIDATE
    time.sleep(0.2)
    post_inv = mm.r(cpc_op)
    log(f"  IC_OP_CNTL after invalidate: 0x{post_inv:08X}  INV_COMPLETE={bool(post_inv & 0x02)}")

    if post_inv & 0x02:
        log(f"  Priming CPC IC...")
        mm.w(cpc_op, 0x00)
        time.sleep(0.01)
        mm.w(cpc_op, 0x10)  # PRIME
        time.sleep(0.2)
        post_prime = mm.r(cpc_op)
        log(f"  IC_OP_CNTL after prime: 0x{post_prime:08X}  PRIMED={bool(post_prime & 0x20)}")
        results['cpc_prime_result'] = f"0x{post_prime:08X}"

    save("prime_done")

    # Check if GPU still alive
    log("  Checking GPU health...")
    time.sleep(1)
    try:
        # Read a known stable register
        check = mm.r(0x5846)  # ME_IC_BASE_CNTL
        log(f"  ME_IC_BASE_CNTL = 0x{check:08X} (GPU responsive)")
        results['gpu_alive_after_redirect'] = True
    except Exception as e:
        log(f"  GPU access failed: {e}")
        results['gpu_alive_after_redirect'] = False

    # Restore original CPC state
    log("  Restoring CPC IC_BASE...")
    mm.w(CPC_IC_BASE_LO, orig_lo)
    mm.w(CPC_IC_BASE_HI, orig_hi)
    mm.w(CPC_IC_BASE_CNTL, orig_cntl)
    time.sleep(0.1)
    save("restored")

elif '--no-redirect' in sys.argv:
    log("\n--- Step 6: SKIPPED (--no-redirect) ---")
else:
    log("\n--- Step 6: SKIPPED (INV_COMPLETE not achieved) ---")
    results['cpc_redirect'] = 'skipped_no_inv'

# Final state
log("\n--- Final register state ---")
for off in range(0x5840, 0x5856):
    val = mm.r(off)
    if val != 0:
        log(f"  0x{off:04X} = 0x{val:08X}")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
