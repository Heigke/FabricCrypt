#!/usr/bin/env python3
"""z2350: Test IC invalidate+prime on ME engine.

SAFE TEST: Does NOT change IC_BASE or firmware.
Just invalidates ME instruction cache and re-primes it from same address.
If ME continues working afterward, the invalidate+prime mechanism works.

Then tests PM4 NOP dispatch to verify ME is still alive.

Must run as root.
"""
import struct, time, os, json

REGS = '/sys/kernel/debug/dri/0/amdgpu_regs'

def read_reg(off):
    with open(REGS, 'rb') as f:
        f.seek(off * 4)
        return struct.unpack('<I', f.read(4))[0]

def write_reg(off, val):
    with open(REGS, 'r+b') as f:
        f.seek(off * 4)
        f.write(struct.pack('<I', val))
        f.flush()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

ME_IC_BASE_LO   = 0x5844
ME_IC_BASE_HI   = 0x5845
ME_IC_BASE_CNTL = 0x5846
ME_IC_OP_CNTL   = 0x5847

# IC_OP_CNTL bits (from z2347 decode)
IC_INVALIDATE    = 1 << 0  # bit 0: invalidate cache
IC_INV_COMPLETE  = 1 << 1  # bit 1: invalidation complete (status, read)
IC_PRIME         = 1 << 4  # bit 4: prime/reload cache
IC_PRIMED        = 1 << 5  # bit 5: cache primed (status, read)

def main():
    results = {}

    log("=== z2350: ME IC Invalidate+Prime Test (SAFE) ===")

    # Read current state
    base_lo = read_reg(ME_IC_BASE_LO)
    base_hi = read_reg(ME_IC_BASE_HI)
    base_cntl = read_reg(ME_IC_BASE_CNTL)
    op_cntl = read_reg(ME_IC_OP_CNTL)

    log(f"  IC_BASE_LO   = 0x{base_lo:08X}")
    log(f"  IC_BASE_HI   = 0x{base_hi:08X}")
    log(f"  IC_BASE_CNTL = 0x{base_cntl:08X}")
    log(f"  IC_OP_CNTL   = 0x{op_cntl:08X}")

    results['pre_state'] = {
        'IC_BASE_LO': f"0x{base_lo:08X}",
        'IC_BASE_HI': f"0x{base_hi:08X}",
        'IC_BASE_CNTL': f"0x{base_cntl:08X}",
        'IC_OP_CNTL': f"0x{op_cntl:08X}",
    }

    # Step 1: Invalidate ME IC
    log("\n=== Step 1: Invalidate ME IC ===")
    write_reg(ME_IC_OP_CNTL, IC_INVALIDATE)
    time.sleep(0.1)

    # Check if invalidation completed
    op_after_inv = read_reg(ME_IC_OP_CNTL)
    inv_complete = bool(op_after_inv & IC_INV_COMPLETE)
    log(f"  IC_OP_CNTL after invalidate = 0x{op_after_inv:08X}")
    log(f"  INV_COMPLETE = {inv_complete}")
    results['invalidate'] = {
        'op_cntl': f"0x{op_after_inv:08X}",
        'inv_complete': inv_complete,
    }

    # Step 2: Prime ME IC (reload from same address)
    log("\n=== Step 2: Prime ME IC ===")
    write_reg(ME_IC_OP_CNTL, IC_PRIME)
    time.sleep(0.1)

    # Check if priming completed
    op_after_prime = read_reg(ME_IC_OP_CNTL)
    primed = bool(op_after_prime & IC_PRIMED)
    log(f"  IC_OP_CNTL after prime = 0x{op_after_prime:08X}")
    log(f"  ICACHE_PRIMED = {primed}")
    results['prime'] = {
        'op_cntl': f"0x{op_after_prime:08X}",
        'primed': primed,
    }

    # Step 3: Verify ME is still alive by reading registers
    log("\n=== Step 3: Verify ME alive ===")
    post_cntl = read_reg(ME_IC_BASE_CNTL)
    post_lo = read_reg(ME_IC_BASE_LO)
    log(f"  IC_BASE_CNTL = 0x{post_cntl:08X} (was 0x{base_cntl:08X})")
    log(f"  IC_BASE_LO   = 0x{post_lo:08X} (was 0x{base_lo:08X})")

    # Check GPU status
    try:
        with open('/sys/class/drm/card0/device/gpu_busy_percent', 'r') as f:
            busy = f.read().strip()
        log(f"  GPU busy: {busy}%")
    except:
        log("  GPU busy: unknown")

    try:
        with open('/sys/class/drm/card0/device/current_link_speed', 'r') as f:
            link = f.read().strip()
        log(f"  PCIe link: {link}")
    except:
        pass

    results['post_state'] = {
        'IC_BASE_CNTL': f"0x{post_cntl:08X}",
        'IC_BASE_LO': f"0x{post_lo:08X}",
        'me_alive': (post_cntl == base_cntl),
    }

    log(f"\n  ME status: {'ALIVE' if post_cntl == base_cntl else 'CHANGED/CRASHED'}")

    # Step 4: Test IC_BASE_CNTL bit manipulation (non-destructive)
    # Try clearing and setting individual bits to understand which ones matter
    log("\n=== Step 4: IC_BASE_CNTL bit exploration ===")

    # Try setting EXE_DISABLE (bit 23) — this should stop ME execution
    log("  Testing EXE_DISABLE (bit 23)...")
    orig_cntl = read_reg(ME_IC_BASE_CNTL)
    new_cntl = orig_cntl | (1 << 23)  # Set EXE_DISABLE
    write_reg(ME_IC_BASE_CNTL, new_cntl)
    time.sleep(0.1)
    check = read_reg(ME_IC_BASE_CNTL)
    exe_disabled = bool(check & (1 << 23))
    log(f"    Wrote 0x{new_cntl:08X}, read 0x{check:08X}")
    log(f"    EXE_DISABLE stuck: {exe_disabled}")

    # Immediately restore
    write_reg(ME_IC_BASE_CNTL, orig_cntl)
    time.sleep(0.1)
    restored = read_reg(ME_IC_BASE_CNTL)
    log(f"    Restored: 0x{restored:08X} {'OK' if restored == orig_cntl else 'MISMATCH'}")

    results['exe_disable_test'] = {
        'written': f"0x{new_cntl:08X}",
        'readback': f"0x{check:08X}",
        'exe_disable_took': exe_disabled,
        'restored': (restored == orig_cntl),
    }

    # Try bit 29 (set in PFP/CPC but not ME — might be important)
    log("  Testing bit 29...")
    new_cntl = orig_cntl | (1 << 29)
    write_reg(ME_IC_BASE_CNTL, new_cntl)
    time.sleep(0.1)
    check = read_reg(ME_IC_BASE_CNTL)
    bit29_set = bool(check & (1 << 29))
    log(f"    Wrote 0x{new_cntl:08X}, read 0x{check:08X}")
    log(f"    bit[29] stuck: {bit29_set}")
    write_reg(ME_IC_BASE_CNTL, orig_cntl)
    time.sleep(0.01)

    # Try clearing bit 12 (set in ME, unknown function)
    log("  Testing clear bit 12...")
    new_cntl = orig_cntl & ~(1 << 12)
    write_reg(ME_IC_BASE_CNTL, new_cntl)
    time.sleep(0.1)
    check = read_reg(ME_IC_BASE_CNTL)
    bit12_cleared = not bool(check & (1 << 12))
    log(f"    Wrote 0x{new_cntl:08X}, read 0x{check:08X}")
    log(f"    bit[12] cleared: {bit12_cleared}")
    write_reg(ME_IC_BASE_CNTL, orig_cntl)
    time.sleep(0.01)

    # Final state check
    log("\n=== Final state ===")
    final_cntl = read_reg(ME_IC_BASE_CNTL)
    final_op = read_reg(ME_IC_OP_CNTL)
    log(f"  IC_BASE_CNTL = 0x{final_cntl:08X}")
    log(f"  IC_OP_CNTL   = 0x{final_op:08X}")
    log(f"  State preserved: {final_cntl == base_cntl}")

    results['final'] = {
        'IC_BASE_CNTL': f"0x{final_cntl:08X}",
        'IC_OP_CNTL': f"0x{final_op:08X}",
        'preserved': (final_cntl == base_cntl),
    }

    # Save results
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2350_ic_prime_test.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {outpath}")

if __name__ == "__main__":
    main()
