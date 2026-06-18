#!/usr/bin/env python3
"""z2352b: Direct /dev/mem IC register probe and write test.

FOUND: GPU MMIO register aperture at physical 0xB4400000 (BAR5, 1MB).
Register address = 0xB4400000 + register_offset * 4.

This bypasses debugfs entirely — direct MMIO like the kernel driver's WREG32_SOC15().
Test if writes here reach the REAL IC hardware (not shadow registers).
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000  # GPU register aperture physical address

# IC register offsets (DWORDs) — multiply by 4 for byte offset
IC_REGS = {
    'PFP_IC_BASE_LO':   0x5840,
    'PFP_IC_BASE_HI':   0x5841,
    'PFP_IC_BASE_CNTL': 0x5842,
    'PFP_IC_OP_CNTL':   0x5843,
    'ME_IC_BASE_LO':    0x5844,
    'ME_IC_BASE_HI':    0x5845,
    'ME_IC_BASE_CNTL':  0x5846,
    'ME_IC_OP_CNTL':    0x5847,
    'CPC_IC_BASE_LO':   0x584C,
    'CPC_IC_BASE_HI':   0x584D,
    'CPC_IC_BASE_CNTL': 0x584E,
    'MES_IC_BASE_LO':   0x5850,
    'MES_IC_BASE_HI':   0x5851,
    'MES_IC_BASE_CNTL': 0x5852,
}

# Some RLC registers
RLC_REGS = {
    'RLC_CNTL':           0x4C00,
    'RLC_SAFE_MODE':      0x4C50,
    'RLC_CP_SCHEDULERS':  0x4CA4,
    'RLC_GPM_GENERAL_0':  0x4C80,
    'RLC_GPM_GENERAL_1':  0x4C81,
}

# CP status registers
CP_REGS = {
    'CP_ME_CNTL':        0x2186,
    'CP_MEC_CNTL':       0x2188,
    'CP_HQD_ACTIVE':     0x2358,
    'CP_RB_WPTR':        0x2014,
    'CP_INT_STATUS':     0x2170,
    'CP_STALLED_STAT1':  0x219C,
    'CP_STALLED_STAT2':  0x219D,
    'CP_BUSY_STAT':      0x219E,
    'GRBM_STATUS':       0x2004,
    'GRBM_STATUS2':      0x2002,
}

results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'mmio_base': f"0x{MMIO_BASE:08X}"}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_devmem_ic_probe.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    if tag:
        log(f"  [saved: {tag}]")

class MMIOAccess:
    """Direct MMIO register access via /dev/mem."""
    def __init__(self, base_phys, size=1024*1024):
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=base_phys)
        self.base = base_phys

    def read32(self, reg_offset):
        """Read register at DWORD offset."""
        byte_off = reg_offset * 4
        return struct.unpack('<I', self.mm[byte_off:byte_off+4])[0]

    def write32(self, reg_offset, value):
        """Write register at DWORD offset."""
        byte_off = reg_offset * 4
        self.mm[byte_off:byte_off+4] = struct.pack('<I', value)

    def close(self):
        self.mm.close()
        os.close(self.fd)

# Also read via debugfs for comparison
REGS_PATH = '/sys/kernel/debug/dri/0/amdgpu_regs'
def debugfs_read(off):
    with open(REGS_PATH, 'rb') as f:
        f.seek(off * 4)
        return struct.unpack('<I', f.read(4))[0]

log("=== z2352b: Direct /dev/mem IC Register Probe ===")
log(f"  MMIO base: 0x{MMIO_BASE:08X}")

mmio = MMIOAccess(MMIO_BASE)

# === STEP 1: Read ALL IC registers via both paths and compare ===
log("\n--- Step 1: Read IC registers (devmem vs debugfs) ---")
ic_state = {}
for name, off in IC_REGS.items():
    devmem_val = mmio.read32(off)
    debugfs_val = debugfs_read(off)
    match = (devmem_val == debugfs_val)
    tag = "" if match else " *** DIFFERENT ***"
    log(f"  {name:20s} (0x{off:04X}): devmem=0x{devmem_val:08X}  debugfs=0x{debugfs_val:08X}{tag}")
    ic_state[name] = {
        'offset': f"0x{off:04X}",
        'devmem': f"0x{devmem_val:08X}",
        'debugfs': f"0x{debugfs_val:08X}",
        'match': match,
    }
results['ic_registers'] = ic_state
save("ic_read")

# === STEP 2: Read RLC registers via devmem (were all zero via debugfs) ===
log("\n--- Step 2: Read RLC registers via /dev/mem ---")
rlc_state = {}
for name, off in RLC_REGS.items():
    devmem_val = mmio.read32(off)
    debugfs_val = debugfs_read(off)
    match = (devmem_val == debugfs_val)
    tag = "" if match else " *** DIFFERENT ***"
    log(f"  {name:25s} (0x{off:04X}): devmem=0x{devmem_val:08X}  debugfs=0x{debugfs_val:08X}{tag}")
    rlc_state[name] = {
        'devmem': f"0x{devmem_val:08X}",
        'debugfs': f"0x{debugfs_val:08X}",
        'match': match,
    }
results['rlc_registers'] = rlc_state
save("rlc_read")

# === STEP 3: Read CP status registers ===
log("\n--- Step 3: Read CP/GRBM status registers ---")
cp_state = {}
for name, off in CP_REGS.items():
    val = mmio.read32(off)
    log(f"  {name:25s} (0x{off:04X}): 0x{val:08X}")
    cp_state[name] = f"0x{val:08X}"
results['cp_status'] = cp_state
save("cp_status")

# === STEP 4: WRITE TEST — Try writing ME_IC_BASE_CNTL via /dev/mem ===
if '--no-write' not in sys.argv:
    log("\n--- Step 4: WRITE TEST — ME_IC_BASE_CNTL via /dev/mem ---")

    orig = mmio.read32(0x5846)  # ME_IC_BASE_CNTL
    log(f"  Original ME_IC_BASE_CNTL = 0x{orig:08X}")

    # Test 4a: Change VMID bits [3:0] from 15 to 0
    test_val = (orig & ~0xF) | 0  # VMID=0
    log(f"  Writing 0x{test_val:08X} (VMID=0) via /dev/mem...")
    mmio.write32(0x5846, test_val)
    time.sleep(0.1)

    # Read back via BOTH paths
    devmem_post = mmio.read32(0x5846)
    debugfs_post = debugfs_read(0x5846)
    devmem_changed = (devmem_post != orig)
    debugfs_changed = (debugfs_post != orig)

    log(f"  After write:")
    log(f"    devmem  readback: 0x{devmem_post:08X} {'CHANGED' if devmem_changed else 'unchanged'}")
    log(f"    debugfs readback: 0x{debugfs_post:08X} {'CHANGED' if debugfs_changed else 'unchanged'}")

    results['write_test_vmid'] = {
        'original': f"0x{orig:08X}",
        'wrote': f"0x{test_val:08X}",
        'devmem_readback': f"0x{devmem_post:08X}",
        'debugfs_readback': f"0x{debugfs_post:08X}",
        'devmem_changed': devmem_changed,
        'debugfs_changed': debugfs_changed,
    }
    save("write_vmid")

    # Restore original
    if devmem_changed:
        log(f"  Restoring original 0x{orig:08X}...")
        mmio.write32(0x5846, orig)
        time.sleep(0.1)
        restored = mmio.read32(0x5846)
        log(f"  Restored: 0x{restored:08X}")

    # Test 4b: Try IC_OP_CNTL invalidate via /dev/mem
    log(f"\n  Testing IC_OP_CNTL invalidate via /dev/mem...")
    orig_op = mmio.read32(0x5847)  # ME_IC_OP_CNTL
    log(f"  Original ME_IC_OP_CNTL = 0x{orig_op:08X}")

    mmio.write32(0x5847, 0x01)  # IC_INVALIDATE bit
    time.sleep(0.1)

    # Check INV_COMPLETE
    post_op = mmio.read32(0x5847)
    inv_complete = bool(post_op & 0x02)
    log(f"  After invalidate: ME_IC_OP_CNTL = 0x{post_op:08X}  INV_COMPLETE={inv_complete}")

    results['write_test_invalidate'] = {
        'original': f"0x{orig_op:08X}",
        'wrote': '0x00000001',
        'readback': f"0x{post_op:08X}",
        'inv_complete': inv_complete,
    }
    save("write_invalidate")

    if inv_complete:
        log("  *** INV_COMPLETE is HIGH — /dev/mem reaches REAL IC hardware! ***")

        # Try prime
        mmio.write32(0x5847, 0x00)  # Clear
        time.sleep(0.01)
        mmio.write32(0x5847, 0x10)  # IC_PRIME bit
        time.sleep(0.1)
        post_prime = mmio.read32(0x5847)
        primed = bool(post_prime & 0x20)
        log(f"  After prime: ME_IC_OP_CNTL = 0x{post_prime:08X}  PRIMED={primed}")
        results['write_test_prime'] = {
            'readback': f"0x{post_prime:08X}",
            'primed': primed,
        }
        save("write_prime")
    else:
        log("  INV_COMPLETE not set — same shadow behavior as debugfs")

    # Test 4c: Try ME_IC_BASE_LO (was read-only via debugfs)
    log(f"\n  Testing IC_BASE_LO write via /dev/mem...")
    orig_lo = mmio.read32(0x5844)  # ME_IC_BASE_LO
    log(f"  Original ME_IC_BASE_LO = 0x{orig_lo:08X}")

    test_lo = 0x0000F000  # Safe test value
    mmio.write32(0x5844, test_lo)
    time.sleep(0.1)
    post_lo = mmio.read32(0x5844)
    lo_changed = (post_lo != orig_lo)
    log(f"  After write 0x{test_lo:08X}: readback 0x{post_lo:08X} {'CHANGED' if lo_changed else 'unchanged'}")

    results['write_test_base_lo'] = {
        'original': f"0x{orig_lo:08X}",
        'wrote': f"0x{test_lo:08X}",
        'readback': f"0x{post_lo:08X}",
        'changed': lo_changed,
    }
    save("write_base_lo")

    # Restore if changed
    if lo_changed:
        log(f"  *** IC_BASE_LO is WRITABLE via /dev/mem! Restoring... ***")
        mmio.write32(0x5844, orig_lo)
        time.sleep(0.1)
        restored_lo = mmio.read32(0x5844)
        log(f"  Restored: 0x{restored_lo:08X}")

else:
    log("\n--- Step 4: SKIPPED (--no-write) ---")
    results['write_test'] = 'skipped'

# Final state
log("\n--- Final IC register state ---")
for name, off in IC_REGS.items():
    val = mmio.read32(off)
    log(f"  {name:20s} = 0x{val:08X}")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")

mmio.close()
log("\nDone.")
