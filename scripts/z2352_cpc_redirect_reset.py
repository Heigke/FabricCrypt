#!/usr/bin/env python3
"""z2352j: CPC IC_BASE redirect + GPU reset — firmware injection test.

BREAKTHROUGH: CPC_IC_BASE_LO survives GPU reset!

Test plan:
1. Write NOP sled + marker instructions to VRAM
2. Set CPC_IC_BASE_LO to NOP sled address
3. Before reset, record all IC state
4. Trigger GPU reset
5. After reset, check:
   - Did CPC_IC_BASE_LO persist? (already confirmed YES)
   - Did IC_OP_CNTL change? (indicates firmware load attempted)
   - Did CPC_IC_BASE_CNTL change? (initialization state)
   - Any evidence of execution from NOP sled?
6. Check if INV_COMPLETE now fires (maybe PSP unlocked it during reset)
7. Also test: set IC_BASE BEFORE reset, trigger reset, check post-reset INV behavior
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
BAR0_PHYS = 0x6800000000
VRAM_NOP_OFFSET = 0x0F00000  # 15MB into VRAM
RISCV_NOP = 0x00000013

results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_cpc_redirect_reset.json'
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
log("=== z2352j: CPC IC Redirect + GPU Reset Test ===")

# === STEP 1: Write NOP sled to VRAM ===
log("\n--- Step 1: Write NOP sled to VRAM ---")
vram_fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
vram_mm = mmap.mmap(vram_fd, 64*1024, mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=BAR0_PHYS + VRAM_NOP_OFFSET)

# Write 16KB of RISC-V NOPs followed by a distinctive marker
nop_data = struct.pack('<I', RISCV_NOP) * (16 * 1024 // 4)
vram_mm[:len(nop_data)] = nop_data

# Write marker at end: ECALL (0x00000073) which traps to supervisor
# This would be noticeable if CPC executes it
marker = struct.pack('<I', 0x00000073)  # ECALL
vram_mm[len(nop_data):len(nop_data)+4] = marker

verify = struct.unpack('<I', vram_mm[:4])[0]
verify_end = struct.unpack('<I', vram_mm[len(nop_data):len(nop_data)+4])[0]
log(f"  NOP sled: {len(nop_data)} bytes at VRAM+0x{VRAM_NOP_OFFSET:08X}")
log(f"  Verify start: 0x{verify:08X} (expect NOP=0x{RISCV_NOP:08X})")
log(f"  Verify marker: 0x{verify_end:08X} (expect ECALL=0x00000073)")
results['vram_nop'] = {'verified': verify == RISCV_NOP, 'marker': verify_end == 0x73}
vram_mm.close()
os.close(vram_fd)
save("vram")

# === STEP 2: Record full pre-reset state ===
log("\n--- Step 2: Full pre-reset IC state ---")
all_regs = {}
for off in range(0x5840, 0x5860):
    val = mm.r(off)
    all_regs[f"0x{off:04X}"] = f"0x{val:08X}"
    if val != 0:
        log(f"  0x{off:04X} = 0x{val:08X}")

results['pre_reset_all'] = all_regs
save("pre_reset")

# === STEP 3: Set CPC IC_BASE to NOP sled ===
log("\n--- Step 3: Setting CPC IC_BASE to NOP sled ---")
# CPC_IC_BASE_LO = VRAM offset for the NOP sled
# Note: IC_BASE_LO value might be byte address or DWORD address
# MES_IC_BASE_LO = 0x8000, which is likely byte address (32KB into VRAM)
# So we use byte address for our NOP sled too
mm.w(0x584C, VRAM_NOP_OFFSET)  # CPC_IC_BASE_LO
time.sleep(0.01)
verify_lo = mm.r(0x584C)
log(f"  CPC_IC_BASE_LO = 0x{verify_lo:08X} (target: 0x{VRAM_NOP_OFFSET:08X})")

# Keep CPC_IC_BASE_HI the same (0x00000200)
# Keep CPC_IC_BASE_CNTL the same (0x20000000)

results['cpc_redirect'] = {
    'base_lo': f"0x{verify_lo:08X}",
    'base_hi': f"0x{mm.r(0x584D):08X}",
    'base_cntl': f"0x{mm.r(0x584E):08X}",
    'op_cntl': f"0x{mm.r(0x584F):08X}",
}
save("redirect_set")

# === STEP 4: Trigger GPU reset ===
log("\n--- Step 4: Triggering GPU reset ---")
log("  (saving state before reset)")
save("pre_reset_final")

mm.close()  # Close MMIO before reset

log("  Resetting GPU...")
with open('/sys/class/drm/card0/device/reset', 'w') as f:
    f.write('1')
time.sleep(3)
log("  GPU reset triggered, waiting 3s for recovery...")

# Reopen MMIO
mm = MMIO()

# === STEP 5: Post-reset analysis ===
log("\n--- Step 5: Post-reset IC state ---")
post_reset = {}
for off in range(0x5840, 0x5860):
    val = mm.r(off)
    pre = all_regs.get(f"0x{off:04X}", "N/A")
    changed = f"0x{val:08X}" != pre
    post_reset[f"0x{off:04X}"] = f"0x{val:08X}"
    if val != 0 or changed:
        tag = " *** CHANGED ***" if changed else ""
        log(f"  0x{off:04X} = 0x{val:08X}  (was {pre}){tag}")

results['post_reset_all'] = post_reset
save("post_reset")

# Check CPC specifically
cpc_lo = mm.r(0x584C)
cpc_hi = mm.r(0x584D)
cpc_cntl = mm.r(0x584E)
cpc_op = mm.r(0x584F)
log(f"\n  CPC IC state after reset:")
log(f"    CPC_IC_BASE_LO = 0x{cpc_lo:08X} {'*** SURVIVED ***' if cpc_lo == VRAM_NOP_OFFSET else 'RESET'}")
log(f"    CPC_IC_BASE_HI = 0x{cpc_hi:08X}")
log(f"    CPC_IC_BASE_CNTL = 0x{cpc_cntl:08X}")
log(f"    CPC_IC_OP_CNTL = 0x{cpc_op:08X}")
results['cpc_survived'] = (cpc_lo == VRAM_NOP_OFFSET)
save("cpc_check")

# === STEP 6: Try IC invalidate/prime NOW (post-reset, maybe PSP unlocked it) ===
log("\n--- Step 6: Post-reset IC invalidate test ---")

# Test on CPC (whose IC_BASE points to NOP sled)
mm.w(0x584F, 0x00)
time.sleep(0.01)
mm.w(0x584F, 0x01)  # INVALIDATE
time.sleep(0.5)
post_inv = mm.r(0x584F)
inv_complete = bool(post_inv & 0x02)
log(f"  CPC IC_OP_CNTL after invalidate: 0x{post_inv:08X}  INV_COMPLETE={inv_complete}")

if inv_complete:
    log("  *** INV_COMPLETE FIRES POST-RESET! ***")
    # Try prime
    mm.w(0x584F, 0x00)
    time.sleep(0.01)
    mm.w(0x584F, 0x10)  # PRIME
    time.sleep(0.5)
    post_prime = mm.r(0x584F)
    primed = bool(post_prime & 0x20)
    log(f"  CPC IC_OP_CNTL after prime: 0x{post_prime:08X}  PRIMED={primed}")
    results['post_reset_inv_complete'] = True
    results['post_reset_primed'] = primed
else:
    log("  INV_COMPLETE still not firing post-reset")
    results['post_reset_inv_complete'] = False

# Also test on ME
mm.w(0x5847, 0x00)
time.sleep(0.01)
mm.w(0x5847, 0x01)
time.sleep(0.5)
me_inv = mm.r(0x5847)
me_inv_complete = bool(me_inv & 0x02)
log(f"  ME IC_OP_CNTL after invalidate: 0x{me_inv:08X}  INV_COMPLETE={me_inv_complete}")
results['me_post_reset_inv'] = f"0x{me_inv:08X}"
save("inv_test")

# === STEP 7: Check GRBM/CP status for any anomalies ===
log("\n--- Step 7: Post-reset status ---")
for name, off in [('GRBM_STATUS', 0x2004), ('GRBM_STATUS2', 0x2002)]:
    val = mm.r(off)
    log(f"  {name} = 0x{val:08X}")

# Restore CPC_IC_BASE_LO to original (0x00000000)
log("\n--- Restoring CPC_IC_BASE_LO to 0 ---")
mm.w(0x584C, 0)
time.sleep(0.01)
log(f"  CPC_IC_BASE_LO = 0x{mm.r(0x584C):08X}")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
