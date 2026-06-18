#!/usr/bin/env python3
"""z2352l: MEC IC Reload Vector Search.

MEC is the most promising engine:
  - IC_BASE_LO/HI/CNTL fully writable
  - bit29=0 (plaintext fetch mode)
  - But IC_OP_CNTL invalidate won't fire

Search for ALTERNATIVE ways to trigger MEC IC reload:
1. CP_MEC_CNTL halt/unhalt cycle (force re-fetch on resume)
2. CP_MEC_ME1_UCODE_ADDR reset (legacy trigger)
3. GRBM soft reset of MEC block
4. RLC-assisted MEC restart
5. SDMA-assisted IC fill
6. Race condition during GPU reset — snapshot register changes
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_mec_reload_vectors.json'
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
log("=== z2352l: MEC IC Reload Vector Search ===")

# MEC IC registers (from z2352 series)
MEC_IC_BASE_LO   = 0x5848
MEC_IC_BASE_HI   = 0x5849
MEC_IC_BASE_CNTL = 0x584A
MEC_IC_OP_CNTL   = 0x584B

# === STEP 1: Baseline MEC state ===
log("\n--- Step 1: MEC baseline state ---")
mec_baseline = {}
for name, off in [('MEC_IC_BASE_LO', MEC_IC_BASE_LO), ('MEC_IC_BASE_HI', MEC_IC_BASE_HI),
                   ('MEC_IC_BASE_CNTL', MEC_IC_BASE_CNTL), ('MEC_IC_OP_CNTL', MEC_IC_OP_CNTL)]:
    val = mm.r(off)
    log(f"  {name:20s} = 0x{val:08X}")
    mec_baseline[name] = f"0x{val:08X}"

# Also read CP_MEC_CNTL and related control registers
cp_mec_cntl = mm.r(0x2188)  # CP_MEC_CNTL
log(f"  CP_MEC_CNTL (0x2188) = 0x{cp_mec_cntl:08X}")
mec_baseline['CP_MEC_CNTL'] = f"0x{cp_mec_cntl:08X}"

# GRBM_STATUS2 — has MEC busy bits
grbm2 = mm.r(0x2002)
log(f"  GRBM_STATUS2 (0x2002) = 0x{grbm2:08X}")
mec_baseline['GRBM_STATUS2'] = f"0x{grbm2:08X}"

results['mec_baseline'] = mec_baseline
save("baseline")

# === STEP 2: CP_MEC_CNTL writability deep analysis ===
log("\n--- Step 2: CP_MEC_CNTL (0x2188) bit analysis ---")
orig_mec_cntl = mm.r(0x2188)
mec_cntl_bits = {}
writable_mask = 0
for bit in range(32):
    mask = 1 << bit
    mm.w(0x2188, orig_mec_cntl | mask)
    time.sleep(0.001)
    val_set = mm.r(0x2188)
    mm.w(0x2188, orig_mec_cntl & ~mask)
    time.sleep(0.001)
    val_clr = mm.r(0x2188)
    mm.w(0x2188, orig_mec_cntl)

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

    if status != "R/O" or (orig_mec_cntl & mask):
        log(f"  bit {bit:2d}: [{status}]")
        mec_cntl_bits[f"bit{bit}"] = status

mm.w(0x2188, orig_mec_cntl)
results['cp_mec_cntl_bits'] = mec_cntl_bits
results['cp_mec_cntl_writable_mask'] = f"0x{writable_mask:08X}"
log(f"  CP_MEC_CNTL writable mask: 0x{writable_mask:08X}")
save("mec_cntl_bits")

# === STEP 3: Try MEC halt/unhalt cycle to trigger IC re-fetch ===
log("\n--- Step 3: MEC halt/unhalt IC reload test ---")

# First set MEC_IC_BASE_LO to a known test pattern
orig_mec_lo = mm.r(MEC_IC_BASE_LO)
test_addr = 0x00F00000  # Our NOP sled location
mm.w(MEC_IC_BASE_LO, test_addr)
verify = mm.r(MEC_IC_BASE_LO)
log(f"  Set MEC_IC_BASE_LO = 0x{test_addr:08X}, verify = 0x{verify:08X}")

# On GFX11, CP_MEC_CNTL bit layout:
#   bit 28: MEC_ME1_HALT
#   bit 29: MEC_ME2_HALT
#   bit 30: MEC_ME1_STEP (single-step)
# Try setting halt bits if writable
halt_results = {}

if writable_mask & (1 << 28):
    log("  CP_MEC_CNTL bit 28 (MEC_ME1_HALT) is writable!")

    # Read pre-halt IC_OP_CNTL
    pre_halt_op = mm.r(MEC_IC_OP_CNTL)
    log(f"  Pre-halt MEC_IC_OP_CNTL = 0x{pre_halt_op:08X}")

    # Halt MEC_ME1
    mm.w(0x2188, orig_mec_cntl | (1 << 28))
    time.sleep(0.1)
    halted_status = mm.r(0x2002)  # GRBM_STATUS2
    log(f"  After halt: GRBM_STATUS2 = 0x{halted_status:08X}")

    # While halted, try IC invalidate
    mm.w(MEC_IC_OP_CNTL, 0x01)
    time.sleep(0.1)
    op_after_inv = mm.r(MEC_IC_OP_CNTL)
    inv_complete_halted = bool(op_after_inv & 0x02)
    log(f"  IC invalidate while halted: 0x{op_after_inv:08X} INV_COMPLETE={inv_complete_halted}")

    # Try multiple invalidate bit positions
    for inv_bit in [0, 1, 4, 8, 16]:
        mm.w(MEC_IC_OP_CNTL, 0)
        time.sleep(0.01)
        mm.w(MEC_IC_OP_CNTL, 1 << inv_bit)
        time.sleep(0.05)
        op_val = mm.r(MEC_IC_OP_CNTL)
        log(f"    INV bit {inv_bit}: op=0x{op_val:08X}")

    # Unhalt
    mm.w(0x2188, orig_mec_cntl & ~(1 << 28))
    time.sleep(0.1)
    unhalted_status = mm.r(0x2002)
    log(f"  After unhalt: GRBM_STATUS2 = 0x{unhalted_status:08X}")

    # Check if IC_OP_CNTL changed after unhalt (maybe re-fetch happens on resume)
    post_unhalt_op = mm.r(MEC_IC_OP_CNTL)
    log(f"  Post-unhalt MEC_IC_OP_CNTL = 0x{post_unhalt_op:08X}")

    halt_results['halt_supported'] = True
    halt_results['inv_complete_while_halted'] = inv_complete_halted
    halt_results['grbm2_halted'] = f"0x{halted_status:08X}"
    halt_results['grbm2_unhalted'] = f"0x{unhalted_status:08X}"
    halt_results['post_unhalt_op'] = f"0x{post_unhalt_op:08X}"
else:
    log("  CP_MEC_CNTL bit 28 NOT writable — cannot halt MEC")
    halt_results['halt_supported'] = False

# Restore MEC_IC_BASE_LO
mm.w(MEC_IC_BASE_LO, orig_mec_lo)
mm.w(0x2188, orig_mec_cntl)
results['halt_unhalt_test'] = halt_results
save("halt_unhalt")

# === STEP 4: GRBM soft reset of compute block ===
log("\n--- Step 4: GRBM soft reset probe ---")
# GRBM_SOFT_RESET is at 0x2008 on older gens
grbm_soft_candidates = [0x2008, 0x2038, 0x2058]
soft_reset_results = {}

for off in grbm_soft_candidates:
    val = mm.r(off)
    # Test writability
    mm.w(off, 0xFFFFFFFF)
    time.sleep(0.001)
    post = mm.r(off)
    mm.w(off, val)  # restore
    writable = (post != val)
    if val != 0 or writable:
        log(f"  0x{off:04X} = 0x{val:08X}  write→0x{post:08X}  {'WRITABLE' if writable else 'R/O'}")
    soft_reset_results[f"0x{off:04X}"] = {
        'value': f"0x{val:08X}",
        'post_write': f"0x{post:08X}",
        'writable': writable,
    }

# Also check GRBM_SOFT_RESET bit-by-bit if writable
grbm_sr = 0x2008
orig_sr = mm.r(grbm_sr)
sr_writable = 0
for bit in range(32):
    mask = 1 << bit
    mm.w(grbm_sr, orig_sr | mask)
    time.sleep(0.001)
    v = mm.r(grbm_sr)
    mm.w(grbm_sr, orig_sr)
    if v & mask:
        sr_writable |= mask
        # On GFX11, bit 7 = SOFT_RESET_CP, bit 17 = SOFT_RESET_CPF
        # bit 12 = SOFT_RESET_GFX, bit 20 = SOFT_RESET_CPG
        names = {7: 'SOFT_RESET_CP', 12: 'SOFT_RESET_GFX', 17: 'SOFT_RESET_CPF', 20: 'SOFT_RESET_CPG'}
        label = names.get(bit, f'bit{bit}')
        log(f"  GRBM_SOFT_RESET bit {bit} ({label}) settable")

soft_reset_results['writable_mask'] = f"0x{sr_writable:08X}"
mm.w(grbm_sr, orig_sr)
results['grbm_soft_reset'] = soft_reset_results
save("soft_reset")

# === STEP 5: Scan for MEC-specific control registers in 0x2180-0x21FF ===
log("\n--- Step 5: MEC control register scan 0x2180-0x2200 ---")
mec_ctrl_regs = {}
for off in range(0x2180, 0x2200):
    val = mm.r(off)
    if val != 0:
        # Quick writability
        mm.w(off, 0xFFFFFFFF)
        time.sleep(0.001)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        log(f"  0x{off:04X} = 0x{val:08X}  {'W' if writable else 'R'}")
        mec_ctrl_regs[f"0x{off:04X}"] = {
            'value': f"0x{val:08X}",
            'writable': writable,
        }

results['mec_ctrl_regs'] = mec_ctrl_regs
save("mec_ctrl")

# === STEP 6: Search for RLC scratch registers that control CP restart ===
log("\n--- Step 6: RLC scratch/GPM registers scan ---")
# RLC_GPM_GENERAL_0..15 and RLC scratch are sometimes used for inter-block control
rlc_gpm = {}
for off in range(0x4C80, 0x4C90):
    val = mm.r(off)
    if val != 0:
        mm.w(off, 0xDEAD0000 | (off & 0xFF))
        time.sleep(0.001)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        log(f"  RLC_GPM_{off-0x4C80:02d} (0x{off:04X}) = 0x{val:08X}  {'W' if writable else 'R'}")
        rlc_gpm[f"0x{off:04X}"] = {'value': f"0x{val:08X}", 'writable': writable}

# RLC scratch registers (0x4CA0-0x4CBF on some gens)
for off in range(0x4CA0, 0x4CC0):
    val = mm.r(off)
    if val != 0:
        mm.w(off, 0xFFFFFFFF)
        time.sleep(0.001)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        log(f"  RLC_SCRATCH_{off-0x4CA0:02d} (0x{off:04X}) = 0x{val:08X}  {'W' if writable else 'R'}")
        rlc_gpm[f"0x{off:04X}"] = {'value': f"0x{val:08X}", 'writable': writable}

results['rlc_gpm_scratch'] = rlc_gpm
save("rlc_gpm")

# === STEP 7: Snapshot registers during GPU reset ===
log("\n--- Step 7: Pre-reset register snapshot ---")
# Save all critical registers before we do a reset race test
pre_reset_snap = {}
critical_offsets = [
    ('GRBM_STATUS', 0x2004),
    ('GRBM_STATUS2', 0x2002),
    ('CP_ME_CNTL', 0x2186),
    ('CP_MEC_CNTL', 0x2188),
    ('MEC_IC_BASE_LO', MEC_IC_BASE_LO),
    ('MEC_IC_BASE_HI', MEC_IC_BASE_HI),
    ('MEC_IC_BASE_CNTL', MEC_IC_BASE_CNTL),
    ('MEC_IC_OP_CNTL', MEC_IC_OP_CNTL),
    ('CPC_IC_BASE_LO', 0x584C),
    ('CPC_IC_BASE_HI', 0x584D),
    ('ME_IC_BASE_LO', 0x5844),
    ('PFP_IC_BASE_LO', 0x5840),
    ('MES_IC_BASE_LO', 0x5850),
]
for name, off in critical_offsets:
    val = mm.r(off)
    pre_reset_snap[name] = f"0x{val:08X}"
    log(f"  {name:20s} = 0x{val:08X}")

results['pre_reset_snapshot'] = pre_reset_snap
save("pre_reset")

# === STEP 8: Set MEC_IC_BASE_LO, then do GPU reset and race to read ===
log("\n--- Step 8: MEC IC redirect + GPU reset race ---")
# Set MEC_IC_BASE_LO to NOP sled
orig_mec_lo = mm.r(MEC_IC_BASE_LO)
nop_addr = 0x00F00000
mm.w(MEC_IC_BASE_LO, nop_addr)
verify = mm.r(MEC_IC_BASE_LO)
log(f"  Set MEC_IC_BASE_LO = 0x{nop_addr:08X}, verify = 0x{verify:08X}")
save("mec_redirect_set")

# Trigger GPU MODE2 reset and IMMEDIATELY start polling
log("  Triggering GPU MODE2 reset...")
reset_time = time.time()
try:
    with open('/sys/class/drm/card0/device/reset', 'w') as f:
        f.write('1')
except Exception as e:
    log(f"  Reset trigger: {e}")

# Rapid polling during reset — try to catch the re-initialization window
race_samples = []
for i in range(200):
    t = time.time() - reset_time
    try:
        mec_lo = mm.r(MEC_IC_BASE_LO)
        mec_op = mm.r(MEC_IC_OP_CNTL)
        grbm = mm.r(0x2004)
        race_samples.append({
            't_ms': round(t * 1000, 1),
            'mec_lo': f"0x{mec_lo:08X}",
            'mec_op': f"0x{mec_op:08X}",
            'grbm': f"0x{grbm:08X}",
        })
        if i < 10 or i % 20 == 0:
            log(f"  t={t*1000:.1f}ms MEC_LO=0x{mec_lo:08X} OP=0x{mec_op:08X} GRBM=0x{grbm:08X}")
    except Exception as e:
        race_samples.append({'t_ms': round(t * 1000, 1), 'error': str(e)})
        if i < 5:
            log(f"  t={t*1000:.1f}ms ERROR: {e}")
    time.sleep(0.005)  # 5ms between samples

log(f"  Collected {len(race_samples)} samples over {(time.time()-reset_time)*1000:.0f}ms")

# Analyze: did MEC_IC_BASE_LO change during reset?
mec_lo_values = set()
mec_op_values = set()
for s in race_samples:
    if 'mec_lo' in s:
        mec_lo_values.add(s['mec_lo'])
    if 'mec_op' in s:
        mec_op_values.add(s['mec_op'])

log(f"  Unique MEC_IC_BASE_LO values: {mec_lo_values}")
log(f"  Unique MEC_IC_OP_CNTL values: {mec_op_values}")

# Check if driver restored original MEC_IC_BASE_LO
time.sleep(2)
post_reset_lo = mm.r(MEC_IC_BASE_LO)
log(f"  Post-reset MEC_IC_BASE_LO = 0x{post_reset_lo:08X} (was 0x{nop_addr:08X}, orig 0x{orig_mec_lo:08X})")

results['reset_race'] = {
    'set_value': f"0x{nop_addr:08X}",
    'sample_count': len(race_samples),
    'unique_mec_lo': list(mec_lo_values),
    'unique_mec_op': list(mec_op_values),
    'post_reset_lo': f"0x{post_reset_lo:08X}",
    'samples_first_20': race_samples[:20],
    'samples_last_10': race_samples[-10:],
}
save("reset_race")

# === STEP 9: Post-reset full state comparison ===
log("\n--- Step 9: Post-reset state comparison ---")
post_reset_snap = {}
changes = []
for name, off in critical_offsets:
    val = mm.r(off)
    post_reset_snap[name] = f"0x{val:08X}"
    pre_val = pre_reset_snap[name]
    changed = (f"0x{val:08X}" != pre_val)
    tag = " *** CHANGED ***" if changed else ""
    log(f"  {name:20s} = 0x{val:08X}  (was {pre_val}){tag}")
    if changed:
        changes.append(name)

results['post_reset_snapshot'] = post_reset_snap
results['changed_registers'] = changes
save("post_reset")

# === STEP 10: Try IC invalidate again post-reset (maybe trust chain refreshed) ===
log("\n--- Step 10: IC invalidate post-reset ---")
for name, op_off in [('MEC', MEC_IC_OP_CNTL), ('CPC', 0x584F), ('ME', 0x5847)]:
    mm.w(op_off, 0x00)
    time.sleep(0.01)
    mm.w(op_off, 0x01)
    time.sleep(0.1)
    op_val = mm.r(op_off)
    inv_complete = bool(op_val & 0x02)
    log(f"  {name} IC_OP_CNTL = 0x{op_val:08X}  INV_COMPLETE={inv_complete}")
    results[f'post_reset_inv_{name}'] = {
        'op_val': f"0x{op_val:08X}",
        'inv_complete': inv_complete,
    }

save("post_reset_inv")

# === STEP 11: Check for CP_HQD registers — compute queue descriptors ===
log("\n--- Step 11: CP_HQD compute queue descriptors ---")
# HQD (Hardware Queue Descriptor) controls compute queue execution
# If we can point an HQD to our code, MEC will execute it
hqd_regs = {}
hqd_base = 0x2350  # CP_HQD area
for off in range(hqd_base, hqd_base + 0x30):
    val = mm.r(off)
    if val != 0:
        mm.w(off, 0xFFFFFFFF)
        time.sleep(0.001)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        log(f"  0x{off:04X} = 0x{val:08X}  {'W' if writable else 'R'}")
        hqd_regs[f"0x{off:04X}"] = {'value': f"0x{val:08X}", 'writable': writable}

# Also check CP_HQD_PQ_BASE_LO/HI (ring buffer base for compute queue)
for name, off in [('CP_HQD_PQ_BASE_LO', 0x2360), ('CP_HQD_PQ_BASE_HI', 0x2361),
                   ('CP_HQD_PQ_RPTR', 0x2362), ('CP_HQD_PQ_WPTR_LO', 0x2368),
                   ('CP_HQD_PQ_CONTROL', 0x2370), ('CP_HQD_IB_BASE_ADDR', 0x2378),
                   ('CP_HQD_IB_RPTR', 0x237A), ('CP_HQD_ACTIVE', 0x2358)]:
    val = mm.r(off)
    mm.w(off, 0xAAAAAAAA)
    time.sleep(0.001)
    post = mm.r(off)
    mm.w(off, val)
    writable = (post != val)
    log(f"  {name:25s} (0x{off:04X}) = 0x{val:08X}  {'W' if writable else 'R'}")
    hqd_regs[name] = {'value': f"0x{val:08X}", 'writable': writable, 'offset': f"0x{off:04X}"}

results['hqd_registers'] = hqd_regs
save("hqd")

# === STEP 12: Scan for ADDITIONAL MEC-related registers ===
log("\n--- Step 12: Extended MEC register scan 0x5860-0x58FF ---")
ext_mec = {}
for off in range(0x5860, 0x5900):
    val = mm.r(off)
    if val != 0:
        mm.w(off, 0xFFFFFFFF)
        time.sleep(0.001)
        post = mm.r(off)
        mm.w(off, val)
        writable = (post != val)
        log(f"  0x{off:04X} = 0x{val:08X}  {'W' if writable else 'R'}")
        ext_mec[f"0x{off:04X}"] = {'value': f"0x{val:08X}", 'writable': writable}

results['ext_mec_regs'] = ext_mec
save("ext_mec")

# === STEP 13: Check if SDMA can DMA into IC ===
log("\n--- Step 13: SDMA register scan for IC DMA capability ---")
# SDMA0 registers around 0x4E00
sdma_regs = {}
sdma_interesting = [
    ('SDMA0_GFX_RB_BASE', 0x4E04),
    ('SDMA0_GFX_RB_BASE_HI', 0x4E05),
    ('SDMA0_GFX_RB_RPTR', 0x4E06),
    ('SDMA0_GFX_RB_WPTR', 0x4E08),
    ('SDMA0_GFX_DOORBELL', 0x4E0C),
    ('SDMA0_CNTL', 0x4E00),
    ('SDMA0_STATUS_REG', 0x4E68),
    ('SDMA0_CHICKEN_BITS', 0x4E10),
]
for name, off in sdma_interesting:
    val = mm.r(off)
    mm.w(off, 0xAAAAAAAA)
    time.sleep(0.001)
    post = mm.r(off)
    mm.w(off, val)
    writable = (post != val)
    log(f"  {name:30s} (0x{off:04X}) = 0x{val:08X}  {'W' if writable else 'R'}")
    sdma_regs[name] = {'value': f"0x{val:08X}", 'writable': writable}

results['sdma_regs'] = sdma_regs
save("sdma")

# === SUMMARY ===
log("\n=== SUMMARY ===")
log(f"  CP_MEC_CNTL writable mask: {results.get('cp_mec_cntl_writable_mask', 'N/A')}")
log(f"  MEC halt supported: {halt_results.get('halt_supported', 'N/A')}")
if halt_results.get('halt_supported'):
    log(f"  IC invalidate while halted: {halt_results.get('inv_complete_while_halted', 'N/A')}")
log(f"  GRBM_SOFT_RESET writable: 0x{sr_writable:08X}")
log(f"  Post-reset MEC_IC_BASE_LO: {results.get('reset_race', {}).get('post_reset_lo', 'N/A')}")
log(f"  Changed after reset: {changes}")

writable_hqd = [k for k, v in hqd_regs.items() if isinstance(v, dict) and v.get('writable')]
log(f"  Writable HQD registers: {writable_hqd}")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
