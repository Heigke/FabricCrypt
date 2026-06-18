#!/usr/bin/env python3
"""z2352f: Deep probe of RLC, SDMA, and CP control registers.

Focus on finding:
1. RLC_CNTL and safe mode registers (may enable IC reload)
2. SDMA control (alternative DMA engine for firmware loading)
3. CP_ME_CNTL halt/run bits (halt ME, modify IC_BASE, restart)
4. GPU reset behavior — does IC_BASE survive reset?
5. CP ucode RAM loading registers

Key insight: RLC controls firmware loading. If we can enter RLC safe mode
and then poke the right registers, we might bypass PSP.
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_rlc_deep_probe.json'
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
log("=== z2352f: RLC/SDMA/CP Deep Probe ===")

# === STEP 1: Full RLC register scan (0x4C00-0x4DFF) ===
log("\n--- Step 1: RLC registers 0x4C00-0x4DFF (non-zero + writable) ---")
rlc_regs = {}
rlc_nz = 0
for off in range(0x4C00, 0x4E00):
    val = mm.r(off)
    if val != 0:
        rlc_nz += 1
        if rlc_nz <= 40:
            # Quick writability
            mm.w(off, 0xFFFFFFFF)
            post = mm.r(off)
            mm.w(off, val)  # restore
            writable = (post != val)
            log(f"  0x{off:04X} = 0x{val:08X}  {'W(0x'+format(post,'08X')+')' if writable else 'R/O'}")
            rlc_regs[f"0x{off:04X}"] = {
                'val': f"0x{val:08X}",
                'writable': writable,
                'w_mask': f"0x{post:08X}" if writable else None,
            }

log(f"  Total non-zero in RLC: {rlc_nz}")
results['rlc_regs'] = rlc_regs
results['rlc_nonzero_count'] = rlc_nz
save("rlc")

# === STEP 2: CP_ME_CNTL and related control registers ===
log("\n--- Step 2: CP control registers ---")
cp_ctrl = {}
# GFX11 CP_ME_CNTL candidates (may have moved from 0x2186)
cp_candidates = {
    'CP_ME_CNTL_0x2186': 0x2186,
    'CP_MEC_CNTL_0x2188': 0x2188,
    # GFX11 may use 0x2900+ range
    'CP_GFX_RS64_DC_BASE_LO': 0x2900,
    'CP_GFX_RS64_DC_BASE_HI': 0x2901,
    # Check known GFX11 CP_ME_CNTL location
    'CP_ME_CNTL_0x586C': 0x586C,  # alternate?
}

for name, off in cp_candidates.items():
    val = mm.r(off)
    mm.w(off, 0xFFFFFFFF)
    post = mm.r(off)
    mm.w(off, val)
    writable = (post != val)
    log(f"  {name:35s} = 0x{val:08X}  {'W(0x'+format(post,'08X')+')' if writable else 'R/O'}")
    cp_ctrl[name] = {'val': f"0x{val:08X}", 'writable': writable}

results['cp_control'] = cp_ctrl
save("cp_ctrl")

# === STEP 3: Scan 0x2900-0x2AFF for RS64-specific registers ===
log("\n--- Step 3: RS64 register scan 0x2900-0x2AFF ---")
rs64_regs = {}
rs64_nz = 0
for off in range(0x2900, 0x2B00):
    val = mm.r(off)
    if val != 0:
        rs64_nz += 1
        if rs64_nz <= 30:
            log(f"  0x{off:04X} = 0x{val:08X}")
            rs64_regs[f"0x{off:04X}"] = f"0x{val:08X}"

log(f"  Total non-zero in 0x2900-0x2AFF: {rs64_nz}")
results['rs64_regs'] = rs64_regs
save("rs64")

# === STEP 4: SDMA registers (0x4E00-0x4FFF) ===
log("\n--- Step 4: SDMA registers 0x4E00-0x4FFF ---")
sdma_regs = {}
sdma_nz = 0
for off in range(0x4E00, 0x5000):
    val = mm.r(off)
    if val != 0:
        sdma_nz += 1
        if sdma_nz <= 30:
            mm.w(off, 0xFFFFFFFF)
            post = mm.r(off)
            mm.w(off, val)
            writable = (post != val)
            log(f"  0x{off:04X} = 0x{val:08X}  {'W' if writable else 'R/O'}")
            sdma_regs[f"0x{off:04X}"] = {
                'val': f"0x{val:08X}",
                'writable': writable,
            }

log(f"  Total non-zero in SDMA: {sdma_nz}")
results['sdma_regs'] = sdma_regs
results['sdma_nonzero_count'] = sdma_nz
save("sdma")

# === STEP 5: Check key 0x1000-0x11FF range (GC config, BIF) ===
log("\n--- Step 5: GC/BIF config 0x1000-0x11FF ---")
gc_config = {}
for off in range(0x1000, 0x1200):
    val = mm.r(off)
    if val != 0:
        gc_config[f"0x{off:04X}"] = f"0x{val:08X}"
        if len(gc_config) <= 25:
            log(f"  0x{off:04X} = 0x{val:08X}")

results['gc_config'] = gc_config
save("gc_config")

# === STEP 6: Try CP_ME_CNTL halt/unhalt ===
# On GFX11, CP_ME_CNTL should have ME_HALT (bit 28) and PFP_HALT (bit 26)
# If we can halt ME, change IC_BASE, and unhalt, we might redirect firmware
log("\n--- Step 6: CP_ME_CNTL halt test ---")

# First find which register is actually CP_ME_CNTL by checking for halt bits
# On previous gens: bit 28=ME_HALT, bit 26=PFP_HALT, bit 30=CE_HALT
me_cntl_candidates = [0x2186, 0x586C, 0x586D, 0x586E, 0x586F]
halt_test = {}
for off in me_cntl_candidates:
    orig = mm.r(off)
    # Try setting bit 28 (ME_HALT)
    mm.w(off, orig | (1 << 28))
    post = mm.r(off)
    mm.w(off, orig)  # restore immediately
    bit28_set = bool(post & (1 << 28))
    log(f"  0x{off:04X}: orig=0x{orig:08X}  set bit28→0x{post:08X}  bit28={bit28_set}")
    halt_test[f"0x{off:04X}"] = {
        'orig': f"0x{orig:08X}",
        'after_set28': f"0x{post:08X}",
        'bit28_stuck': bit28_set,
    }

results['halt_test'] = halt_test
save("halt")

# === STEP 7: Check for RLC_GPM registers that might trigger fw reload ===
log("\n--- Step 7: RLC_GPM and related registers ---")
rlc_gpm = {}
# RLC_GPM_GENERAL_0 through _15 are scratch registers used by RLC firmware
# RLC firmware uses these for signaling — maybe we can trigger a reload
for i in range(16):
    off = 0x4C80 + i
    val = mm.r(off)
    mm.w(off, 0xFFFFFFFF)
    post = mm.r(off)
    mm.w(off, val)
    writable = (post != val)
    if val != 0 or writable:
        log(f"  RLC_GPM_GENERAL_{i:d} (0x{off:04X}) = 0x{val:08X}  {'W' if writable else 'R/O'}")
        rlc_gpm[f"RLC_GPM_GENERAL_{i}"] = {
            'offset': f"0x{off:04X}",
            'val': f"0x{val:08X}",
            'writable': writable,
        }

# RLC_SAFE_MODE — entering safe mode suspends graphics and gives us more control
rlc_safe = 0x4C50
val = mm.r(rlc_safe)
mm.w(rlc_safe, 0xFFFFFFFF)
post = mm.r(rlc_safe)
mm.w(rlc_safe, val)
writable = (post != val)
log(f"  RLC_SAFE_MODE (0x4C50) = 0x{val:08X}  {'W(0x'+format(post,'08X')+')' if writable else 'R/O'}")
rlc_gpm['RLC_SAFE_MODE'] = {'val': f"0x{val:08X}", 'writable': writable}

# RLC_CNTL — master RLC control
rlc_cntl = 0x4C00
val = mm.r(rlc_cntl)
mm.w(rlc_cntl, 0xFFFFFFFF)
post = mm.r(rlc_cntl)
mm.w(rlc_cntl, val)
writable = (post != val)
log(f"  RLC_CNTL (0x4C00) = 0x{val:08X}  {'W(0x'+format(post,'08X')+')' if writable else 'R/O'}")
rlc_gpm['RLC_CNTL'] = {'val': f"0x{val:08X}", 'writable': writable}

results['rlc_gpm'] = rlc_gpm
save("rlc_gpm")

# === STEP 8: Look for PSP/SMU mailbox registers ===
log("\n--- Step 8: PSP mailbox region scan ---")
# PSP mailbox is usually at MP0_BASE or MP1_BASE
# On GFX11, MP0 is around 0x16000, MP1 (SMU) around 0xB8000
# But within 1MB aperture, check 0x16000-0x160FF
psp_regs = {}
for off in range(0x16000, 0x16100):
    try:
        val = mm.r(off)
        if val != 0:
            psp_regs[f"0x{off:05X}"] = f"0x{val:08X}"
            if len(psp_regs) <= 15:
                log(f"  0x{off:05X} = 0x{val:08X}")
    except:
        break

# Also check MP1 (SMU) area 0xB8000+ — but within 1MB = 0x40000 DWORDs
# 0xB8000 > 0x40000, so it's outside our aperture
# Check alternative SMU locations
for off in [0x3B000, 0x3B800, 0x3C000]:
    try:
        val = mm.r(off)
        if val != 0:
            log(f"  0x{off:05X} = 0x{val:08X}")
            psp_regs[f"0x{off:05X}"] = f"0x{val:08X}"
    except:
        pass

log(f"  PSP registers found: {len(psp_regs)}")
results['psp_regs'] = psp_regs
save("psp")

# === STEP 9: Test if IC_BASE_LO for CPC survives across register read/write cycles ===
log("\n--- Step 9: CPC IC_BASE stability test ---")
# Write a known value to CPC_IC_BASE_LO, verify it persists
cpc_lo = 0x584C
orig_cpc = mm.r(cpc_lo)
test_val = 0xDEAD0000
mm.w(cpc_lo, test_val)
time.sleep(0.5)

# Read back multiple times
reads = []
for i in range(10):
    val = mm.r(cpc_lo)
    reads.append(f"0x{val:08X}")
    time.sleep(0.1)

mm.w(cpc_lo, orig_cpc)  # restore
log(f"  CPC_IC_BASE_LO write 0x{test_val:08X}, reads: {reads[:5]}")
log(f"  Persists: {all(r == f'0x{test_val:08X}' for r in reads)}")
results['cpc_stability'] = {'wrote': f"0x{test_val:08X}", 'reads': reads}
save("stability")

# === STEP 10: Scan for MEC (Micro Engine Compute) registers ===
# MEC is separate from ME — handles compute dispatch
# MEC_IC_BASE_LO should be near 0x5848 based on earlier gap scan
log("\n--- Step 10: MEC IC register analysis ---")
mec_regs = {}
for name, off in [('MEC_IC_BASE_LO?', 0x5848), ('MEC_IC_BASE_HI?', 0x5849),
                   ('MEC_IC_BASE_CNTL?', 0x584A), ('MEC_IC_OP_CNTL?', 0x584B)]:
    val = mm.r(off)
    # Full write test
    mm.w(off, 0xAAAAAAAA)
    post_aa = mm.r(off)
    mm.w(off, 0x55555555)
    post_55 = mm.r(off)
    mm.w(off, val)  # restore
    writable = (post_aa != val) or (post_55 != val)
    log(f"  {name:20s} (0x{off:04X}) = 0x{val:08X}  AA→0x{post_aa:08X}  55→0x{post_55:08X}  {'W' if writable else 'R/O'}")
    mec_regs[name] = {
        'offset': f"0x{off:04X}",
        'val': f"0x{val:08X}",
        'write_aa': f"0x{post_aa:08X}",
        'write_55': f"0x{post_55:08X}",
        'writable': writable,
    }

results['mec_regs'] = mec_regs
save("mec")

# === STEP 11: Look at 0x4200-0x48FF for hidden SDMA/RLC registers ===
log("\n--- Step 11: Non-zero in 0x4200-0x48FF ---")
hidden = {}
hidden_nz = 0
for off in range(0x4200, 0x4900):
    val = mm.r(off)
    if val != 0:
        hidden_nz += 1
        if hidden_nz <= 20:
            log(f"  0x{off:04X} = 0x{val:08X}")
            hidden[f"0x{off:04X}"] = f"0x{val:08X}"

log(f"  Total non-zero 0x4200-0x48FF: {hidden_nz}")
results['hidden_4200_48FF'] = hidden
save("hidden")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
