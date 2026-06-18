#!/usr/bin/env python3
"""z2352f: Post-PSP-Load Injection Vectors.

PSP validates firmware signature during load. But AFTER the firmware is loaded
and running, can we modify it?

Attack vectors:
1. MEC_IC_BASE_LO/HI point to GTT buffer — can we find and modify it?
2. Can we read MEC instruction memory through ucode RAM registers?
3. Can we use GRBM backdoor to inject after validation?
4. What happens if we modify VRAM at the IC_BASE address?
5. Can we find the firmware in VRAM/GTT via BAR0 scanning?

Also: Check if amdgpu exposes the firmware buffer address anywhere.
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
BAR0_PHYS = 0x6800000000  # VRAM BAR
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_post_load_inject.json'
    with open(p, 'w') as f:
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

log("=== z2352f: Post-PSP-Load Injection Vectors ===")
mm = MMIO(MMIO_BASE)

# === STEP 1: Read current IC_BASE addresses for ALL engines ===
log("\n--- Step 1: Current IC_BASE addresses ---")
engines = {
    'PFP': {'base_lo': 0x5840, 'base_hi': 0x5841, 'base_cntl': 0x5842, 'op_cntl': 0x5843},
    'ME':  {'base_lo': 0x5844, 'base_hi': 0x5845, 'base_cntl': 0x5846, 'op_cntl': 0x5847},
    'MEC': {'base_lo': 0x5848, 'base_hi': 0x5849, 'base_cntl': 0x584A, 'op_cntl': 0x584B},
    'CPC': {'base_lo': 0x584C, 'base_hi': 0x584D, 'base_cntl': 0x584E, 'op_cntl': 0x297A},
    'MES': {'base_lo': 0x5850, 'base_hi': 0x5851, 'base_cntl': 0x5852, 'op_cntl': 0x5853},
}

ic_base_info = {}
for name, regs in engines.items():
    lo = mm.r(regs['base_lo'])
    hi = mm.r(regs['base_hi'])
    cntl = mm.r(regs['base_cntl'])
    addr_64 = (hi << 32) | lo
    log(f"  {name:4s}: IC_BASE = 0x{hi:08X}_{lo:08X}  CNTL=0x{cntl:08X}")
    ic_base_info[name] = {
        'base_lo': f"0x{lo:08X}",
        'base_hi': f"0x{hi:08X}",
        'base_cntl': f"0x{cntl:08X}",
        'addr_64': f"0x{addr_64:016X}",
    }

results['ic_base_addresses'] = ic_base_info
save("ic_base")

# === STEP 2: Try to read MEC microcode via UCODE registers ===
log("\n--- Step 2: Read MEC microcode via CP_MEC_ME1_UCODE_ADDR/DATA ---")

# Set address to 0 and try reading sequential UCODE_DATA
ucode_addr_reg = 0x2158  # CP_MEC_ME1_UCODE_ADDR
ucode_data_reg = 0x2159  # CP_MEC_ME1_UCODE_DATA

orig_addr = mm.r(ucode_addr_reg)
log(f"  Original UCODE_ADDR: 0x{orig_addr:08X}")

# Try setting UCODE_ADDR to 0
mm.w(ucode_addr_reg, 0x00000000)
time.sleep(0.01)
post_addr = mm.r(ucode_addr_reg)
log(f"  After write 0: UCODE_ADDR = 0x{post_addr:08X}")

# Read first 32 DWORDS from UCODE_DATA (should auto-increment)
ucode_words = []
mm.w(ucode_addr_reg, 0x00000000)
time.sleep(0.01)
for i in range(32):
    word = mm.r(ucode_data_reg)
    ucode_words.append(word)

# Check addr after reading
post_read_addr = mm.r(ucode_addr_reg)
log(f"  After reading 32 DWORDs: UCODE_ADDR = 0x{post_read_addr:08X}")

# Show first 16 words
non_zero = sum(1 for w in ucode_words if w != 0)
log(f"  Non-zero words: {non_zero}/32")
for i, w in enumerate(ucode_words[:16]):
    log(f"    [{i:3d}] 0x{w:08X}")

results['ucode_readback'] = {
    'addr_writable': (post_addr != orig_addr),
    'auto_increment': (post_read_addr != 0),
    'non_zero_count': non_zero,
    'first_16': [f"0x{w:08X}" for w in ucode_words[:16]],
}
mm.w(ucode_addr_reg, orig_addr)
save("ucode_readback")

# === STEP 3: Check if IC_BASE points into VRAM or GTT ===
log("\n--- Step 3: Decode IC_BASE addresses ---")

# MEC IC_BASE_LO is 0x5848
mec_lo = mm.r(0x5848)
mec_hi = mm.r(0x5849)
mec_addr = (mec_hi << 32) | mec_lo
log(f"  MEC IC_BASE = 0x{mec_addr:016X}")

# CPC IC_BASE_LO is 0x584C
cpc_lo = mm.r(0x584C)
cpc_hi = mm.r(0x584D)
cpc_addr = (cpc_hi << 32) | cpc_lo
log(f"  CPC IC_BASE = 0x{cpc_addr:016X}")

# Check if these are GPU virtual addresses or physical
# If in VRAM range (BAR0 = 0x6800000000), we can access via BAR0
# If GTT (system memory), we might access via /dev/mem
log(f"  BAR0 (VRAM) = 0x{BAR0_PHYS:016X}")

# MEC_IC_BASE might be 0 or a VRAM offset
# The driver uses amdgpu_bo_create_reserved with AMDGPU_GEM_DOMAIN_GTT
# GTT addresses are GPU virtual addresses that map to system memory via GART
# We need the GART table to translate

results['ic_base_decode'] = {
    'mec_addr': f"0x{mec_addr:016X}",
    'cpc_addr': f"0x{cpc_addr:016X}",
    'is_vram_range': mec_addr >= BAR0_PHYS,
}
save("ic_base_decode")

# === STEP 4: Try scanning VRAM for f32 opcodes (firmware signatures) ===
log("\n--- Step 4: Scan VRAM near IC_BASE offsets for f32 opcodes ---")

# Known f32 opcode patterns from MEC firmware
# 0xC424000B, 0x800003B0, 0xD800008B are common
f32_signatures = [0xC424000B, 0x800003B0, 0xD800008B, 0xC0310800]

# If MEC IC_BASE_LO has a value, try scanning VRAM at that offset
if mec_lo > 0 and mec_lo < 256*1024*1024:
    log(f"  MEC IC_BASE_LO = 0x{mec_lo:08X} — trying VRAM scan at that offset")
    try:
        scan_size = 4096  # 4KB
        vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
        # Read from VRAM at the IC_BASE offset
        vram_mm = mmap.mmap(vram_fd, scan_size, mmap.MAP_SHARED,
                           mmap.PROT_READ, offset=BAR0_PHYS + mec_lo)

        vram_words = []
        for i in range(scan_size // 4):
            w = struct.unpack('<I', vram_mm[i*4:(i+1)*4])[0]
            vram_words.append(w)

        # Check for f32 signatures
        found = []
        for sig in f32_signatures:
            for j, w in enumerate(vram_words):
                if w == sig:
                    found.append((j, sig))

        non_zero_vram = sum(1 for w in vram_words if w != 0)
        log(f"  VRAM at IC_BASE: {non_zero_vram}/{len(vram_words)} non-zero")
        log(f"  First 8 words:")
        for i in range(8):
            log(f"    [{i:3d}] 0x{vram_words[i]:08X}")

        if found:
            log(f"  FOUND f32 signatures: {found[:5]}")

        results['vram_at_ic_base'] = {
            'offset': f"0x{mec_lo:08X}",
            'non_zero': non_zero_vram,
            'first_16': [f"0x{w:08X}" for w in vram_words[:16]],
            'f32_matches': [(j, f"0x{s:08X}") for j, s in found[:10]],
        }

        vram_mm.close()
        os.close(vram_fd)
    except Exception as e:
        log(f"  VRAM scan failed: {e}")
        results['vram_at_ic_base'] = {'error': str(e)}
else:
    log(f"  MEC IC_BASE_LO = 0x{mec_lo:08X} — outside VRAM range or zero")
    results['vram_at_ic_base'] = 'no_valid_offset'

save("vram_scan")

# === STEP 5: Check if VRAM at offset 0 has firmware code ===
log("\n--- Step 5: Scan VRAM offset 0 for firmware ---")
try:
    vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
    vram_mm = mmap.mmap(vram_fd, 65536, mmap.MAP_SHARED,
                       mmap.PROT_READ, offset=BAR0_PHYS)

    # Read first 64KB of VRAM
    vram_first = []
    for i in range(65536 // 4):
        w = struct.unpack('<I', vram_mm[i*4:(i+1)*4])[0]
        vram_first.append(w)

    non_zero = sum(1 for w in vram_first if w != 0)
    log(f"  First 64KB of VRAM: {non_zero}/{len(vram_first)} non-zero words")

    # Search for f32 signatures in first 64KB
    sig_locs = {}
    for sig in f32_signatures:
        locs = [i for i, w in enumerate(vram_first) if w == sig]
        if locs:
            sig_locs[f"0x{sig:08X}"] = locs[:5]
            log(f"  Sig 0x{sig:08X} found at offsets: {[f'0x{l*4:06X}' for l in locs[:5]]}")

    # Also search for the PSP $PS1 header
    ps1_locs = [i for i, w in enumerate(vram_first) if w == 0x31535024]  # '$PS1' LE
    if ps1_locs:
        log(f"  $PS1 header found at: {[f'0x{l*4:06X}' for l in ps1_locs[:5]]}")

    results['vram_0_scan'] = {
        'non_zero': non_zero,
        'f32_sigs': sig_locs,
        'ps1_headers': [f"0x{l*4:06X}" for l in ps1_locs[:5]] if ps1_locs else [],
    }

    vram_mm.close()
    os.close(vram_fd)
except Exception as e:
    log(f"  VRAM scan error: {e}")
    results['vram_0_scan'] = {'error': str(e)}

save("vram_0_scan")

# === STEP 6: Check RLC RLCG/RLCS interface registers ===
# During PSP loading, RLC might have a "backdoor" interface
log("\n--- Step 6: RLC register probe ---")
rlc_regs = {
    'RLC_CNTL': 0x4C00,
    'RLC_CGCG_CGLS_CTRL': 0x4C48,
    'RLC_PG_CNTL': 0x4C43,
    'RLC_GPM_UCODE_ADDR': 0x4C04,
    'RLC_GPM_UCODE_DATA': 0x4C05,
    'RLC_AUTOLOAD_VRAM_ADDR_LO': 0x4CAC,
    'RLC_AUTOLOAD_VRAM_ADDR_HI': 0x4CAD,
}

rlc_state = {}
for name, off in rlc_regs.items():
    val = mm.r(off)
    log(f"  {name:40s} (0x{off:04X}) = 0x{val:08X}")
    rlc_state[name] = f"0x{val:08X}"

# The autoload VRAM address might tell us where PSP placed firmware!
autoload_lo = mm.r(0x4CAC)
autoload_hi = mm.r(0x4CAD)
autoload_addr = (autoload_hi << 32) | autoload_lo
log(f"\n  RLC AUTOLOAD VRAM addr = 0x{autoload_addr:016X}")
if autoload_addr > 0:
    log(f"  This is where PSP+RLC loaded firmware into VRAM!")
    # Check if this is within our BAR0 range
    if autoload_addr < 256*1024*1024:
        log(f"  Offset 0x{autoload_addr:08X} within VRAM BAR range — can access via BAR0!")
    else:
        log(f"  Address is beyond standard BAR0 range")

results['rlc_state'] = rlc_state
results['autoload_vram_addr'] = f"0x{autoload_addr:016X}"
save("rlc_probe")

# === STEP 7: Try to find TMR (Trusted Memory Region) boundaries ===
log("\n--- Step 7: PSP TMR probe ---")
# PSP TMR is in VRAM, typically at high addresses
# On this system: 0x97E0000000, size 0x8C00000 (from prior probing)
# TMR offset from BAR0 base: 0x97E0000000 - 0x6800000000 = 0x2FE0000000
# This is way beyond 256MB VRAM — TMR uses GPU virtual addressing

# Check PSP scratch registers for TMR info
psp_scratch_regs = []
for off in range(0x4C80, 0x4CA0):
    val = mm.r(off)
    if val != 0:
        log(f"  0x{off:04X} = 0x{val:08X}")
        psp_scratch_regs.append((f"0x{off:04X}", f"0x{val:08X}"))

results['psp_scratch'] = psp_scratch_regs
save("psp_tmr")

# === STEP 8: Test write to VRAM at IC_BASE address ===
log("\n--- Step 8: Test VRAM write at MEC IC_BASE (if valid) ---")
if mec_lo > 0 and mec_lo < 256*1024*1024:
    try:
        # Read the first DWORD at IC_BASE
        vram_fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        vram_mm = mmap.mmap(vram_fd, 4096, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE,
                           offset=BAR0_PHYS + mec_lo)

        orig_word = struct.unpack('<I', vram_mm[0:4])[0]
        log(f"  Original DWORD at VRAM+0x{mec_lo:08X}: 0x{orig_word:08X}")

        # Try writing a test pattern
        test_pattern = 0xDEADBEEF
        vram_mm[0:4] = struct.pack('<I', test_pattern)
        time.sleep(0.01)
        verify = struct.unpack('<I', vram_mm[0:4])[0]
        log(f"  After write 0x{test_pattern:08X}: readback = 0x{verify:08X}")

        writable = (verify == test_pattern)
        log(f"  VRAM at IC_BASE {'IS WRITABLE!' if writable else 'is NOT writable (encrypted/protected)'}")

        # RESTORE original value immediately!
        vram_mm[0:4] = struct.pack('<I', orig_word)
        time.sleep(0.01)
        restore_check = struct.unpack('<I', vram_mm[0:4])[0]
        log(f"  Restored: 0x{restore_check:08X}")

        results['vram_ic_base_writable'] = writable
        results['vram_ic_base_verify'] = f"0x{verify:08X}"

        vram_mm.close()
        os.close(vram_fd)
    except Exception as e:
        log(f"  VRAM write test failed: {e}")
        results['vram_ic_base_writable'] = False
        results['vram_ic_base_error'] = str(e)
else:
    log(f"  MEC IC_BASE_LO = 0x{mec_lo:08X} — no valid VRAM offset")
    results['vram_ic_base_writable'] = 'no_valid_offset'

save("vram_write")

# === STEP 9: Check TMR / TSME encryption effects on VRAM reads ===
log("\n--- Step 9: TSME encryption test (read known VRAM patterns) ---")
# If TSME is enabled, VRAM reads through BAR0 will return encrypted data
# Test: write a known pattern, read it back — if TSME, it should still work
# for BAR0 access (TSME encrypts DRAM, but BAR0 goes through GPU's MC)
try:
    # Use a safe VRAM area (offset 0x0FF0000 = 15.9MB)
    test_off = 0x0FF0000
    vram_fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    vram_mm = mmap.mmap(vram_fd, 4096, mmap.MAP_SHARED,
                       mmap.PROT_READ | mmap.PROT_WRITE,
                       offset=BAR0_PHYS + test_off)

    # Write known pattern
    patterns = [0xAAAAAAAA, 0x55555555, 0x12345678, 0xCAFEBABE]
    pattern_results = {}
    for pat in patterns:
        vram_mm[0:4] = struct.pack('<I', pat)
        time.sleep(0.001)
        readback = struct.unpack('<I', vram_mm[0:4])[0]
        match = (readback == pat)
        log(f"  Write 0x{pat:08X} → Read 0x{readback:08X}  {'MATCH' if match else 'MISMATCH (TSME?)'}")
        pattern_results[f"0x{pat:08X}"] = {'readback': f"0x{readback:08X}", 'match': match}

    results['tsme_test'] = pattern_results
    vram_mm.close()
    os.close(vram_fd)
except Exception as e:
    log(f"  TSME test error: {e}")
    results['tsme_test'] = {'error': str(e)}

save("tsme_test")

# === STEP 10: Check amdgpu iomem for firmware buffer mapping ===
log("\n--- Step 10: Check amdgpu debugfs for firmware addresses ---")
try:
    # Try reading amdgpu_vram_mm for BO info
    with open('/sys/kernel/debug/dri/0/amdgpu_vram_mm', 'r') as f:
        vram_mm_info = f.read()[:2000]
    log(f"  amdgpu_vram_mm: {len(vram_mm_info)} bytes")
    # Look for MEC-related allocations
    for line in vram_mm_info.split('\n')[:20]:
        if line.strip():
            log(f"    {line.strip()}")
    results['vram_mm_info'] = vram_mm_info[:500]
except Exception as e:
    log(f"  vram_mm read error: {e}")

try:
    # GTT memory manager
    with open('/sys/kernel/debug/dri/0/amdgpu_gtt_mm', 'r') as f:
        gtt_mm_info = f.read()[:2000]
    log(f"\n  amdgpu_gtt_mm: {len(gtt_mm_info)} bytes")
    for line in gtt_mm_info.split('\n')[:20]:
        if line.strip():
            log(f"    {line.strip()}")
    results['gtt_mm_info'] = gtt_mm_info[:500]
except Exception as e:
    log(f"  gtt_mm read error: {e}")

save("debugfs_info")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
