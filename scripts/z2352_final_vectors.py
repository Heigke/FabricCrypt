#!/usr/bin/env python3
"""z2352k: Final vector analysis — encryption, CNTL bit 29, and firmware patching.

CRITICAL FINDING: VRAM firmware is ENCRYPTED.
IC_BASE_CNTL bit 29 = encryption enable.

Remaining vectors:
1. Can we CLEAR bit 29 on CPC_IC_BASE_CNTL? (disable encryption for fetch)
2. ME_IC_BASE_CNTL has bit 29 CLEAR (0x0000531F) — is ME already unencrypted?
3. Can we redirect MES (bit 29=0, but IC_BASE_LO locked at 0x8000)?
4. What's at VRAM+0x8000 (MES firmware)?
5. PSP TMR boundaries — where is trusted memory?
6. Can we find the IC encryption key in registers?
"""
import mmap, struct, os, json, time

MMIO_BASE = 0xB4400000
BAR0_PHYS = 0x6800000000
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_final_vectors.json'
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
log("=== z2352k: Final Vector Analysis ===")

# === STEP 1: Encryption bit analysis ===
log("\n--- Step 1: IC_BASE_CNTL encryption bit (bit 29) analysis ---")
cntl_regs = {
    'PFP_IC_BASE_CNTL': 0x5842,
    'ME_IC_BASE_CNTL': 0x5846,
    'MEC_IC_BASE_CNTL': 0x584A,
    'CPC_IC_BASE_CNTL': 0x584E,
    'MES_IC_BASE_CNTL': 0x5852,
}

enc_analysis = {}
for name, off in cntl_regs.items():
    val = mm.r(off)
    bit29 = bool(val & (1 << 29))
    bit28 = bool(val & (1 << 28))
    vmid = val & 0xF
    
    # Try to clear bit 29
    mm.w(off, val & ~(1 << 29))
    time.sleep(0.01)
    after_clear = mm.r(off)
    can_clear_29 = not bool(after_clear & (1 << 29))
    mm.w(off, val)  # restore
    
    # Try to set bit 29
    mm.w(off, val | (1 << 29))
    time.sleep(0.01)
    after_set = mm.r(off)
    can_set_29 = bool(after_set & (1 << 29))
    mm.w(off, val)  # restore
    
    log(f"  {name:20s} = 0x{val:08X}  bit29(enc)={int(bit29)}  clear→{int(can_clear_29)}  set→{int(can_set_29)}  VMID={vmid}")
    enc_analysis[name] = {
        'value': f"0x{val:08X}",
        'bit29_encrypted': bit29,
        'can_clear_bit29': can_clear_29,
        'can_set_bit29': can_set_29,
    }

results['encryption_analysis'] = enc_analysis
save("encryption")

# === STEP 2: ME analysis — bit 29 is CLEAR, what does that mean? ===
log("\n--- Step 2: ME firmware — unencrypted? ---")
me_cntl = mm.r(0x5846)
me_lo = mm.r(0x5844)
me_hi = mm.r(0x5845)
log(f"  ME_IC_BASE_CNTL = 0x{me_cntl:08X} (bit29={bool(me_cntl & (1<<29))})")
log(f"  ME_IC_BASE_LO = 0x{me_lo:08X}")
log(f"  ME_IC_BASE_HI = 0x{me_hi:08X}")
log(f"  ME firmware base = HI:LO = 0x{me_hi:08X}:{me_lo:08X}")

# If ME firmware is at VRAM offset 0 (IC_BASE_LO=0), check if it's plaintext
if me_lo == 0:
    log("  ME_IC_BASE_LO = 0 → firmware at VRAM start")
    vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
    vram_mm = mmap.mmap(vram_fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=BAR0_PHYS)
    words = struct.unpack('<32I', vram_mm[:128])
    vram_mm.close()
    os.close(vram_fd)
    
    riscv = sum(1 for w in words if (w & 0x7F) in [0x13, 0x33, 0x03, 0x23, 0x63, 0x67, 0x6F, 0x37, 0x17, 0x73])
    log(f"  VRAM[0:128]: RISC-V matches={riscv}/32")
    log(f"  First 8: {' '.join(f'0x{w:08X}' for w in words[:8])}")
    results['me_vram_data'] = [f"0x{w:08X}" for w in words[:16]]
    results['me_vram_riscv_matches'] = riscv

save("me_analysis")

# === STEP 3: CPC bit 29 clear attempt with proper sequence ===
log("\n--- Step 3: CPC_IC_BASE_CNTL bit 29 clear attempt ---")
orig_cpc_cntl = mm.r(0x584E)
log(f"  CPC_IC_BASE_CNTL original: 0x{orig_cpc_cntl:08X}")

# Try clearing ALL bits (including bit 29)
mm.w(0x584E, 0x00000000)
time.sleep(0.01)
after_zero = mm.r(0x584E)
log(f"  After write 0x00000000: 0x{after_zero:08X}")
log(f"  Bit 29 cleared? {not bool(after_zero & (1<<29))}")

# Try writing with VMID=0, no encryption
mm.w(0x584E, 0x00000000)  # VMID=0, no enc, no addr_clamp
time.sleep(0.01)
after = mm.r(0x584E)
log(f"  With VMID=0, enc=0: 0x{after:08X}")

# Restore
mm.w(0x584E, orig_cpc_cntl)

results['cpc_bit29_clear'] = {
    'original': f"0x{orig_cpc_cntl:08X}",
    'after_zero': f"0x{after_zero:08X}",
    'bit29_cleared': not bool(after_zero & (1 << 29)),
}
save("cpc_clear")

# === STEP 4: MES analysis — IC_BASE_LO locked at 0x8000 ===
log("\n--- Step 4: MES firmware analysis ---")
mes_lo = mm.r(0x5850)
mes_hi = mm.r(0x5851)
mes_cntl = mm.r(0x5852)
log(f"  MES_IC_BASE_LO = 0x{mes_lo:08X} (locked)")
log(f"  MES_IC_BASE_HI = 0x{mes_hi:08X}")
log(f"  MES_IC_BASE_CNTL = 0x{mes_cntl:08X} (bit29={bool(mes_cntl & (1<<29))})")

# MES has bit 29=0 and IC_BASE_LO=0x8000
# Check if VRAM at 0x8000 is plaintext or encrypted
vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
vram_mm = mmap.mmap(vram_fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=BAR0_PHYS + 0x8000)
words = struct.unpack('<32I', vram_mm[:128])
vram_mm.close()
os.close(vram_fd)

riscv = sum(1 for w in words if (w & 0x7F) in [0x13, 0x33, 0x03, 0x23, 0x63, 0x67, 0x6F, 0x37, 0x17, 0x73])
unique = len(set(words))
log(f"  VRAM[0x8000:0x8080]: unique={unique}/32  RISC-V matches={riscv}/32")
log(f"  First 8: {' '.join(f'0x{w:08X}' for w in words[:8])}")
results['mes_vram_data'] = [f"0x{w:08X}" for w in words[:16]]
results['mes_vram_riscv_matches'] = riscv
save("mes_analysis")

# === STEP 5: Scan for TMR (Trusted Memory Region) boundaries ===
log("\n--- Step 5: TMR boundary search ---")
# PSP defines a TMR in VRAM. Anything inside TMR is protected.
# TMR typically shows up as all-zeros or all-FF when read from outside
# Scan at 1MB intervals to find boundaries
vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
tmr_map = []
for mb in range(0, 256):
    offset = mb * 1024 * 1024
    try:
        vram_mm = mmap.mmap(vram_fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ, offset=BAR0_PHYS + offset)
        words = struct.unpack('<16I', vram_mm[:64])
        vram_mm.close()
        
        all_zero = all(w == 0 for w in words)
        all_ff = all(w == 0xFFFFFFFF for w in words)
        unique = len(set(words))
        
        if mb < 4 or all_zero or all_ff or unique < 3:
            pattern = "ZERO" if all_zero else ("FF" if all_ff else f"unique={unique}")
            log(f"  VRAM+{mb:3d}MB: {pattern}  first=0x{words[0]:08X}")
            tmr_map.append({'mb': mb, 'pattern': pattern, 'first': f"0x{words[0]:08X}"})
    except Exception as e:
        if mb < 4:
            log(f"  VRAM+{mb:3d}MB: ERROR {e}")

os.close(vram_fd)
results['tmr_scan'] = tmr_map
save("tmr")

# === STEP 6: Check for debug/fuse registers that control encryption ===
log("\n--- Step 6: Fuse/debug registers ---")
# On AMD GPUs, fuse registers control features like signature verification
# CC_GC_SHADER_ARRAY_CONFIG, GC_USER_GC_CONFIG, etc.
fuse_candidates = {
    'CC_GC_SHADER_ARRAY_CONFIG': 0x1120,
    'GC_USER_SHADER_ARRAY_CONFIG': 0x1121,
    'CC_RB_BACKEND_DISABLE': 0x1122,
    'GC_USER_RB_BACKEND_DISABLE': 0x1123,
    'CC_GC_SA_UNIT_DISABLE': 0x1124,
    # Some known fuse/config regs
    'RLC_GPU_IOV_F32': 0x5854,
    'CP_CONFIG_0x585C': 0x585C,
}

fuse_results = {}
for name, off in fuse_candidates.items():
    val = mm.r(off)
    mm.w(off, 0xFFFFFFFF)
    post = mm.r(off)
    mm.w(off, val)
    writable = (post != val)
    if val != 0:
        log(f"  {name:35s} (0x{off:04X}) = 0x{val:08X}  {'W' if writable else 'R/O'}")
    fuse_results[name] = {
        'val': f"0x{val:08X}",
        'writable': writable,
    }

results['fuse_debug'] = fuse_results
save("fuse")

# === STEP 7: Look for SRAM/GDS writeable memory ===
log("\n--- Step 7: GDS/LDS/SRAM scan ---")
# GDS (Global Data Share) is accessible GPU memory
# On GFX11, GDS might be at 0x2300+ or through specific registers
# Check CP_GDS_BKUP_ADDR/DATA
gds_candidates = {
    'GDS_ADDR': 0x2320,
    'GDS_DATA': 0x2321,
    'GDS_OA_ADDRESS': 0x2327,
    'GDS_OA_COUNTER': 0x2328,
}
for name, off in gds_candidates.items():
    val = mm.r(off)
    mm.w(off, 0xAAAAAAAA)
    post = mm.r(off)
    mm.w(off, val)
    writable = (post != val)
    log(f"  {name:20s} (0x{off:04X}) = 0x{val:08X}  W→0x{post:08X}  {'W' if writable else 'R/O'}")

save("gds")

# === STEP 8: Check what IC_BASE_HI values mean ===
log("\n--- Step 8: IC_BASE_HI interpretation ---")
# CPC_IC_BASE_HI = 0x00000200
# This could be: upper bits of 48-bit address, VMID, or other config
cpc_hi = mm.r(0x584D)
log(f"  CPC_IC_BASE_HI = 0x{cpc_hi:08X}")
log(f"    If upper address: full addr = 0x{cpc_hi:08X}_{mm.r(0x584C):08X}")
log(f"    If VMID context: VMID = {cpc_hi & 0xF}")
log(f"    Bits: {cpc_hi:032b}")

# Check MES_IC_BASE_HI writability detail
mes_hi = mm.r(0x5851)
mm.w(0x5851, 0xFFFFFFFF)
mes_hi_w = mm.r(0x5851)
mm.w(0x5851, mes_hi)
log(f"  MES_IC_BASE_HI = 0x{mes_hi:08X}  W→0x{mes_hi_w:08X}  mask=0x{mes_hi_w:08X}")

results['base_hi_analysis'] = {
    'CPC': f"0x{cpc_hi:08X}",
    'MES': f"0x{mes_hi:08X}",
    'MES_write_mask': f"0x{mes_hi_w:08X}",
}
save("base_hi")

# === STEP 9: Final summary ===
log("\n--- FINAL SUMMARY ---")
log("Register Access Hierarchy:")
log("  FULLY WRITABLE: CPC_IC_BASE_LO, CPC_IC_BASE_HI, MEC_IC_BASE_LO/HI")
log("  PARTIAL WRITE:  PFP_IC_BASE_LO (14 bits), MES_IC_BASE_HI, various CNTL bits")
log("  READ-ONLY:      ME_IC_BASE_LO/HI, MES_IC_BASE_LO")
log("")
log("Encryption Status:")
log(f"  PFP: bit29=1 (ENCRYPTED)")
log(f"  ME:  bit29=0 (PLAINTEXT?)")
log(f"  MEC: bit29=0 (PLAINTEXT?)")
log(f"  CPC: bit29=1 (ENCRYPTED)")
log(f"  MES: bit29=0 (PLAINTEXT?)")
log("")
log("IC Invalidate/Prime: BLOCKED on ALL engines (requires PSP trust chain)")
log("GPU Reset: CPC_IC_BASE_LO PERSISTS through MODE2 reset")
log("VRAM Firmware: HIGH ENTROPY (encrypted or TMR-protected)")
log("Legacy Ucode RAM: DEAD on RS64 (all zeros, not writable)")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
