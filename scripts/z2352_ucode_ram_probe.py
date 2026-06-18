#!/usr/bin/env python3
"""z2352h: Deep probe of CP ucode RAM FIFO and MEC redirect.

Key findings so far:
- CP_ME_RAM_DATA (0x2153) is FIFO — read-only, auto-advancing
- CP_ME_RAM_WADDR (0x2152) is writable
- MEC IC registers (0x5848-0x584B) are ALL fully writable
- PSP C2PMSG registers are partially writable

Test plan:
1. Read CP_ME_RAM sequentially (firmware dump attempt)
2. Test CP_ME_RAM_WADDR + DATA write path
3. MEC IC_BASE redirect with triggered invalidate (try writing IC_OP_CNTL bits differently)
4. Check if CP_PFP_UCODE_ADDR/DATA path works on GFX11
5. Look for MEC_ME1 RAM path
6. Try alternate invalidate trigger via RLC-style command
"""
import mmap, struct, os, json, time, sys

MMIO_BASE = 0xB4400000
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_ucode_ram_probe.json'
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
log("=== z2352h: CP Ucode RAM + MEC Redirect Deep Probe ===")

# === STEP 1: Read CP_ME_RAM firmware dump ===
log("\n--- Step 1: CP_ME_RAM firmware dump attempt ---")

# Set read address to 0
mm.w(0x2152, 0)  # CP_ME_RAM_WADDR = 0
time.sleep(0.01)

# Read first 64 DWORDs through the DATA FIFO
fw_dump = []
for i in range(64):
    val = mm.r(0x2153)  # CP_ME_RAM_DATA (auto-incrementing)
    fw_dump.append(f"0x{val:08X}")
    if i < 16:
        log(f"  ME_RAM[{i:3d}] = 0x{val:08X}")

# Check if it looks like actual firmware or garbage
unique_vals = len(set(fw_dump))
all_zero = all(v == "0x00000000" for v in fw_dump)
log(f"  64 DWORDs read, {unique_vals} unique values, all_zero={all_zero}")

# Check for RISC-V instruction patterns
# RISC-V has specific opcode patterns in bits [6:0]
riscv_count = 0
for v in fw_dump:
    val = int(v, 16)
    opcode = val & 0x7F
    if opcode in [0x13, 0x33, 0x03, 0x23, 0x63, 0x67, 0x6F, 0x37, 0x17, 0x73]:
        riscv_count += 1

log(f"  RISC-V opcode matches: {riscv_count}/64")
results['fw_dump_first64'] = fw_dump[:32]
results['fw_dump_unique'] = unique_vals
results['fw_dump_riscv_matches'] = riscv_count
save("fw_dump")

# === STEP 2: Test ucode RAM write path ===
log("\n--- Step 2: CP_ME_RAM write path test ---")

# Save original first 4 words
mm.w(0x2152, 0)  # Reset address
time.sleep(0.01)
orig_words = [mm.r(0x2153) for _ in range(4)]
log(f"  Original ME_RAM[0:4] = {[f'0x{v:08X}' for v in orig_words]}")

# Try writing via WADDR + DATA
mm.w(0x2152, 0)  # Set write address to 0
time.sleep(0.01)

# Write RISC-V NOPs
RISCV_NOP = 0x00000013
for i in range(4):
    mm.w(0x2153, RISCV_NOP)
    time.sleep(0.001)

# Read back
mm.w(0x2152, 0)  # Reset address
time.sleep(0.01)
post_words = [mm.r(0x2153) for _ in range(4)]
log(f"  After NOP write: ME_RAM[0:4] = {[f'0x{v:08X}' for v in post_words]}")

wrote_nops = all(v == RISCV_NOP for v in post_words)
data_changed = (post_words != orig_words)
log(f"  Data changed: {data_changed}")
log(f"  NOPs written successfully: {wrote_nops}")

results['ram_write'] = {
    'original': [f"0x{v:08X}" for v in orig_words],
    'after_write': [f"0x{v:08X}" for v in post_words],
    'changed': data_changed,
    'nops_verified': wrote_nops,
}
save("ram_write")

# Restore if changed
if data_changed:
    mm.w(0x2152, 0)
    for v in orig_words:
        mm.w(0x2153, v)
    log(f"  Restored original data")

# === STEP 3: Check ALL ucode RAM registers ===
log("\n--- Step 3: All ucode RAM register pairs ---")
ucode_pairs = [
    ('CP_PFP_UCODE', 0x2150, 0x2151),
    ('CP_ME_RAM',    0x2152, 0x2153),
    ('CP_CE_UCODE',  0x2154, 0x2155),
    ('CP_MEC_ME1',   0x2158, 0x2159),
    ('CP_MEC_ME2',   0x215A, 0x215B),
]

ucode_info = {}
for name, addr_reg, data_reg in ucode_pairs:
    # Check addr writability
    orig_addr = mm.r(addr_reg)
    mm.w(addr_reg, 0)
    time.sleep(0.001)
    addr_val = mm.r(addr_reg)
    addr_writable = (addr_val == 0) or (addr_val != orig_addr)
    mm.w(addr_reg, orig_addr)
    
    # Read first 8 words from data
    mm.w(addr_reg, 0)
    time.sleep(0.001)
    data_words = [mm.r(data_reg) for _ in range(8)]
    
    unique = len(set(data_words))
    log(f"  {name:15s}: ADDR(0x{addr_reg:04X})={'W' if addr_writable else 'R/O'}  DATA unique={unique}  first={[f'0x{v:08X}' for v in data_words[:4]]}")
    
    ucode_info[name] = {
        'addr_reg': f"0x{addr_reg:04X}",
        'data_reg': f"0x{data_reg:04X}",
        'addr_writable': addr_writable,
        'data_first8': [f"0x{v:08X}" for v in data_words],
        'unique_values': unique,
    }

results['ucode_paths'] = ucode_info
save("ucode_paths")

# === STEP 4: MEC IC_BASE redirect test ===
log("\n--- Step 4: MEC IC_BASE redirect ---")

# Read current MEC IC state
mec_state = {}
for name, off in [('MEC_IC_BASE_LO', 0x5848), ('MEC_IC_BASE_HI', 0x5849),
                   ('MEC_IC_BASE_CNTL', 0x584A), ('MEC_IC_OP_CNTL', 0x584B)]:
    val = mm.r(off)
    mec_state[name] = f"0x{val:08X}"
    log(f"  {name:20s} = 0x{val:08X}")

results['mec_state'] = mec_state
save("mec_state")

# Write NOP sled address to MEC IC_BASE
NOP_VRAM_OFF = 0x0F00000  # 15MB into VRAM (NOP sled from z2352d)
log(f"\n  Setting MEC_IC_BASE_LO = 0x{NOP_VRAM_OFF:08X}")
orig_mec_lo = mm.r(0x5848)
orig_mec_hi = mm.r(0x5849)
orig_mec_cntl = mm.r(0x584A)
orig_mec_op = mm.r(0x584B)

mm.w(0x5848, NOP_VRAM_OFF)
time.sleep(0.01)
verify = mm.r(0x5848)
log(f"  MEC_IC_BASE_LO verify: 0x{verify:08X}  {'OK' if verify == NOP_VRAM_OFF else 'FAIL'}")

# Try various invalidate trigger methods
log("\n  Testing MEC IC invalidate (multiple methods)...")

# Method 1: Write IC_OP_CNTL bit 0 (INVALIDATE)
mm.w(0x584B, 0x01)
time.sleep(0.2)
post1 = mm.r(0x584B)
inv1 = bool(post1 & 0x02)
log(f"  Method 1 (bit 0): IC_OP_CNTL=0x{post1:08X} INV_COMPLETE={inv1}")

# Method 2: Write higher bits (maybe GFX11 uses different bit layout)
mm.w(0x584B, 0x00)
time.sleep(0.01)
mm.w(0x584B, 0x100)  # bit 8
time.sleep(0.2)
post2 = mm.r(0x584B)
inv2 = bool(post2 & 0x200)  # bit 9 = complete?
log(f"  Method 2 (bit 8): IC_OP_CNTL=0x{post2:08X} bit9={inv2}")

# Method 3: Try full 32-bit pattern
mm.w(0x584B, 0x00)
time.sleep(0.01)
mm.w(0x584B, 0x00010001)  # both bit 0 and bit 16
time.sleep(0.2)
post3 = mm.r(0x584B)
log(f"  Method 3 (bit 0+16): IC_OP_CNTL=0x{post3:08X}")

# Method 4: Clear then set
mm.w(0x584B, 0x00000000)
time.sleep(0.05)
mm.w(0x584B, 0x00000001)
time.sleep(0.05)
mm.w(0x584B, 0x00000011)  # INV + PRIME both
time.sleep(0.2)
post4 = mm.r(0x584B)
log(f"  Method 4 (INV+PRIME): IC_OP_CNTL=0x{post4:08X}")

results['mec_invalidate'] = {
    'method1': f"0x{post1:08X}",
    'method2': f"0x{post2:08X}",
    'method3': f"0x{post3:08X}",
    'method4': f"0x{post4:08X}",
}
save("mec_inv")

# Restore MEC state
mm.w(0x5848, orig_mec_lo)
mm.w(0x5849, orig_mec_hi)
mm.w(0x584A, orig_mec_cntl)
mm.w(0x584B, orig_mec_op)
log("  MEC state restored")

# === STEP 5: Try CP_MEC_ME1 ucode RAM path ===
log("\n--- Step 5: CP_MEC_ME1 ucode RAM deep test ---")

# Set MEC ME1 address to 0
mec1_addr = 0x2158
mec1_data = 0x2159
mm.w(mec1_addr, 0)
time.sleep(0.01)

# Read first 32 words
mec_fw = []
for i in range(32):
    val = mm.r(mec1_data)
    mec_fw.append(val)
    if i < 8:
        log(f"  MEC_ME1_RAM[{i:3d}] = 0x{val:08X}")

unique = len(set(mec_fw))
log(f"  32 DWORDs, {unique} unique")

# Try writing
mm.w(mec1_addr, 0)
time.sleep(0.01)
mm.w(mec1_data, RISCV_NOP)
time.sleep(0.001)
mm.w(mec1_data, RISCV_NOP)

# Read back
mm.w(mec1_addr, 0)
time.sleep(0.01)
post = [mm.r(mec1_data) for _ in range(4)]
log(f"  After write: {[f'0x{v:08X}' for v in post]}")
write_took = (post[0] == RISCV_NOP)
log(f"  Write successful: {write_took}")

results['mec_me1_ram'] = {
    'first8': [f"0x{v:08X}" for v in mec_fw[:8]],
    'unique': unique,
    'write_test': [f"0x{v:08X}" for v in post],
    'write_success': write_took,
}
save("mec_ram")

# === STEP 6: Scan for additional register areas we haven't checked ===
log("\n--- Step 6: Unexplored areas scan ---")

# Check 0x2100-0x21FF for more CP control registers
cp_2100 = {}
for off in range(0x2100, 0x2200):
    val = mm.r(off)
    if val != 0:
        cp_2100[f"0x{off:04X}"] = f"0x{val:08X}"
        if len(cp_2100) <= 15:
            log(f"  0x{off:04X} = 0x{val:08X}")

log(f"  Non-zero in 0x2100-0x21FF: {len(cp_2100)}")
results['cp_2100'] = cp_2100

# Check 0x2300-0x23FF (HQD area)
hqd_nz = 0
for off in range(0x2300, 0x2400):
    val = mm.r(off)
    if val != 0:
        hqd_nz += 1
        if hqd_nz <= 5:
            log(f"  0x{off:04X} = 0x{val:08X}")
log(f"  Non-zero in HQD 0x2300-0x23FF: {hqd_nz}")

# Check 0x5900-0x5BFF (extended CP/IC area)
ext_nz = 0
for off in range(0x5900, 0x5C00):
    val = mm.r(off)
    if val != 0:
        ext_nz += 1
        if ext_nz <= 10:
            log(f"  0x{off:04X} = 0x{val:08X}")
log(f"  Non-zero in 0x5900-0x5BFF: {ext_nz}")

save("unexplored")

# === STEP 7: GPU reset test ===
# Check if CPC IC_BASE_LO survives a GPU reset
# First write a distinctive value, then reset, then check
log("\n--- Step 7: CPC IC_BASE_LO reset persistence test ---")
log("  Writing 0xCAFE0000 to CPC_IC_BASE_LO before reset test...")
mm.w(0x584C, 0xCAFE0000)
time.sleep(0.1)
verify = mm.r(0x584C)
log(f"  CPC_IC_BASE_LO = 0x{verify:08X}")
results['pre_reset_cpc'] = f"0x{verify:08X}"
save("pre_reset")

# Don't actually reset yet — just save state for manual reset test
log("  (GPU reset requires manual trigger: echo 1 > /sys/class/drm/card0/device/reset)")
log("  CPC_IC_BASE_LO is set to 0xCAFE0000 — check after reset if value survives")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
mm.close()
log("\nDone.")
