#!/usr/bin/env python3
"""z2343 Part 2: Deeper VRAM firmware probing.
- Decompress actual gfx11_5_1 firmware and search in VRAM
- Wake GPU from GFXOFF to read CP SRAM
- Parse GRBM_STATUS bits
- Read more register ranges
- Try amdgpu_regs2 for extended register space
"""
import os, sys, json, time, struct, subprocess, glob
try:
    import zstandard
except ImportError:
    # Fallback: use zstd CLI tool
    zstandard = None
from datetime import datetime
from pathlib import Path

RESULTS = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
LOG = RESULTS / "z2343_vram_dump_log.txt"
FW_DUMP = RESULTS / "z2343_vram_fw_dump.txt"
CP_SRAM = RESULTS / "z2343_cp_sram.txt"
JSON_OUT = RESULTS / "z2343_vram_dump.json"

DEBUGFS = "/sys/kernel/debug/dri/0"
VRAM_FILE = f"{DEBUGFS}/amdgpu_vram"
REGS_FILE = f"{DEBUGFS}/amdgpu_regs"
THERMAL = "/sys/class/thermal/thermal_zone0/temp"

VRAM_BASE = 0x8000000000

# Load existing findings
findings = json.load(open(JSON_OUT))

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def save_json():
    with open(JSON_OUT, "w") as f:
        json.dump(findings, f, indent=2, default=str)

def check_temp():
    try:
        return int(open(THERMAL).read().strip()) / 1000.0
    except:
        return 0.0

def safe_vram_read(offset, size, timeout=30):
    import signal
    class TimeoutError(Exception): pass
    def handler(signum, frame): raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        f = open(VRAM_FILE, "rb")
        f.seek(offset)
        data = f.read(size)
        f.close()
        signal.alarm(0)
        return data
    except:
        signal.alarm(0)
        return None
    finally:
        signal.signal(signal.SIGALRM, old)

def safe_reg_read(offset, timeout=10):
    import signal
    class TimeoutError(Exception): pass
    def handler(signum, frame): raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        f = open(REGS_FILE, "rb")
        f.seek(offset)
        data = f.read(4)
        f.close()
        signal.alarm(0)
        if len(data) == 4:
            return struct.unpack("<I", data)[0]
        return None
    except:
        signal.alarm(0)
        return None
    finally:
        signal.signal(signal.SIGALRM, old)

def hexdump(data, offset=0, limit=256):
    lines = []
    for i in range(0, min(len(data), limit), 16):
        hex_part = " ".join(f"{b:02x}" for b in data[i:i+16])
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in data[i:i+16])
        lines.append(f"  {offset+i:012x}: {hex_part:<48s}  {ascii_part}")
    return "\n".join(lines)

log("\n" + "=" * 70)
log("z2343 Part 2: Deep VRAM + CP firmware probing")
log("=" * 70)
log(f"Temperature: {check_temp():.1f}°C")

# ============================================================
# STEP 2A: Parse GRBM_STATUS to understand GPU state
# ============================================================
log("\n--- STEP 2A: GRBM_STATUS Decode ---")
grbm = safe_reg_read(0x8010)  # GRBM_STATUS
grbm2 = safe_reg_read(0x8008)  # GRBM_STATUS2
log(f"GRBM_STATUS  = {hex(grbm) if grbm else 'N/A'}")
log(f"GRBM_STATUS2 = {hex(grbm2) if grbm2 else 'N/A'}")

if grbm is not None:
    bits = {
        "ME0PIPE0_CMDFIFO_AVAIL": (grbm >> 0) & 0xF,
        "RSMU_RQ_PENDING": (grbm >> 5) & 1,
        "ME0PIPE0_CF_RQ_PENDING": (grbm >> 7) & 1,
        "ME0PIPE0_PF_RQ_PENDING": (grbm >> 8) & 1,
        "GDS_DMA_RQ_PENDING": (grbm >> 9) & 1,
        "DB_CLEAN": (grbm >> 12) & 1,
        "CB_CLEAN": (grbm >> 13) & 1,
        "TA_BUSY": (grbm >> 14) & 1,
        "GDS_BUSY": (grbm >> 15) & 1,
        "WD_BUSY_NO_DMA": (grbm >> 16) & 1,
        "VGT_BUSY": (grbm >> 17) & 1,
        "IA_BUSY_NO_DMA": (grbm >> 18) & 1,
        "IA_BUSY": (grbm >> 19) & 1,
        "SX_BUSY": (grbm >> 20) & 1,
        "WD_BUSY": (grbm >> 21) & 1,
        "SPI_BUSY": (grbm >> 22) & 1,
        "BCI_BUSY": (grbm >> 23) & 1,
        "SC_BUSY": (grbm >> 24) & 1,
        "PA_BUSY": (grbm >> 25) & 1,
        "DB_BUSY": (grbm >> 26) & 1,
        "CP_COHERENCY_BUSY": (grbm >> 28) & 1,
        "CP_BUSY": (grbm >> 29) & 1,
        "CB_BUSY": (grbm >> 30) & 1,
        "GUI_ACTIVE": (grbm >> 31) & 1,
    }
    for name, val in bits.items():
        if val:
            log(f"  {name} = {val}")
    findings["grbm_status_decoded"] = bits

if grbm2 is not None:
    bits2 = {
        "ME0PIPE1_CMDFIFO_AVAIL": (grbm2 >> 0) & 0xF,
        "ME0PIPE1_CF_RQ_PENDING": (grbm2 >> 4) & 1,
        "ME0PIPE1_PF_RQ_PENDING": (grbm2 >> 5) & 1,
        "ME1PIPE0_RQ_PENDING": (grbm2 >> 6) & 1,
        "ME1PIPE1_RQ_PENDING": (grbm2 >> 7) & 1,
        "ME1PIPE2_RQ_PENDING": (grbm2 >> 8) & 1,
        "ME1PIPE3_RQ_PENDING": (grbm2 >> 9) & 1,
        "ME2PIPE0_RQ_PENDING": (grbm2 >> 10) & 1,
        "ME2PIPE1_RQ_PENDING": (grbm2 >> 11) & 1,
        "ME2PIPE2_RQ_PENDING": (grbm2 >> 12) & 1,
        "ME2PIPE3_RQ_PENDING": (grbm2 >> 13) & 1,
        "RLC_RQ_PENDING": (grbm2 >> 14) & 1,
        "RLC_BUSY": (grbm2 >> 24) & 1,
        "TC_BUSY": (grbm2 >> 25) & 1,
        "TCC_CC_RESIDENT": (grbm2 >> 26) & 1,
        "CPF_BUSY": (grbm2 >> 28) & 1,
        "CPC_BUSY": (grbm2 >> 29) & 1,
        "CPG_BUSY": (grbm2 >> 30) & 1,
    }
    for name, val in bits2.items():
        if val:
            log(f"  {name} = {val}")
    findings["grbm_status2_decoded"] = bits2

save_json()

# ============================================================
# STEP 2B: Disable GFXOFF to wake CP, then read SRAM
# ============================================================
log("\n--- STEP 2B: Wake CP from GFXOFF ---")

# Check GFXOFF status
gfxoff_path = f"{DEBUGFS}/amdgpu_gfxoff"
gfxoff_status_path = f"{DEBUGFS}/amdgpu_gfxoff_status"

try:
    status = open(gfxoff_status_path).read().strip()
    log(f"  GFXOFF status: {status}")
    findings["gfxoff_status_before"] = status
except Exception as e:
    log(f"  GFXOFF status read: {e}")

# Disable GFXOFF temporarily to wake CP
log("  Disabling GFXOFF (writing 0)...")
try:
    with open(gfxoff_path, "w") as f:
        f.write("0\n")
    time.sleep(0.5)  # Let GPU wake up

    status = open(gfxoff_status_path).read().strip()
    log(f"  GFXOFF status after disable: {status}")
    findings["gfxoff_status_disabled"] = status
except Exception as e:
    log(f"  GFXOFF disable: {e}")

# Now re-read CP registers with GPU awake
log("  Re-reading CP registers with GFXOFF disabled...")
cp_wake = {}

# GFX11 CP register offsets - try multiple known locations
# The register offsets in gfx11 may differ from earlier gens
regs_to_try = {
    # Standard GC offsets (dword addr * 4)
    "GRBM_STATUS": 0x8010,
    "GRBM_STATUS2": 0x8008,
    "CP_STAT": 0x8400,
    "CP_BUSY_STAT": 0x8410,
    "CP_STALLED_STAT1": 0x8414,
    "CP_STALLED_STAT2": 0x8418,
    "CP_STALLED_STAT3": 0x841C,
    # CP microcode registers
    "CP_MEC_ME1_UCODE_ADDR": 0xB680,
    "CP_MEC_ME1_UCODE_DATA": 0xB684,
    "CP_PFP_UCODE_ADDR": 0x8540,
    "CP_PFP_UCODE_DATA": 0x8544,
    "CP_ME_RAM_RADDR": 0x8554,
    "CP_ME_RAM_DATA": 0x8558,
    # RLC registers
    "RLC_CNTL": 0x13400 - 0xC00,  # try various bases
    "RLC_STAT": 0x13400,
    "RLC_SAFE_MODE": 0x1340C,
    "RLC_GPM_GENERAL_0": 0x13480,
    "RLC_GPM_GENERAL_1": 0x13484,
    "RLC_GPM_GENERAL_2": 0x13488,
    "RLC_GPM_GENERAL_3": 0x1348C,
    "RLC_GPM_GENERAL_4": 0x13490,
    "RLC_GPM_GENERAL_5": 0x13494,
    "RLC_GPM_GENERAL_6": 0x13498,
    "RLC_GPM_GENERAL_7": 0x1349C,
    # MES registers (gfx11)
    "CP_MES_CNTL": 0x10000,
    "CP_MES_HEADER_DUMP": 0x10028,
    "CP_MES_PRGRM_CNTR_START": 0x1002C,
    # CP_HQD (Hardware Queue Descriptor) - per-pipe
    "CP_HQD_ACTIVE": 0xC890,
    "CP_HQD_VMID": 0xC898,
    "CP_HQD_PQ_BASE": 0xC8B0,
    "CP_HQD_PQ_BASE_HI": 0xC8B4,
    "CP_HQD_PQ_RPTR": 0xC8C0,
    "CP_HQD_PQ_WPTR_LO": 0xC8D8,
    "CP_HQD_PQ_WPTR_HI": 0xC8DC,
    "CP_HQD_PQ_CONTROL": 0xC8E0,
}

for name, offset in regs_to_try.items():
    val = safe_reg_read(offset)
    if val is not None:
        cp_wake[name] = hex(val)
        if val != 0:
            log(f"  {name} ({hex(offset)}): {hex(val)}")
    else:
        cp_wake[name] = "ERROR"

findings["cp_regs_gfxoff_disabled"] = cp_wake
save_json()

# Now try sequential MEC SRAM dump
log("\n  Attempting MEC SRAM sequential read...")
mec_addr_reg = 0xB680
mec_data_reg = 0xB684

# Try setting address to 0 via write to regs file
# This is a CP register, NOT an SMU register, should be safe
try:
    with open(REGS_FILE, "r+b") as f:
        f.seek(mec_addr_reg)
        f.write(struct.pack("<I", 0))  # Set read address to 0
    log("  Set MEC UCODE_ADDR = 0")
    time.sleep(0.1)

    # Now read DATA register sequentially — it should auto-increment
    mec_sram = []
    for i in range(256):  # Read 256 dwords = 1KB
        val = safe_reg_read(mec_data_reg)
        if val is None:
            break
        mec_sram.append(val)

    nonzero = sum(1 for w in mec_sram if w != 0)
    log(f"  MEC SRAM: read {len(mec_sram)} dwords, {nonzero} nonzero")

    if nonzero > 0:
        findings["mec_sram_dump"] = [hex(w) for w in mec_sram]
        with open(CP_SRAM, "a") as f:
            f.write(f"\n\n=== MEC SRAM Dump (GFXOFF disabled) ===\n")
            for i, w in enumerate(mec_sram):
                if w != 0:
                    f.write(f"  [{i:4d}] {hex(w)}\n")
        log(f"  First nonzero words: {[hex(w) for w in mec_sram if w != 0][:20]}")
except Exception as e:
    log(f"  MEC SRAM dump error: {e}")

# Try PFP SRAM
log("  Attempting PFP SRAM read...")
pfp_addr_reg = 0x8540
pfp_data_reg = 0x8544
try:
    with open(REGS_FILE, "r+b") as f:
        f.seek(pfp_addr_reg)
        f.write(struct.pack("<I", 0))
    time.sleep(0.1)

    pfp_sram = []
    for i in range(256):
        val = safe_reg_read(pfp_data_reg)
        if val is None:
            break
        pfp_sram.append(val)

    nonzero = sum(1 for w in pfp_sram if w != 0)
    log(f"  PFP SRAM: read {len(pfp_sram)} dwords, {nonzero} nonzero")

    if nonzero > 0:
        findings["pfp_sram_dump"] = [hex(w) for w in pfp_sram]
        with open(CP_SRAM, "a") as f:
            f.write(f"\n\n=== PFP SRAM Dump ===\n")
            for i, w in enumerate(pfp_sram):
                if w != 0:
                    f.write(f"  [{i:4d}] {hex(w)}\n")
        log(f"  First nonzero PFP words: {[hex(w) for w in pfp_sram if w != 0][:20]}")
except Exception as e:
    log(f"  PFP SRAM dump error: {e}")

# Try ME SRAM
log("  Attempting ME SRAM read...")
me_raddr_reg = 0x8554
me_data_reg = 0x8558
try:
    with open(REGS_FILE, "r+b") as f:
        f.seek(me_raddr_reg)
        f.write(struct.pack("<I", 0))
    time.sleep(0.1)

    me_sram = []
    for i in range(256):
        val = safe_reg_read(me_data_reg)
        if val is None:
            break
        me_sram.append(val)

    nonzero = sum(1 for w in me_sram if w != 0)
    log(f"  ME SRAM: read {len(me_sram)} dwords, {nonzero} nonzero")

    if nonzero > 0:
        findings["me_sram_dump"] = [hex(w) for w in me_sram]
        with open(CP_SRAM, "a") as f:
            f.write(f"\n\n=== ME SRAM Dump ===\n")
            for i, w in enumerate(me_sram):
                if w != 0:
                    f.write(f"  [{i:4d}] {hex(w)}\n")
except Exception as e:
    log(f"  ME SRAM dump error: {e}")

save_json()

# ============================================================
# STEP 3: Decompress actual gfx11_5_1 firmware and search VRAM
# ============================================================
log("\n--- STEP 3: Decompress gfx11_5_1 firmware ---")

fw_dir = "/lib/firmware/amdgpu"
target_fws = {}

# Find all gc_11_5_1 firmware files
for pat in ["gc_11_5_1_*", "gc_11_5_2_*"]:
    for fpath in sorted(glob.glob(f"{fw_dir}/{pat}")):
        basename = os.path.basename(fpath)
        try:
            if fpath.endswith(".zst"):
                if zstandard:
                    dctx = zstandard.ZstdDecompressor()
                    with open(fpath, "rb") as f:
                        raw = dctx.decompress(f.read())
                else:
                    r = subprocess.run(["zstd", "-d", "-c", fpath], capture_output=True, timeout=10)
                    raw = r.stdout
            else:
                raw = open(fpath, "rb").read()

            target_fws[basename] = raw
            log(f"  {basename}: {len(raw)} bytes decompressed")

            # Show header
            if len(raw) >= 256:
                with open(FW_DUMP, "a") as f:
                    f.write(f"\n=== Disk FW: {basename} ({len(raw)} bytes) ===\n")
                    f.write(hexdump(raw, 0, 256) + "\n")
                    # Check for common firmware header
                    if raw[:4] == b'\x01\x00\x00\x00':
                        magic = struct.unpack_from("<I", raw, 0)[0]
                        hdr_size = struct.unpack_from("<I", raw, 4)[0]
                        ucode_size = struct.unpack_from("<I", raw, 8)[0]
                        ucode_ver = struct.unpack_from("<I", raw, 12)[0]
                        f.write(f"  Header: magic={hex(magic)} hdr_size={hdr_size} ucode_size={ucode_size} ucode_ver={hex(ucode_ver)}\n")
                        log(f"    header: magic={hex(magic)} hdr={hdr_size} ucode={ucode_size} ver={hex(ucode_ver)}")
        except Exception as e:
            log(f"  {basename}: {e}")

# Also look for sdma, rlc, psp firmware
for pat in ["sdma_6_1_*", "psp_14_0_*", "rlc_*"]:
    for fpath in sorted(glob.glob(f"{fw_dir}/{pat}")):
        basename = os.path.basename(fpath)
        try:
            if fpath.endswith(".zst"):
                if zstandard:
                    dctx = zstandard.ZstdDecompressor()
                    with open(fpath, "rb") as f:
                        raw = dctx.decompress(f.read())
                else:
                    r = subprocess.run(["zstd", "-d", "-c", fpath], capture_output=True, timeout=10)
                    raw = r.stdout
            else:
                raw = open(fpath, "rb").read()
            target_fws[basename] = raw
            log(f"  {basename}: {len(raw)} bytes")
        except Exception as e:
            log(f"  {basename}: {e}")

findings["decompressed_firmware"] = {k: len(v) for k, v in target_fws.items()}
save_json()

# ============================================================
# STEP 3B: Search VRAM for decompressed firmware signatures
# ============================================================
log("\n--- STEP 3B: Search VRAM for firmware code ---")

# For each firmware, extract multiple unique signatures
fw_sigs = {}
for name, data in target_fws.items():
    sigs = []
    # Extract 32-byte blocks from various positions
    for pos_frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        pos = int(len(data) * pos_frac)
        pos = pos & ~3  # align to dword
        sig = data[pos:pos+32]
        if len(sig) == 32 and sig != b'\x00' * 32:
            sigs.append((pos, sig))
    fw_sigs[name] = sigs

# Search interesting VRAM regions
# The driver loads firmware into VRAM GTT buffers before sending to CP
# These are typically in the first few GB
SEARCH_RANGES = [
    (0, 256 * 1024 * 1024, "First 256MB"),
    (256 * 1024 * 1024, 512 * 1024 * 1024, "256-512MB"),
]

CHUNK = 1024 * 1024
fw_matches = []

for range_start, range_end, range_name in SEARCH_RANGES:
    log(f"\n  Searching {range_name}...")
    for chunk_off in range(range_start, range_end, CHUNK):
        t = check_temp()
        if t > 85:
            log(f"  THERMAL PAUSE at {t:.1f}°C")
            time.sleep(30)

        vram_data = safe_vram_read(chunk_off, CHUNK)
        if vram_data is None or vram_data == b'\x00' * len(vram_data):
            continue

        for fw_name, sigs in fw_sigs.items():
            for sig_pos, sig in sigs:
                idx = vram_data.find(sig)
                if idx >= 0:
                    match_vram = chunk_off + idx
                    log(f"    MATCH: {fw_name} sig@{sig_pos} found at VRAM {hex(match_vram)}")
                    fw_matches.append({
                        "firmware": fw_name,
                        "sig_offset_in_fw": sig_pos,
                        "vram_offset": hex(match_vram),
                        "vram_offset_int": match_vram,
                    })

        if (chunk_off - range_start) % (64 * CHUNK) == 0 and chunk_off > range_start:
            log(f"    Scanned {(chunk_off-range_start)//(1024*1024)}MB of {range_name}")

findings["firmware_vram_matches"] = fw_matches
if fw_matches:
    log(f"\n  TOTAL MATCHES: {len(fw_matches)}")
    with open(FW_DUMP, "a") as f:
        f.write(f"\n=== Firmware Found in VRAM ===\n")
        for m in fw_matches:
            f.write(f"  {m['firmware']} at VRAM {m['vram_offset']} (sig@{m['sig_offset_in_fw']})\n")

    # For each match, try to dump the full firmware region from VRAM
    for m in fw_matches[:5]:  # limit to first 5
        fw_name = m["firmware"]
        fw_size = len(target_fws[fw_name])
        vram_off = m["vram_offset_int"]
        sig_in_fw = m["sig_offset_in_fw"]

        # Calculate where firmware starts in VRAM
        fw_start_vram = vram_off - sig_in_fw
        if fw_start_vram < 0:
            continue

        log(f"\n  Dumping {fw_name} from VRAM {hex(fw_start_vram)} ({fw_size} bytes)...")
        vram_fw = safe_vram_read(fw_start_vram, fw_size)
        if vram_fw and len(vram_fw) == fw_size:
            disk_fw = target_fws[fw_name]
            # Compare
            diffs = 0
            diff_positions = []
            for i in range(fw_size):
                if vram_fw[i] != disk_fw[i]:
                    diffs += 1
                    if len(diff_positions) < 20:
                        diff_positions.append((i, disk_fw[i], vram_fw[i]))

            pct = (1 - diffs / fw_size) * 100
            log(f"    Match: {pct:.2f}% identical ({diffs} different bytes)")

            m["comparison"] = {
                "identical_pct": pct,
                "diff_bytes": diffs,
                "total_bytes": fw_size,
                "first_diffs": [(hex(p), hex(d), hex(v)) for p, d, v in diff_positions],
            }

            if diffs > 0:
                log(f"    First diffs: {diff_positions[:5]}")
                with open(FW_DUMP, "a") as f:
                    f.write(f"\n=== VRAM vs Disk: {fw_name} ({pct:.2f}% match, {diffs} diffs) ===\n")
                    for pos, disk_byte, vram_byte in diff_positions:
                        f.write(f"  offset {hex(pos)}: disk={hex(disk_byte)} vram={hex(vram_byte)}\n")
else:
    log("  No firmware matches found in VRAM (0-512MB)")

save_json()

# ============================================================
# STEP 4: Extended register scan for firmware state
# ============================================================
log("\n--- STEP 4: Extended Register Scan ---")

# Scan known GC register ranges for nonzero values
interesting_ranges = [
    (0x8000, 0x8800, "CP/GFX"),       # CP registers
    (0xB600, 0xB800, "MEC"),            # MEC registers
    (0xC800, 0xCA00, "HQD"),            # Hardware Queue Descriptor
    (0x10000, 0x10100, "MES"),          # Micro Engine Scheduler
    (0x13400, 0x13600, "RLC"),          # Run Length Controller
    (0x15000, 0x15100, "SDMA"),         # SDMA
]

reg_scan_results = {}
for start, end, name in interesting_ranges:
    nonzero_regs = {}
    for offset in range(start, end, 4):
        val = safe_reg_read(offset)
        if val is not None and val != 0:
            nonzero_regs[hex(offset)] = hex(val)

    if nonzero_regs:
        log(f"  {name} ({hex(start)}-{hex(end)}): {len(nonzero_regs)} nonzero registers")
        for off, val in list(nonzero_regs.items())[:10]:
            log(f"    {off}: {val}")
    else:
        log(f"  {name} ({hex(start)}-{hex(end)}): all zero")

    reg_scan_results[name] = nonzero_regs

findings["extended_reg_scan"] = reg_scan_results
save_json()

with open(CP_SRAM, "a") as f:
    f.write(f"\n\n=== Extended Register Scan (GFXOFF disabled) ===\n")
    for name, regs in reg_scan_results.items():
        if regs:
            f.write(f"\n--- {name} ---\n")
            for off, val in regs.items():
                f.write(f"  {off}: {val}\n")

# ============================================================
# STEP 5: amdgpu_regs2 (SMC/extended registers)
# ============================================================
log("\n--- STEP 5: SMC Registers (read-only) ---")
try:
    smc_path = f"{DEBUGFS}/amdgpu_regs_smc"
    with open(smc_path, "rb") as f:
        # Read first 4KB of SMC register space
        smc_data = f.read(4096)
    nonzero = sum(1 for i in range(0, len(smc_data), 4)
                  if struct.unpack_from("<I", smc_data, i)[0] != 0)
    log(f"  SMC regs: {len(smc_data)} bytes, {nonzero} nonzero dwords")

    smc_regs = {}
    for i in range(0, min(len(smc_data), 4096), 4):
        val = struct.unpack_from("<I", smc_data, i)[0]
        if val != 0:
            smc_regs[hex(i)] = hex(val)

    if smc_regs:
        findings["smc_regs"] = smc_regs
        with open(CP_SRAM, "a") as f:
            f.write(f"\n\n=== SMC Registers (nonzero) ===\n")
            for off, val in list(smc_regs.items())[:50]:
                f.write(f"  {off}: {val}\n")
                log(f"    {off}: {val}")
except Exception as e:
    log(f"  SMC regs: {e}")

# PCIe registers
log("  Reading PCIe config space registers...")
try:
    pcie_path = f"{DEBUGFS}/amdgpu_regs_pcie"
    with open(pcie_path, "rb") as f:
        pcie_data = f.read(4096)
    nonzero = sum(1 for i in range(0, len(pcie_data), 4)
                  if struct.unpack_from("<I", pcie_data, i)[0] != 0)
    log(f"  PCIe regs: {len(pcie_data)} bytes, {nonzero} nonzero dwords")

    pcie_regs = {}
    for i in range(0, min(len(pcie_data), 256), 4):  # First 256 bytes
        val = struct.unpack_from("<I", pcie_data, i)[0]
        if val != 0:
            pcie_regs[hex(i)] = hex(val)

    findings["pcie_regs_sample"] = pcie_regs
except Exception as e:
    log(f"  PCIe regs: {e}")

save_json()

# ============================================================
# STEP 6: Re-enable GFXOFF
# ============================================================
log("\n--- STEP 6: Re-enable GFXOFF ---")
try:
    with open(gfxoff_path, "w") as f:
        f.write("1\n")
    log("  GFXOFF re-enabled")
except Exception as e:
    log(f"  GFXOFF re-enable: {e}")

# ============================================================
# STEP 7: GEM info — find allocated GPU buffers
# ============================================================
log("\n--- STEP 7: GEM buffer info ---")
try:
    gem = open(f"{DEBUGFS}/amdgpu_gem_info").read()
    lines = gem.strip().split('\n')
    log(f"  GEM info: {len(lines)} lines")
    findings["gem_info_lines"] = len(lines)
    findings["gem_info_sample"] = lines[:30]
    with open(FW_DUMP, "a") as f:
        f.write(f"\n=== GEM Buffer Info ({len(lines)} lines) ===\n")
        for l in lines[:100]:
            f.write(l + "\n")

    # Parse for VRAM allocations
    vram_allocs = [l for l in lines if 'vram' in l.lower()]
    log(f"  VRAM allocations: {len(vram_allocs)}")
    for l in vram_allocs[:10]:
        log(f"    {l.strip()}")
except Exception as e:
    log(f"  GEM info: {e}")

# ============================================================
# STEP 8: VRAM memory manager info
# ============================================================
log("\n--- STEP 8: VRAM MM info ---")
try:
    vram_mm = open(f"{DEBUGFS}/amdgpu_vram_mm").read()
    lines = vram_mm.strip().split('\n')
    log(f"  VRAM MM: {len(lines)} lines")
    findings["vram_mm_lines"] = len(lines)
    findings["vram_mm_sample"] = lines[:30]
    with open(FW_DUMP, "a") as f:
        f.write(f"\n=== VRAM Memory Manager ({len(lines)} lines) ===\n")
        for l in lines[:100]:
            f.write(l + "\n")
except Exception as e:
    log(f"  VRAM MM: {e}")

save_json()

# ============================================================
# Final Summary
# ============================================================
log("\n" + "=" * 70)
log("PART 2 SUMMARY")
log("=" * 70)
log(f"Temperature: {check_temp():.1f}°C")
log(f"Firmware files decompressed: {len(target_fws)}")
log(f"VRAM firmware matches: {len(fw_matches)}")
log(f"Nonzero CP regs (GFXOFF off): {sum(1 for v in cp_wake.values() if v not in ('0x0', 'ERROR'))}")
nonzero_extended = sum(len(v) for v in reg_scan_results.values())
log(f"Nonzero extended regs: {nonzero_extended}")

findings["part2_summary"] = {
    "fw_decompressed": len(target_fws),
    "vram_fw_matches": len(fw_matches),
    "cp_regs_nonzero": sum(1 for v in cp_wake.values() if v not in ('0x0', 'ERROR')),
    "extended_regs_nonzero": nonzero_extended,
}
save_json()
log("Part 2 complete.")
