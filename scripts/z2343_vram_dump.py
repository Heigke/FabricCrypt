#!/usr/bin/env python3
"""z2343: Dump decrypted/loaded firmware from GPU VRAM on gfx1151.
READ-ONLY. Saves results incrementally.
"""
import os, sys, json, time, struct, traceback
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

# From dmesg
VRAM_BASE = 0x8000000000
VRAM_END  = 0x97FFFFFFFF
VRAM_SIZE = VRAM_END - VRAM_BASE + 1
TMR_PHYS  = 0x97e0000000
TMR_SIZE  = 0x8c00000  # 140MB
TMR_OFFSET = TMR_PHYS - VRAM_BASE  # offset within VRAM file

findings = {
    "experiment": "z2343",
    "date": datetime.now().isoformat(),
    "gpu": "gfx1151 (Radeon 8060S)",
    "vram_base": hex(VRAM_BASE),
    "vram_size_mb": VRAM_SIZE // (1024*1024),
    "tmr_phys": hex(TMR_PHYS),
    "tmr_offset": hex(TMR_OFFSET),
    "tmr_size_mb": TMR_SIZE // (1024*1024),
    "steps": [],
    "firmware_found": [],
    "cp_sram": {},
}

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
        t = int(open(THERMAL).read().strip()) / 1000.0
        return t
    except:
        return 0.0

def safe_vram_read(offset, size, timeout=30):
    """Read VRAM at offset, with timeout protection."""
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
    except TimeoutError:
        log(f"  TIMEOUT reading VRAM at {hex(offset)}")
        signal.alarm(0)
        return None
    except Exception as e:
        log(f"  ERROR reading VRAM at {hex(offset)}: {e}")
        signal.alarm(0)
        return None
    finally:
        signal.signal(signal.SIGALRM, old)

def safe_reg_read(offset, timeout=10):
    """Read a single MMIO register via debugfs."""
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
    except TimeoutError:
        log(f"  TIMEOUT reading reg at {hex(offset)}")
        signal.alarm(0)
        return None
    except Exception as e:
        log(f"  ERROR reading reg at {hex(offset)}: {e}")
        signal.alarm(0)
        return None
    finally:
        signal.signal(signal.SIGALRM, old)

def hexdump(data, offset=0, limit=256):
    """Return hex dump string."""
    lines = []
    for i in range(0, min(len(data), limit), 16):
        hex_part = " ".join(f"{b:02x}" for b in data[i:i+16])
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in data[i:i+16])
        lines.append(f"  {offset+i:012x}: {hex_part:<48s}  {ascii_part}")
    return "\n".join(lines)

def find_pattern_in_data(data, pattern, name, base_offset=0):
    """Find all occurrences of pattern in data."""
    hits = []
    start = 0
    while True:
        idx = data.find(pattern, start)
        if idx == -1:
            break
        hits.append(base_offset + idx)
        start = idx + 1
    if hits:
        log(f"  Found {len(hits)} '{name}' at: {[hex(h) for h in hits[:10]]}")
    return hits

# ============================================================
# STEP 0: Basic checks
# ============================================================
log("=" * 70)
log("z2343: VRAM Firmware Dump — gfx1151 (Radeon 8060S)")
log("=" * 70)
log(f"Temperature: {check_temp():.1f}°C")
log(f"VRAM: {hex(VRAM_BASE)} - {hex(VRAM_END)} ({VRAM_SIZE//(1024**3)}GB)")
log(f"TMR:  {hex(TMR_PHYS)} (offset {hex(TMR_OFFSET)} in VRAM, {TMR_SIZE//(1024**2)}MB)")

# Check permissions
for path in [VRAM_FILE, REGS_FILE]:
    ok = os.access(path, os.R_OK)
    log(f"  {path}: {'readable' if ok else 'NOT readable'}")

# ============================================================
# STEP 1: Map firmware in GPU memory
# ============================================================
log("\n--- STEP 1: VRAM Firmware Search ---")

# 1a: Read firmware info from debugfs
log("Reading amdgpu_firmware_info...")
try:
    fw_info = open(f"{DEBUGFS}/amdgpu_firmware_info").read()
    log(f"  Got {len(fw_info)} bytes of firmware info")
    findings["firmware_info"] = fw_info
    with open(FW_DUMP, "w") as f:
        f.write("=== amdgpu_firmware_info ===\n")
        f.write(fw_info + "\n")
except Exception as e:
    log(f"  ERROR: {e}")

save_json()

# 1b: Probe VRAM at key offsets
log("\nProbing VRAM at key offsets...")
probe_offsets = [
    (0x0, "VRAM start"),
    (0x1000, "VRAM +4K"),
    (0x10000, "VRAM +64K"),
    (0x100000, "VRAM +1MB"),
    (0x1000000, "VRAM +16MB"),
    (0x10000000, "VRAM +256MB"),
    (TMR_OFFSET, "TMR base"),
    (TMR_OFFSET + 0x1000, "TMR +4K"),
    (TMR_OFFSET + 0x10000, "TMR +64K"),
    (TMR_OFFSET + 0x100000, "TMR +1MB"),
    (TMR_OFFSET + TMR_SIZE - 0x1000, "TMR end -4K"),
]

vram_probes = {}
for offset, label in probe_offsets:
    t = check_temp()
    if t > 85:
        log(f"  THERMAL PAUSE at {t:.1f}°C, waiting...")
        time.sleep(30)

    log(f"  Reading {label} ({hex(offset)})...")
    data = safe_vram_read(offset, 4096)
    if data is None:
        vram_probes[label] = "TIMEOUT/ERROR"
        continue

    # Check if all zeros or all FF
    if data == b'\x00' * len(data):
        vram_probes[label] = f"ALL ZEROS ({len(data)} bytes)"
        log(f"    -> all zeros")
    elif data == b'\xff' * len(data):
        vram_probes[label] = f"ALL 0xFF ({len(data)} bytes)"
        log(f"    -> all 0xFF")
    else:
        nonzero = sum(1 for b in data if b != 0)
        vram_probes[label] = f"{nonzero}/{len(data)} nonzero bytes"
        log(f"    -> {nonzero}/{len(data)} nonzero bytes")
        log(hexdump(data, offset, 128))

        # Check for firmware signatures
        find_pattern_in_data(data, b'\x24\x50\x53\x31', '$PS1', offset)  # PSP header
        find_pattern_in_data(data, struct.pack("<I", 0xBF810000), 'S_ENDPGM', offset)
        find_pattern_in_data(data, b'ATOMBIOS', 'ATOMBIOS', offset)

        # Save interesting data
        with open(FW_DUMP, "a") as f:
            f.write(f"\n=== {label} @ {hex(offset)} ===\n")
            f.write(hexdump(data, offset, 512) + "\n")

findings["vram_probes"] = vram_probes
findings["steps"].append({"step": "1b", "desc": "VRAM probes", "probes": len(vram_probes)})
save_json()

# 1c: Search for firmware signatures in VRAM
# Scan first 256MB in 1MB chunks looking for $PS1, S_ENDPGM patterns
log("\nSearching first 256MB of VRAM for firmware signatures...")
CHUNK = 1024 * 1024  # 1MB
SCAN_END = 256 * 1024 * 1024  # 256MB

ps1_hits = []
endpgm_regions = []
ascii_regions = []

PS1_MAGIC = b'\x24\x50\x53\x31'  # $PS1
ENDPGM_LE = struct.pack("<I", 0xBF810000)  # s_endpgm gfx10+

for chunk_off in range(0, SCAN_END, CHUNK):
    t = check_temp()
    if t > 85:
        log(f"  THERMAL PAUSE at {t:.1f}°C at offset {hex(chunk_off)}")
        time.sleep(30)

    data = safe_vram_read(chunk_off, CHUNK)
    if data is None:
        continue

    # $PS1 search
    hits = find_pattern_in_data(data, PS1_MAGIC, '$PS1', chunk_off)
    ps1_hits.extend(hits)

    # S_ENDPGM density (regions with lots = likely ISA code)
    endpgm_count = data.count(ENDPGM_LE)
    if endpgm_count > 10:
        endpgm_regions.append((chunk_off, endpgm_count))
        log(f"  S_ENDPGM dense region at {hex(chunk_off)}: {endpgm_count} hits")

    # ASCII strings (firmware logs?)
    ascii_count = sum(1 for i in range(0, len(data)-4, 4)
                      if all(32 <= data[i+j] < 127 for j in range(4)))
    if ascii_count > 1000:
        ascii_regions.append((chunk_off, ascii_count))

    if chunk_off % (32 * CHUNK) == 0:
        log(f"  Scanned {chunk_off // CHUNK}MB / {SCAN_END // CHUNK}MB, temp={t:.1f}°C")

findings["ps1_hits"] = [hex(h) for h in ps1_hits]
findings["endpgm_dense_regions"] = [(hex(o), c) for o, c in endpgm_regions]
findings["ascii_dense_regions"] = [(hex(o), c) for o, c in ascii_regions]
findings["steps"].append({
    "step": "1c", "desc": "VRAM signature scan (256MB)",
    "ps1_count": len(ps1_hits),
    "endpgm_regions": len(endpgm_regions),
    "ascii_regions": len(ascii_regions),
})
save_json()
log(f"  Total: {len(ps1_hits)} $PS1 headers, {len(endpgm_regions)} S_ENDPGM-dense regions, {len(ascii_regions)} ASCII-dense regions")

# 1d: Also scan TMR region specifically
log(f"\nScanning TMR region ({TMR_SIZE//(1024*1024)}MB at {hex(TMR_OFFSET)})...")
tmr_ps1 = []
tmr_endpgm = []
tmr_strings = []

for chunk_off in range(TMR_OFFSET, TMR_OFFSET + TMR_SIZE, CHUNK):
    t = check_temp()
    if t > 85:
        log(f"  THERMAL PAUSE at {t:.1f}°C")
        time.sleep(30)

    data = safe_vram_read(chunk_off, CHUNK)
    if data is None:
        continue

    if data == b'\x00' * len(data):
        continue  # skip zero pages

    hits = find_pattern_in_data(data, PS1_MAGIC, '$PS1-TMR', chunk_off)
    tmr_ps1.extend(hits)

    endpgm_count = data.count(ENDPGM_LE)
    if endpgm_count > 5:
        tmr_endpgm.append((chunk_off, endpgm_count))
        log(f"  TMR S_ENDPGM at {hex(chunk_off)}: {endpgm_count} hits")

    # Extract ASCII strings
    current_str = b""
    for i in range(len(data)):
        if 32 <= data[i] < 127:
            current_str += bytes([data[i]])
        else:
            if len(current_str) >= 8:
                tmr_strings.append((chunk_off + i - len(current_str), current_str.decode('ascii', errors='replace')))
            current_str = b""

    rel = chunk_off - TMR_OFFSET
    if rel % (16 * CHUNK) == 0:
        log(f"  TMR scan: {rel // CHUNK}MB / {TMR_SIZE // CHUNK}MB")

findings["tmr_ps1_hits"] = [hex(h) for h in tmr_ps1]
findings["tmr_endpgm_regions"] = [(hex(o), c) for o, c in tmr_endpgm]
findings["tmr_strings_sample"] = tmr_strings[:50]
findings["steps"].append({
    "step": "1d", "desc": "TMR scan",
    "ps1_count": len(tmr_ps1),
    "endpgm_regions": len(tmr_endpgm),
    "strings_found": len(tmr_strings),
})
save_json()

if tmr_strings:
    log(f"  TMR strings found: {len(tmr_strings)}")
    with open(FW_DUMP, "a") as f:
        f.write("\n=== TMR ASCII Strings ===\n")
        for off, s in tmr_strings[:100]:
            f.write(f"  {hex(off)}: {s}\n")

# ============================================================
# STEP 2: CP SRAM via MMIO registers
# ============================================================
log("\n--- STEP 2: CP SRAM Read Attempt ---")

# GFX11 CP register offsets (from amd kernel headers)
# These are MMIO byte offsets
# For gfx11, the register block is at GC base
# CP_MEC_ME1_UCODE_ADDR = 0x11B68 (regGC offset)
# CP_MEC_ME1_UCODE_DATA = 0x11B6C
# But debugfs amdgpu_regs uses byte offsets from MMIO base

# Common GFX11 CP register offsets (dword addresses * 4)
CP_REGS = {
    # MEC (compute)
    "CP_MEC_ME1_UCODE_ADDR": 0x2DA0 * 4,  # 0xB680
    "CP_MEC_ME1_UCODE_DATA": 0x2DA1 * 4,  # 0xB684
    "CP_MEC_ME2_UCODE_ADDR": 0x2DA2 * 4,
    "CP_MEC_ME2_UCODE_DATA": 0x2DA3 * 4,
    # PFP (pre-fetch parser)
    "CP_PFP_UCODE_ADDR": 0x2150 * 4,      # 0x8540
    "CP_PFP_UCODE_DATA": 0x2151 * 4,
    # ME (micro engine)
    "CP_ME_RAM_RADDR": 0x2155 * 4,
    "CP_ME_RAM_DATA": 0x2156 * 4,
    # CP status
    "CP_STAT": 0x2100 * 4,
    "CP_BUSY_STAT": 0x2104 * 4,
    "GRBM_STATUS": 0x2004 * 4,
    "GRBM_STATUS2": 0x2002 * 4,
    # RLC
    "RLC_STAT": 0x4D00 * 4,
    "RLC_GPU_IOV_VF_ENABLE": 0x4D08 * 4,
}

log("Attempting to read CP registers via amdgpu_regs...")
cp_results = {}
with open(CP_SRAM, "w") as f:
    f.write("=== CP SRAM / Register Reads ===\n\n")

for name, offset in CP_REGS.items():
    val = safe_reg_read(offset)
    if val is not None:
        cp_results[name] = hex(val)
        log(f"  {name} ({hex(offset)}): {hex(val)}")
    else:
        cp_results[name] = "ERROR/TIMEOUT"
        log(f"  {name} ({hex(offset)}): read failed")

findings["cp_registers"] = cp_results
save_json()

with open(CP_SRAM, "a") as f:
    f.write("\n--- CP Register Values ---\n")
    for name, val in cp_results.items():
        f.write(f"  {name}: {val}\n")

# Try to dump MEC SRAM via UCODE_ADDR/DATA pair
# Write ADDR=0, then read DATA repeatedly to get sequential words
# BUT: writing via amdgpu_regs could be unsafe...
# Actually amdgpu_regs does support writes (seek + write), but
# the MEMORY.md says NEVER write to SMU mailbox. CP_MEC_UCODE_ADDR
# is NOT SMU — it's the microcode address pointer. Still, let's be careful.
# We'll try reading the DATA register without setting ADDR first.
log("\nReading CP_MEC_ME1_UCODE_DATA at current ADDR (no write)...")
mec_data_off = CP_REGS["CP_MEC_ME1_UCODE_DATA"]
mec_words = []
for i in range(16):
    val = safe_reg_read(mec_data_off)
    if val is not None:
        mec_words.append(val)
    else:
        break

if mec_words:
    log(f"  Got {len(mec_words)} MEC words: {[hex(w) for w in mec_words]}")
    findings["mec_data_sample"] = [hex(w) for w in mec_words]
    with open(CP_SRAM, "a") as f:
        f.write("\n--- MEC UCODE DATA (current addr) ---\n")
        for i, w in enumerate(mec_words):
            f.write(f"  [{i:3d}] {hex(w)}\n")
save_json()

# ============================================================
# STEP 2b: Try UMR for CP registers
# ============================================================
log("\n--- STEP 2b: UMR CP Register Read ---")
import subprocess

umr_regs = [
    "amd1586.gc.mmCP_MEC_ME1_UCODE_ADDR",
    "amd1586.gc.mmCP_MEC_ME1_UCODE_DATA",
    "amd1586.gc.mmCP_STAT",
    "amd1586.gc.mmGRBM_STATUS",
    "amd1586.gc.mmGRBM_STATUS2",
    "amd1586.gc.mmCP_BUSY_STAT",
    "amd1586.gc.mmRLC_STAT",
    "amd1586.gc.mmCP_PFP_UCODE_ADDR",
    "amd1586.gc.mmCP_PFP_UCODE_DATA",
    "amd1586.gc.mmCP_ME_RAM_RADDR",
    "amd1586.gc.mmCP_ME_RAM_DATA",
]

umr_results = {}
for reg in umr_regs:
    try:
        r = subprocess.run(["umr", "-r", reg], capture_output=True, text=True, timeout=10)
        out = r.stdout.strip()
        umr_results[reg] = out
        log(f"  {reg}: {out}")
    except subprocess.TimeoutExpired:
        umr_results[reg] = "TIMEOUT"
        log(f"  {reg}: TIMEOUT")
    except Exception as e:
        umr_results[reg] = str(e)
        log(f"  {reg}: {e}")

findings["umr_cp_regs"] = umr_results
save_json()

with open(CP_SRAM, "a") as f:
    f.write("\n--- UMR CP Register Values ---\n")
    for reg, val in umr_results.items():
        f.write(f"  {reg}: {val}\n")

# ============================================================
# STEP 3: MQD and Ring Buffer Analysis
# ============================================================
log("\n--- STEP 3: MQD and Ring Buffer Analysis ---")

# Read MQD structures
mqd_files = [
    "amdgpu_mqd_comp_1.0.0",
    "amdgpu_mqd_gfx_0.0.0",
    "amdgpu_mqd_mes_3.0.0",
    "amdgpu_mqd_mes_kiq_3.1.0",
]

mqd_data = {}
for mqd in mqd_files:
    path = f"{DEBUGFS}/{mqd}"
    try:
        data = open(path, "rb").read()
        mqd_data[mqd] = {
            "size": len(data),
            "nonzero": sum(1 for b in data if b != 0),
            "hex_head": data[:128].hex(),
        }
        log(f"  {mqd}: {len(data)} bytes, {sum(1 for b in data if b != 0)} nonzero")
        with open(FW_DUMP, "a") as f:
            f.write(f"\n=== {mqd} ({len(data)} bytes) ===\n")
            f.write(hexdump(data, 0, 512) + "\n")
    except Exception as e:
        log(f"  {mqd}: {e}")
        mqd_data[mqd] = {"error": str(e)}

findings["mqd_structures"] = mqd_data
save_json()

# Read ring buffers (first 4KB of each)
ring_files = [
    "amdgpu_ring_comp_1.0.0",
    "amdgpu_ring_gfx_0.0.0",
    "amdgpu_ring_mes_3.0.0",
    "amdgpu_ring_mes_kiq_3.1.0",
    "amdgpu_ring_sdma0",
]

ring_data = {}
for ring in ring_files:
    path = f"{DEBUGFS}/{ring}"
    try:
        data = open(path, "r").read()
        lines = data.strip().split('\n')
        ring_data[ring] = {
            "lines": len(lines),
            "sample": lines[:20],
        }
        log(f"  {ring}: {len(lines)} lines")
        with open(FW_DUMP, "a") as f:
            f.write(f"\n=== {ring} (first 50 lines) ===\n")
            for l in lines[:50]:
                f.write(l + "\n")
    except Exception as e:
        log(f"  {ring}: {e}")
        ring_data[ring] = {"error": str(e)}

findings["ring_buffers"] = ring_data
save_json()

# ============================================================
# STEP 3b: GPR Wave state
# ============================================================
log("\n--- STEP 3b: Wave / GPR State ---")
for name in ["amdgpu_wave", "amdgpu_gprwave"]:
    path = f"{DEBUGFS}/{name}"
    try:
        data = open(path, "r").read()
        lines = data.strip().split('\n') if data.strip() else []
        log(f"  {name}: {len(lines)} lines")
        findings[name] = {"lines": len(lines), "sample": lines[:20]}
        with open(FW_DUMP, "a") as f:
            f.write(f"\n=== {name} ({len(lines)} lines) ===\n")
            for l in lines[:100]:
                f.write(l + "\n")
    except Exception as e:
        log(f"  {name}: {e}")

save_json()

# ============================================================
# STEP 4: Compare disk firmware vs VRAM
# ============================================================
log("\n--- STEP 4: Disk Firmware vs VRAM Comparison ---")

# Find MEC firmware on disk
fw_dir = "/lib/firmware/amdgpu"
import glob
fw_files = sorted(glob.glob(f"{fw_dir}/*1151*")) + sorted(glob.glob(f"{fw_dir}/*mec*"))
# Also try generic gfx11 names
fw_files += sorted(glob.glob(f"{fw_dir}/gc_11_5_*"))
fw_files = list(set(fw_files))

log(f"  Found {len(fw_files)} firmware files")
findings["disk_firmware_files"] = fw_files

disk_fw_patterns = {}
for fwf in fw_files[:20]:  # limit
    try:
        data = open(fwf, "rb").read()
        # Extract some unique patterns (first 64 bytes after header)
        header = data[:256]
        # PSP header is typically first 256 bytes
        # Actual code starts after
        pattern = data[256:320] if len(data) > 320 else data[:64]
        disk_fw_patterns[fwf] = {
            "size": len(data),
            "header_hex": header[:32].hex(),
            "code_pattern": pattern.hex(),
        }
        log(f"  {os.path.basename(fwf)}: {len(data)} bytes")
    except Exception as e:
        log(f"  {fwf}: {e}")

findings["disk_fw_patterns"] = disk_fw_patterns
save_json()

# Search for disk firmware code patterns in VRAM
# Use the S_ENDPGM-dense regions we found earlier as candidates
log("\nSearching for disk MEC code in VRAM S_ENDPGM regions...")
comparison_results = []

for fwf, info in disk_fw_patterns.items():
    if "mec" not in fwf.lower():
        continue

    fw_data = open(fwf, "rb").read()
    # Extract a 32-byte signature from the middle of the firmware
    mid = len(fw_data) // 2
    sig = fw_data[mid:mid+32]
    if len(sig) < 32:
        continue

    log(f"  Searching for {os.path.basename(fwf)} signature in VRAM...")

    # Search in S_ENDPGM dense regions
    for region_off, _ in endpgm_regions:
        data = safe_vram_read(region_off, CHUNK)
        if data is None:
            continue
        idx = data.find(sig)
        if idx >= 0:
            match_off = region_off + idx
            log(f"    MATCH at VRAM {hex(match_off)}!")
            comparison_results.append({
                "firmware": fwf,
                "vram_offset": hex(match_off),
                "match_type": "exact_32byte_signature",
            })

    # Also search in TMR
    for chunk_off in range(TMR_OFFSET, TMR_OFFSET + TMR_SIZE, CHUNK):
        data = safe_vram_read(chunk_off, CHUNK)
        if data is None:
            continue
        idx = data.find(sig)
        if idx >= 0:
            match_off = chunk_off + idx
            log(f"    MATCH in TMR at {hex(match_off)}!")
            comparison_results.append({
                "firmware": fwf,
                "vram_offset": hex(match_off),
                "match_type": "exact_32byte_in_TMR",
            })
        # Don't scan entire TMR for every file
        if chunk_off - TMR_OFFSET > 16 * CHUNK:
            break

findings["disk_vs_vram_comparison"] = comparison_results
findings["steps"].append({
    "step": "4", "desc": "Disk vs VRAM comparison",
    "matches": len(comparison_results),
})
save_json()

# ============================================================
# STEP 5: Firmware debug/trace buffers
# ============================================================
log("\n--- STEP 5: Firmware Debug/Trace Buffers ---")

# 5a: DMUB trace buffer (Display MicroController Unit B)
log("Reading DMUB trace buffer...")
try:
    dmub = open(f"{DEBUGFS}/amdgpu_dm_dmub_tracebuffer", "r").read()
    log(f"  DMUB tracebuffer: {len(dmub)} chars")
    findings["dmub_tracebuffer"] = dmub[:2000]
    with open(FW_DUMP, "a") as f:
        f.write(f"\n=== DMUB Trace Buffer ({len(dmub)} chars) ===\n")
        f.write(dmub[:4000] + "\n")
except Exception as e:
    log(f"  DMUB tracebuffer: {e}")

# 5b: DMUB firmware state
log("Reading DMUB firmware state...")
try:
    dmub_fw = open(f"{DEBUGFS}/amdgpu_dm_dmub_fw_state", "r").read()
    log(f"  DMUB fw state: {len(dmub_fw)} chars")
    findings["dmub_fw_state"] = dmub_fw[:2000]
    with open(FW_DUMP, "a") as f:
        f.write(f"\n=== DMUB FW State ({len(dmub_fw)} chars) ===\n")
        f.write(dmub_fw[:4000] + "\n")
except Exception as e:
    log(f"  DMUB fw state: {e}")

# 5c: Discovery table (IP discovery from VRAM)
log("Reading amdgpu_discovery...")
try:
    with open(f"{DEBUGFS}/amdgpu_discovery", "rb") as f:
        disc = f.read()
    log(f"  Discovery table: {len(disc)} bytes")
    findings["discovery_table_size"] = len(disc)
    with open(FW_DUMP, "a") as f2:
        f2.write(f"\n=== Discovery Table ({len(disc)} bytes) ===\n")
        f2.write(hexdump(disc, 0, 1024) + "\n")
    # Look for interesting strings
    strings = []
    current = b""
    for i, b in enumerate(disc):
        if 32 <= b < 127:
            current += bytes([b])
        else:
            if len(current) >= 4:
                strings.append((i - len(current), current.decode('ascii')))
            current = b""
    findings["discovery_strings"] = strings[:50]
    log(f"  Discovery strings: {len(strings)}")
except Exception as e:
    log(f"  Discovery table: {e}")

# 5d: GCA config
log("Reading amdgpu_gca_config...")
try:
    gca = open(f"{DEBUGFS}/amdgpu_gca_config", "r").read()
    log(f"  GCA config: {len(gca)} chars")
    findings["gca_config"] = gca[:2000]
    with open(FW_DUMP, "a") as f:
        f.write(f"\n=== GCA Config ===\n")
        f.write(gca + "\n")
except Exception as e:
    log(f"  GCA config: {e}")

# 5e: Fence info (timing)
log("Reading amdgpu_fence_info...")
try:
    fence = open(f"{DEBUGFS}/amdgpu_fence_info", "r").read()
    log(f"  Fence info: {len(fence)} chars")
    findings["fence_info"] = fence[:2000]
except Exception as e:
    log(f"  Fence info: {e}")

# 5f: iomem
log("Reading amdgpu_iomem...")
try:
    with open(f"{DEBUGFS}/amdgpu_iomem", "rb") as f:
        # Read first 4KB
        iomem = f.read(4096)
    nonzero = sum(1 for b in iomem if b != 0)
    log(f"  iomem: {len(iomem)} bytes, {nonzero} nonzero")
    if nonzero > 0:
        findings["iomem_sample"] = hexdump(iomem, 0, 256)
        with open(FW_DUMP, "a") as f2:
            f2.write(f"\n=== IOMEM (first 4K) ===\n")
            f2.write(hexdump(iomem, 0, 512) + "\n")
except Exception as e:
    log(f"  iomem: {e}")

save_json()

# ============================================================
# STEP 5f: Search VRAM for ASCII log buffers
# ============================================================
log("\nSearching VRAM for ASCII log/debug buffers in TMR...")
log_strings = []
# Check TMR region for log-like strings
SEARCH_PATTERNS = [b"ERROR", b"WARN", b"firmware", b"PSP", b"boot", b"init", b"version"]
pattern_hits = {p.decode(): [] for p in SEARCH_PATTERNS}

for chunk_off in range(TMR_OFFSET, TMR_OFFSET + min(TMR_SIZE, 32*CHUNK), CHUNK):
    data = safe_vram_read(chunk_off, CHUNK)
    if data is None or data == b'\x00' * len(data):
        continue
    for pat in SEARCH_PATTERNS:
        idx = 0
        while True:
            idx = data.find(pat, idx)
            if idx == -1:
                break
            # Extract surrounding context
            start = max(0, idx - 16)
            end = min(len(data), idx + 64)
            context = data[start:end]
            # Only count if surrounded by printable ASCII
            ascii_ctx = context.decode('ascii', errors='replace')
            pattern_hits[pat.decode()].append((hex(chunk_off + idx), ascii_ctx))
            idx += 1

for pat, hits in pattern_hits.items():
    if hits:
        log(f"  '{pat}' found {len(hits)} times in TMR")
        for off, ctx in hits[:3]:
            log(f"    {off}: {repr(ctx[:60])}")

findings["tmr_pattern_search"] = {k: len(v) for k, v in pattern_hits.items()}
save_json()

# ============================================================
# Final summary
# ============================================================
log("\n" + "=" * 70)
log("SUMMARY")
log("=" * 70)
temp_final = check_temp()
log(f"Final temperature: {temp_final:.1f}°C")
log(f"$PS1 headers in VRAM: {len(ps1_hits)}")
log(f"$PS1 headers in TMR: {len(tmr_ps1)}")
log(f"S_ENDPGM dense regions: {len(endpgm_regions)}")
log(f"ASCII strings in TMR: {len(tmr_strings)}")
log(f"Disk-vs-VRAM matches: {len(comparison_results)}")
log(f"MQD structures read: {sum(1 for v in mqd_data.values() if 'error' not in v)}")
log(f"CP registers read: {sum(1 for v in cp_results.values() if v != 'ERROR/TIMEOUT')}")

findings["summary"] = {
    "final_temp": temp_final,
    "ps1_in_vram": len(ps1_hits),
    "ps1_in_tmr": len(tmr_ps1),
    "endpgm_regions": len(endpgm_regions),
    "tmr_strings": len(tmr_strings),
    "disk_vram_matches": len(comparison_results),
    "mqd_read": sum(1 for v in mqd_data.values() if 'error' not in v),
    "cp_regs_read": sum(1 for v in cp_results.values() if v != 'ERROR/TIMEOUT'),
}
save_json()

log("\nResults saved to:")
log(f"  {LOG}")
log(f"  {FW_DUMP}")
log(f"  {CP_SRAM}")
log(f"  {JSON_OUT}")
log("Done.")
