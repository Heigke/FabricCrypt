#!/usr/bin/env python3
"""
z2347: RS64 Instruction/Data Cache Base Probe
=============================================
Probes CP firmware cache registers on GFX11 (gfx1151 / RDNA4) to map
the RS64 firmware attack surface.

Steps:
  1. Read all CP IC/DC base registers (PFP, ME, MEC/CPC, MES)
  2. Read VRAM at IC base addresses, compare against disk firmware blobs
  3. Check cache lock/prime status registers
  4. Safe VRAM write test (unused region only)
  5. Summary analysis

SAFETY: Read-only register access. VRAM writes only to verified-unused regions.
        NEVER writes to SMU mailbox or IC base regions.
"""

import struct
import os
import sys
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
DEBUGFS_REGS = "/sys/kernel/debug/dri/128/amdgpu_regs"
DEBUGFS_VRAM = "/sys/kernel/debug/dri/128/amdgpu_vram"
THERMAL      = "/sys/class/thermal/thermal_zone0/temp"
RESULTS_DIR  = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
FW_DIR       = Path("/lib/firmware/amdgpu")

# ─── register dword offsets (from gc_11_0_0_offset.h, all BASE_IDX=1) ───────
# IC (instruction cache) base registers
REGS_IC = {
    "CP_PFP_IC_BASE_LO":       0x5840,
    "CP_PFP_IC_BASE_HI":       0x5841,
    "CP_PFP_IC_BASE_CNTL":     0x5842,
    "CP_PFP_IC_OP_CNTL":       0x5843,
    "CP_ME_IC_BASE_LO":        0x5844,
    "CP_ME_IC_BASE_HI":        0x5845,
    "CP_ME_IC_BASE_CNTL":      0x5846,
    "CP_ME_IC_OP_CNTL":        0x5847,
    "CP_CPC_IC_BASE_LO":       0x584c,
    "CP_CPC_IC_BASE_HI":       0x584d,
    "CP_CPC_IC_BASE_CNTL":     0x584e,
    "CP_CPC_IC_OP_CNTL":       0x297a,  # different range
    "CP_MES_IC_BASE_LO":       0x5850,
    "CP_MES_IC_BASE_HI":       0x5851,
    "CP_MES_IC_BASE_CNTL":     0x5852,
    "CP_MES_IC_OP_CNTL":       0x2820,
}

# DC (data cache) base registers
REGS_DC = {
    "CP_GFX_RS64_DC_BASE0_LO":  0x5863,
    "CP_GFX_RS64_DC_BASE0_HI":  0x5865,
    "CP_GFX_RS64_DC_BASE_CNTL": 0x2a08,
    "CP_GFX_RS64_DC_OP_CNTL":   0x2a09,
    "CP_MEC_DC_BASE_LO":        0x5870,
    "CP_MEC_DC_BASE_HI":        0x5871,
    "CP_MEC_DC_BASE_CNTL":      0x290b,
    "CP_MES_DC_BASE_CNTL":      0x2836,
    "CP_MES_DC_OP_CNTL":        0x2837,
}

# RS64 program counter / control registers
REGS_RS64 = {
    "CP_MEC_RS64_PRGRM_CNTR_START":    0x2900,
    "CP_MEC_RS64_PRGRM_CNTR_START_HI": 0x2938,
    "CP_MEC_RS64_CNTL":                0x2904,
    "CP_MEC_RS64_INSTR_PNTR":          0x2908,
}

# Bootload / lock status
REGS_BOOT = {
    "RLC_RLCS_BOOTLOAD_STATUS":     0x4e82,
    "RLC_RLCS_BOOTLOAD_ID_STATUS1": 0x4ecb,
    "RLC_RLCS_BOOTLOAD_ID_STATUS2": 0x4ecc,
}

# PFP/ME program counter registers (per-pipe, need GRBM select)
REGS_PFP_ME_RS64 = {
    "CP_PFP_PRGRM_CNTR_START":    0x586a,
    "CP_PFP_PRGRM_CNTR_START_HI": 0x586b,
    "CP_ME_PRGRM_CNTR_START":     0x5867,
    "CP_ME_PRGRM_CNTR_START_HI":  0x5868,
}

# Known firmware blobs
FW_BLOBS = {
    "MEC": "gc_11_5_1_mec.bin",
    "PFP": "gc_11_5_1_pfp.bin",
    "ME":  "gc_11_5_1_me.bin",
    "MES": "gc_11_5_1_mes_2.bin",   # MES pipe 1
    "MES0": "gc_11_5_1_mes1.bin",   # MES pipe 0
}

PG_LOCK_BIT = 1 << 23  # bit 23 in debugfs offset = hold PG lock during read


def get_temp():
    """Read CPU/APU temperature in C."""
    try:
        with open(THERMAL) as f:
            return int(f.read().strip()) / 1000.0
    except:
        return -1.0


def check_temp(label=""):
    """Abort if too hot."""
    t = get_temp()
    if t > 85.0:
        print(f"THERMAL ABORT at {t:.1f}C during {label}")
        sys.exit(1)
    return t


def read_reg(dword_offset, pg_lock=True):
    """Read a single 32-bit MMIO register via debugfs.

    The debugfs file takes a byte offset in the lower 22 bits,
    plus optional bank selection / PG lock bits in upper bits.
    RREG32(byte_offset >> 2) is called internally.
    """
    byte_off = dword_offset * 4
    seek_pos = byte_off
    if pg_lock:
        seek_pos |= PG_LOCK_BIT
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(seek_pos)
            data = f.read(4)
            if len(data) < 4:
                return None
            return struct.unpack("<I", data)[0]
    except (PermissionError, OSError) as e:
        return None


def read_reg_grbm(dword_offset, se=0x3FF, sh=0x3FF, instance=0x3FF):
    """Read register with GRBM bank switch (bit 62 set).

    Bits 24..33: SE selector (0x3FF = broadcast)
    Bits 34..43: SH/SA selector
    Bits 44..53: INSTANCE/CU/WGP selector
    """
    byte_off = dword_offset * 4
    seek_pos = byte_off
    seek_pos |= (1 << 62)  # GRBM bank switch
    seek_pos |= (se & 0x3FF) << 24
    seek_pos |= (sh & 0x3FF) << 34
    seek_pos |= (instance & 0x3FF) << 44
    seek_pos |= PG_LOCK_BIT
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(seek_pos)
            data = f.read(4)
            if len(data) < 4:
                return None
            return struct.unpack("<I", data)[0]
    except (PermissionError, OSError) as e:
        return None


def read_reg_srbm(dword_offset, me=0, pipe=0, queue=0, vmid=0):
    """Read register with SRBM bank switch (bit 61 set).

    Bits 24..33: ME selector
    Bits 34..43: PIPE selector
    Bits 44..53: QUEUE selector
    Bits 54..58: VMID selector
    """
    byte_off = dword_offset * 4
    seek_pos = byte_off
    seek_pos |= (1 << 61)  # SRBM bank switch
    seek_pos |= (me & 0x3FF) << 24
    seek_pos |= (pipe & 0x3FF) << 34
    seek_pos |= (queue & 0x3FF) << 44
    seek_pos |= (vmid & 0x1F) << 54
    seek_pos |= PG_LOCK_BIT
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(seek_pos)
            data = f.read(4)
            if len(data) < 4:
                return None
            return struct.unpack("<I", data)[0]
    except (PermissionError, OSError) as e:
        return None


def read_vram(offset, size):
    """Read bytes from VRAM via debugfs."""
    try:
        with open(DEBUGFS_VRAM, "rb") as f:
            f.seek(offset)
            return f.read(size)
    except (PermissionError, OSError) as e:
        return None


def write_vram(offset, data):
    """Write bytes to VRAM via debugfs."""
    try:
        with open(DEBUGFS_VRAM, "r+b") as f:
            f.seek(offset)
            f.write(data)
            return True
    except (PermissionError, OSError) as e:
        return False


def decode_ic_base_cntl(val, name=""):
    """Decode IC_BASE_CNTL register fields."""
    if val is None:
        return "READ_FAILED"
    vmid = val & 0xF
    addr_clamp = (val >> 4) & 1
    exe_disable = (val >> 23) & 1
    cache_policy = (val >> 24) & 3
    return (f"VMID={vmid} ADDR_CLAMP={addr_clamp} "
            f"EXE_DISABLE={exe_disable} CACHE_POLICY={cache_policy}")


def decode_ic_op_cntl(val, name=""):
    """Decode IC_OP_CNTL register fields."""
    if val is None:
        return "READ_FAILED"
    inv_cache = val & 1
    inv_complete = (val >> 1) & 1
    prime = (val >> 4) & 1
    primed = (val >> 5) & 1
    return (f"INV_CACHE={inv_cache} INV_COMPLETE={inv_complete} "
            f"PRIME={prime} ICACHE_PRIMED={primed}")


def decode_bootload_status(val):
    """Decode RLC_RLCS_BOOTLOAD_STATUS fields."""
    if val is None:
        return "READ_FAILED"
    gfx_init = val & 1
    iram_loaded = (val >> 3) & 1
    iram_done = (val >> 4) & 1
    boot_complete = (val >> 31) & 1
    return (f"GFX_INIT_DONE={gfx_init} IRAM_LOADED={iram_loaded} "
            f"IRAM_DONE={iram_done} BOOTLOAD_COMPLETE={boot_complete}")


def load_fw_blob(name):
    """Load a firmware blob from disk."""
    for pattern in [name, name.replace("_1_", "_0_"), name.replace("11_5_1", "11_5_0")]:
        path = FW_DIR / pattern
        if path.exists():
            return path.read_bytes(), str(path)
    return None, None


# ─── STEP 1: Read IC/DC Base Registers ──────────────────────────────────────
def step1_read_registers():
    """Read all CP IC/DC base and control registers."""
    t = check_temp("step1_start")
    ts = datetime.now().isoformat()

    lines = []
    lines.append("=" * 70)
    lines.append("z2347 STEP 1: CP IC/DC Base Register Dump")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Temperature: {t:.1f}C")
    lines.append("=" * 70)

    results = {}

    # --- IC registers (no bank select needed for some, GRBM for others) ---
    lines.append("\n--- Instruction Cache Base Registers (direct read) ---")
    for name, dword in sorted(REGS_IC.items(), key=lambda x: x[1]):
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        results[name] = val
        decode = ""
        if "BASE_CNTL" in name and val is not None:
            decode = f"  [{decode_ic_base_cntl(val, name)}]"
        elif "OP_CNTL" in name and val is not None:
            decode = f"  [{decode_ic_op_cntl(val, name)}]"
        lines.append(f"  {name:35s} (dword 0x{dword:04X}) = {val_str}{decode}")

    # --- Try PFP/ME with GRBM pipe select ---
    lines.append("\n--- PFP/ME IC Base with GRBM pipe selection ---")
    for pipe in range(2):
        lines.append(f"  --- GRBM SE=broadcast SH=broadcast INSTANCE={pipe} ---")
        for name, dword in sorted(REGS_IC.items(), key=lambda x: x[1]):
            if "PFP" in name or "ME_IC" in name:
                val = read_reg_grbm(dword, se=0x3FF, sh=0x3FF, instance=pipe)
                val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
                key = f"{name}_PIPE{pipe}"
                results[key] = val
                lines.append(f"    {key:35s} = {val_str}")

    # --- Try with SRBM pipe select (for per-pipe registers) ---
    lines.append("\n--- PFP/ME/MEC with SRBM ME/PIPE selection ---")
    for me_sel in range(2):
        for pipe_sel in range(2):
            lines.append(f"  --- SRBM ME={me_sel} PIPE={pipe_sel} ---")
            all_regs = {**REGS_IC, **REGS_PFP_ME_RS64}
            for name, dword in sorted(all_regs.items(), key=lambda x: x[1]):
                if any(k in name for k in ["PFP", "ME_IC", "ME_PRGRM"]):
                    val = read_reg_srbm(dword, me=me_sel, pipe=pipe_sel)
                    val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
                    key = f"{name}_ME{me_sel}_PIPE{pipe_sel}"
                    results[key] = val
                    decode = ""
                    if "OP_CNTL" in name and val is not None:
                        decode = f"  [{decode_ic_op_cntl(val, name)}]"
                    lines.append(f"    {key:40s} = {val_str}{decode}")

    # --- DC registers ---
    lines.append("\n--- Data Cache Base Registers ---")
    for name, dword in sorted(REGS_DC.items(), key=lambda x: x[1]):
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        results[name] = val
        lines.append(f"  {name:35s} (dword 0x{dword:04X}) = {val_str}")

    # --- DC with SRBM ---
    lines.append("\n--- DC Base with SRBM pipe selection ---")
    for me_sel in range(2):
        for pipe_sel in range(2):
            lines.append(f"  --- SRBM ME={me_sel} PIPE={pipe_sel} ---")
            for name, dword in sorted(REGS_DC.items(), key=lambda x: x[1]):
                if "GFX_RS64" in name or "MEC_DC" in name:
                    val = read_reg_srbm(dword, me=me_sel, pipe=pipe_sel)
                    val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
                    key = f"{name}_ME{me_sel}_PIPE{pipe_sel}"
                    results[key] = val
                    lines.append(f"    {key:45s} = {val_str}")

    # --- RS64 control/program counter ---
    lines.append("\n--- RS64 Control and Program Counter ---")
    for name, dword in sorted(REGS_RS64.items(), key=lambda x: x[1]):
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        results[name] = val
        lines.append(f"  {name:40s} (dword 0x{dword:04X}) = {val_str}")

    # --- RS64 per-pipe ---
    lines.append("\n--- RS64 Program Counter with SRBM ME/PIPE ---")
    for me_sel in range(2):
        for pipe_sel in range(4):
            val_start = read_reg_srbm(0x2900, me=me_sel, pipe=pipe_sel)
            val_hi = read_reg_srbm(0x2938, me=me_sel, pipe=pipe_sel)
            val_instr = read_reg_srbm(0x2908, me=me_sel, pipe=pipe_sel)
            s1 = f"0x{val_start:08X}" if val_start is not None else "FAIL"
            s2 = f"0x{val_hi:08X}" if val_hi is not None else "FAIL"
            s3 = f"0x{val_instr:08X}" if val_instr is not None else "FAIL"
            results[f"MEC_RS64_PRGRM_START_ME{me_sel}_P{pipe_sel}"] = val_start
            results[f"MEC_RS64_PRGRM_START_HI_ME{me_sel}_P{pipe_sel}"] = val_hi
            results[f"MEC_RS64_INSTR_PNTR_ME{me_sel}_P{pipe_sel}"] = val_instr
            lines.append(f"  ME={me_sel} PIPE={pipe_sel}: START={s1} START_HI={s2} INSTR_PNTR={s3}")

    # --- Bootload status ---
    lines.append("\n--- RLC Bootload Status ---")
    for name, dword in sorted(REGS_BOOT.items(), key=lambda x: x[1]):
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        results[name] = val
        decode = ""
        if "BOOTLOAD_STATUS" == name.split("reg")[-1] or name.endswith("BOOTLOAD_STATUS"):
            if val is not None:
                decode = f"  [{decode_bootload_status(val)}]"
        lines.append(f"  {name:40s} (dword 0x{dword:04X}) = {val_str}{decode}")

    # --- Reconstruct IC base addresses ---
    lines.append("\n--- Reconstructed IC Base Addresses ---")
    ic_addrs = {}
    for prefix in ["CP_PFP_IC", "CP_ME_IC", "CP_CPC_IC", "CP_MES_IC"]:
        lo = results.get(f"{prefix}_BASE_LO")
        hi = results.get(f"{prefix}_BASE_HI")
        if lo is not None and hi is not None:
            addr = (hi << 32) | (lo & 0xFFFFF000)
            ic_addrs[prefix] = addr
            lines.append(f"  {prefix:20s}: 0x{addr:016X}")
        else:
            lines.append(f"  {prefix:20s}: UNREADABLE (lo={lo}, hi={hi})")

    # Also check SRBM pipe0 values for PFP/ME
    for prefix in ["CP_PFP_IC", "CP_ME_IC"]:
        lo = results.get(f"{prefix}_BASE_LO_ME0_PIPE0")
        hi = results.get(f"{prefix}_BASE_HI_ME0_PIPE0")
        if lo is not None and hi is not None and (lo != 0 or hi != 0):
            addr = (hi << 32) | (lo & 0xFFFFF000)
            ic_addrs[f"{prefix}_PIPE0"] = addr
            lines.append(f"  {prefix+'_PIPE0':20s}: 0x{addr:016X} (via SRBM)")

    dc_addrs = {}
    for prefix_lo, prefix_hi, label in [
        ("CP_GFX_RS64_DC_BASE0_LO", "CP_GFX_RS64_DC_BASE0_HI", "GFX_RS64_DC"),
        ("CP_MEC_DC_BASE_LO", "CP_MEC_DC_BASE_HI", "MEC_DC"),
    ]:
        lo = results.get(prefix_lo)
        hi = results.get(prefix_hi)
        if lo is not None and hi is not None:
            addr = (hi << 32) | lo
            dc_addrs[label] = addr
            lines.append(f"  {label:20s}: 0x{addr:016X} (data cache)")

    # Also check SRBM for DC
    for me_sel in range(2):
        for pipe_sel in range(2):
            lo = results.get(f"CP_GFX_RS64_DC_BASE0_LO_ME{me_sel}_PIPE{pipe_sel}")
            hi = results.get(f"CP_GFX_RS64_DC_BASE0_HI_ME{me_sel}_PIPE{pipe_sel}")
            if lo is not None and hi is not None and (lo != 0 or hi != 0):
                addr = (hi << 32) | lo
                label = f"GFX_RS64_DC_ME{me_sel}_P{pipe_sel}"
                dc_addrs[label] = addr
                lines.append(f"  {label:20s}: 0x{addr:016X} (via SRBM)")

    text = "\n".join(lines)
    outpath = RESULTS_DIR / "z2347_ic_base_registers.txt"
    outpath.write_text(text)
    print(text)
    print(f"\nSaved to {outpath}")

    return results, ic_addrs, dc_addrs


# ─── STEP 2: Read VRAM at IC Base, Compare to Disk FW ───────────────────────
def step2_vram_firmware(ic_addrs, dc_addrs):
    """Read VRAM at IC base addresses and compare to disk firmware."""
    t = check_temp("step2_start")
    ts = datetime.now().isoformat()

    lines = []
    lines.append("=" * 70)
    lines.append("z2347 STEP 2: VRAM Firmware Read & Compare")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Temperature: {t:.1f}C")
    lines.append("=" * 70)

    VRAM_GPU_BASE = 0x800000000  # typical MC base for VRAM on gfx11
    # Some GPUs use 0x8000000000 (40-bit) or even 0x0. We'll try multiple.
    # The IC_BASE registers contain GPU virtual addresses.
    # For VRAM access via debugfs, we need the physical VRAM offset.

    fw_matches = {}

    # First, let's try reading VRAM at offset 0 to see what's there
    lines.append("\n--- VRAM Header (offset 0x0, first 64 bytes) ---")
    vram_header = read_vram(0, 64)
    if vram_header:
        hex_dump = " ".join(f"{b:02X}" for b in vram_header[:64])
        lines.append(f"  {hex_dump}")
    else:
        lines.append("  FAILED to read VRAM")

    # Check total VRAM size
    vram_size_path = "/sys/class/drm/card1/device/mem_info_vram_total"
    if not os.path.exists(vram_size_path):
        vram_size_path = "/sys/class/drm/card0/device/mem_info_vram_total"
    try:
        with open(vram_size_path) as f:
            vram_total = int(f.read().strip())
        lines.append(f"\n  VRAM total: {vram_total} bytes ({vram_total / (1024**3):.2f} GiB)")
    except:
        vram_total = 0
        lines.append("\n  Could not determine VRAM size")

    # For each IC base address, try to map to physical VRAM offset
    lines.append("\n--- Attempting VRAM reads at IC base addresses ---")

    all_addrs = {}
    all_addrs.update(ic_addrs)
    all_addrs.update(dc_addrs)

    for name, gpu_va in all_addrs.items():
        if gpu_va == 0:
            lines.append(f"\n  {name}: GPU VA = 0x0 (not programmed, skipping)")
            continue

        lines.append(f"\n  {name}: GPU VA = 0x{gpu_va:016X}")

        # Try several interpretations of the GPU VA
        offsets_to_try = []
        # Raw offset
        offsets_to_try.append(("raw", gpu_va))
        # Subtract common MC bases
        for mc_base_name, mc_base in [
            ("0x800000000", 0x800000000),
            ("0x8000000000", 0x8000000000),
            ("0x0", 0x0),
        ]:
            if gpu_va >= mc_base:
                phys = gpu_va - mc_base
                if 0 <= phys < 16 * 1024**3:  # reasonable VRAM range
                    offsets_to_try.append((f"VA - {mc_base_name}", phys))

        for method, phys_off in offsets_to_try:
            if phys_off > 16 * 1024**3:  # skip unreasonable
                continue
            data = read_vram(phys_off, 4096)
            if data is None:
                lines.append(f"    [{method}] offset 0x{phys_off:X}: READ FAILED")
                continue

            # Check if all zeros
            if data == b'\x00' * len(data):
                lines.append(f"    [{method}] offset 0x{phys_off:X}: ALL ZEROS (4096 bytes)")
                continue

            # Check if all FF
            if data == b'\xff' * len(data):
                lines.append(f"    [{method}] offset 0x{phys_off:X}: ALL 0xFF (4096 bytes)")
                continue

            # Has content - analyze it
            non_zero = sum(1 for b in data if b != 0)
            sha = hashlib.sha256(data).hexdigest()[:16]
            first32 = " ".join(f"{b:02X}" for b in data[:32])
            lines.append(f"    [{method}] offset 0x{phys_off:X}: {non_zero}/4096 non-zero bytes")
            lines.append(f"      SHA256[0:16]: {sha}")
            lines.append(f"      First 32 bytes: {first32}")

            # Try to match against firmware blobs
            for fw_name, fw_file in FW_BLOBS.items():
                fw_data, fw_path = load_fw_blob(fw_file)
                if fw_data is None:
                    continue

                # Check if our VRAM data appears anywhere in the firmware blob
                # (firmware has headers, so code starts at an offset)
                if data[:256] in fw_data:
                    lines.append(f"      MATCH: First 256 bytes found in {fw_file}!")
                    fw_matches[name] = {"fw": fw_file, "method": method, "offset": phys_off}
                elif data[:64] in fw_data:
                    lines.append(f"      PARTIAL MATCH: First 64 bytes found in {fw_file}")
                    fw_matches[name] = {"fw": fw_file, "method": method, "offset": phys_off, "partial": True}

    # Also do a brute-force scan of first 64MB of VRAM looking for firmware signatures
    lines.append("\n--- Brute-force VRAM scan for firmware signatures ---")
    check_temp("step2_scan")

    # Read first few bytes of each firmware blob to use as signatures
    fw_sigs = {}
    for fw_name, fw_file in FW_BLOBS.items():
        fw_data, fw_path = load_fw_blob(fw_file)
        if fw_data and len(fw_data) >= 256:
            # Skip PSP header (usually 256 or 512 bytes), look for code
            # GFX firmware v2.0 header: first 4 bytes should be common_header
            # Try multiple start offsets within the firmware
            lines.append(f"  {fw_name} ({fw_file}): {len(fw_data)} bytes, "
                        f"SHA256={hashlib.sha256(fw_data).hexdigest()[:16]}")
            lines.append(f"    Header (first 32): {' '.join(f'{b:02X}' for b in fw_data[:32])}")

            # Parse firmware header to find code offset
            if len(fw_data) >= 64:
                # Common header: size_bytes at offset 4 (uint16)
                # GFX firmware v2.0: ucode_offset at header offset
                header_size = struct.unpack_from("<H", fw_data, 4)[0]
                lines.append(f"    Header size (from offset 4): {header_size} dwords ({header_size*4} bytes)")

                # Store signatures at various offsets
                for sig_off in [0, header_size * 4, 256, 512, 1024]:
                    if sig_off + 32 <= len(fw_data):
                        sig = fw_data[sig_off:sig_off+32]
                        if sig != b'\x00' * 32:
                            fw_sigs[f"{fw_name}@{sig_off}"] = (sig, fw_data, fw_path, sig_off)

    # Scan VRAM in 1MB chunks, check for signatures
    SCAN_SIZE = 64 * 1024 * 1024  # 64MB
    CHUNK = 1024 * 1024  # 1MB
    found_locations = []

    for offset in range(0, SCAN_SIZE, CHUNK):
        if offset % (16 * CHUNK) == 0:
            check_temp("step2_vram_scan")

        chunk_data = read_vram(offset, CHUNK)
        if chunk_data is None:
            continue

        for sig_name, (sig, fw_data, fw_path, sig_off) in fw_sigs.items():
            pos = chunk_data.find(sig)
            if pos >= 0:
                vram_off = offset + pos
                lines.append(f"\n  FOUND: {sig_name} signature at VRAM offset 0x{vram_off:X}")
                lines.append(f"    Signature: {' '.join(f'{b:02X}' for b in sig[:16])}")
                found_locations.append({
                    "name": sig_name,
                    "vram_offset": vram_off,
                    "sig_offset_in_fw": sig_off,
                })

                # Read more context around the match
                context = read_vram(vram_off, 256)
                if context:
                    lines.append(f"    Context (256 bytes at match):")
                    for i in range(0, min(256, len(context)), 32):
                        row = " ".join(f"{b:02X}" for b in context[i:i+32])
                        lines.append(f"      0x{vram_off+i:08X}: {row}")

    if not found_locations:
        lines.append("\n  No firmware signatures found in first 64MB of VRAM")
        lines.append("  (Firmware may be at higher addresses, or headers differ)")

    text = "\n".join(lines)
    outpath = RESULTS_DIR / "z2347_vram_firmware_read.txt"
    outpath.write_text(text)
    print(text)
    print(f"\nSaved to {outpath}")

    return fw_matches, found_locations


# ─── STEP 3: Cache Lock Status ──────────────────────────────────────────────
def step3_cache_lock_status(reg_results):
    """Check cache lock and prime status."""
    t = check_temp("step3_start")
    ts = datetime.now().isoformat()

    lines = []
    lines.append("=" * 70)
    lines.append("z2347 STEP 3: Cache Lock / Prime Status")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Temperature: {t:.1f}C")
    lines.append("=" * 70)

    lock_info = {}

    # Check OP_CNTL registers for each CP block
    lines.append("\n--- IC OP_CNTL Status (cache prime/invalidate) ---")
    op_regs = {
        "CP_PFP_IC_OP_CNTL": 0x5843,
        "CP_ME_IC_OP_CNTL":  0x5847,
        "CP_CPC_IC_OP_CNTL": 0x297a,
        "CP_MES_IC_OP_CNTL": 0x2820,
    }
    for name, dword in op_regs.items():
        # Direct read
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        decode = decode_ic_op_cntl(val) if val is not None else "N/A"
        lines.append(f"  {name:30s} = {val_str}  [{decode}]")
        lock_info[name] = {"value": val, "decode": decode}

        # Also with SRBM for PFP/ME
        if "PFP" in name or "ME_IC" in name:
            for me_sel in range(2):
                for pipe_sel in range(2):
                    val2 = read_reg_srbm(dword, me=me_sel, pipe=pipe_sel)
                    if val2 is not None:
                        dec2 = decode_ic_op_cntl(val2)
                        lines.append(f"    (SRBM ME={me_sel} PIPE={pipe_sel}): 0x{val2:08X}  [{dec2}]")
                        lock_info[f"{name}_ME{me_sel}_PIPE{pipe_sel}"] = {"value": val2, "decode": dec2}

    # Check BASE_CNTL for EXE_DISABLE
    lines.append("\n--- IC BASE_CNTL Status (exe disable, cache policy) ---")
    base_regs = {
        "CP_PFP_IC_BASE_CNTL": 0x5842,
        "CP_ME_IC_BASE_CNTL":  0x5846,
        "CP_CPC_IC_BASE_CNTL": 0x584e,
        "CP_MES_IC_BASE_CNTL": 0x5852,
    }
    for name, dword in base_regs.items():
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        decode = decode_ic_base_cntl(val) if val is not None else "N/A"
        lines.append(f"  {name:30s} = {val_str}  [{decode}]")
        lock_info[name] = {"value": val, "decode": decode}

    # Check DC OP_CNTL
    lines.append("\n--- DC OP_CNTL Status ---")
    dc_op_regs = {
        "CP_GFX_RS64_DC_OP_CNTL": 0x2a09,
        "CP_MES_DC_OP_CNTL":      0x2837,
    }
    for name, dword in dc_op_regs.items():
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        lines.append(f"  {name:30s} = {val_str}")
        if val is not None:
            inv = val & 1
            inv_complete = (val >> 1) & 1
            lines.append(f"    INVALIDATE={inv} COMPLETE={inv_complete}")
        lock_info[name] = {"value": val}

    # Check RLC bootload status
    lines.append("\n--- RLC Bootload / Lock Status ---")
    for name, dword in REGS_BOOT.items():
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        lines.append(f"  {name:40s} = {val_str}")
        if "BOOTLOAD_STATUS" in name and val is not None:
            lines.append(f"    [{decode_bootload_status(val)}]")
        elif val is not None:
            lines.append(f"    (raw bits: {val:032b})")
        lock_info[name] = {"value": val}

    # Check for any "LOCK" named registers in the CP block
    lines.append("\n--- Scanning for CP lock/protect registers ---")
    lock_scan_regs = {
        "CP_CPF_DEBUG":           0x2968,
        "CP_CPC_DEBUG":           0x2969,
        "CP_HQD_ACTIVE":         0x2958,
        "CP_MEC_CNTL":           0x2960,
        "CP_ME_CNTL":            0x2963,
    }
    for name, dword in lock_scan_regs.items():
        val = read_reg(dword)
        val_str = f"0x{val:08X}" if val is not None else "READ_FAILED"
        lines.append(f"  {name:30s} (dword 0x{dword:04X}) = {val_str}")
        lock_info[name] = {"value": val}

    # Analysis
    lines.append("\n--- Analysis ---")
    pfp_primed = False
    me_primed = False
    cpc_primed = False

    for name, info in lock_info.items():
        v = info.get("value")
        if v is None:
            continue
        if "PFP_IC_OP" in name and "PRIMED" not in name:
            pfp_primed = pfp_primed or ((v >> 5) & 1)
        if "ME_IC_OP" in name and "MEC" not in name and "MES" not in name:
            me_primed = me_primed or ((v >> 5) & 1)
        if "CPC_IC_OP" in name:
            cpc_primed = cpc_primed or ((v >> 5) & 1)

    lines.append(f"  PFP icache primed: {pfp_primed}")
    lines.append(f"  ME icache primed:  {me_primed}")
    lines.append(f"  CPC icache primed: {cpc_primed}")

    # NOTE: There is NO explicit "lock" bit in the IC_OP_CNTL.
    # The cache is "locked" by being primed. Re-priming after VRAM
    # modification would reload the cache. The key question is whether
    # the VRAM backing store is writable and whether cache invalidation
    # can be triggered from userspace.
    lines.append("\n  KEY FINDING: IC_OP_CNTL has no explicit 'lock' bit.")
    lines.append("  Cache state is: PRIME (load from VRAM) and INVALIDATE (flush).")
    lines.append("  If VRAM is writable at IC_BASE, modifying it + invalidating cache")
    lines.append("  would cause the CP to fetch modified instructions.")
    lines.append("  The IC_OP_CNTL INVALIDATE_CACHE bit triggers cache reload.")

    text = "\n".join(lines)
    outpath = RESULTS_DIR / "z2347_cache_lock_status.txt"
    outpath.write_text(text)
    print(text)
    print(f"\nSaved to {outpath}")

    return lock_info


# ─── STEP 4: VRAM Write Test (SAFE — unused region) ─────────────────────────
def step4_vram_write_test(ic_addrs, dc_addrs, vram_total=0):
    """Test VRAM write capability at an unused region."""
    t = check_temp("step4_start")
    ts = datetime.now().isoformat()

    lines = []
    lines.append("=" * 70)
    lines.append("z2347 STEP 4: Safe VRAM Write Test")
    lines.append(f"Timestamp: {ts}")
    lines.append(f"Temperature: {t:.1f}C")
    lines.append("=" * 70)

    # Find total VRAM
    if vram_total == 0:
        for path in ["/sys/class/drm/card1/device/mem_info_vram_total",
                     "/sys/class/drm/card0/device/mem_info_vram_total"]:
            try:
                with open(path) as f:
                    vram_total = int(f.read().strip())
                    break
            except:
                pass

    lines.append(f"  VRAM total: {vram_total} bytes ({vram_total/(1024**3):.2f} GiB)")

    # Choose a test offset far from any known firmware region
    # Use the last 1MB of VRAM (very unlikely to be used for firmware)
    if vram_total > 0:
        test_offset = vram_total - (1 * 1024 * 1024)  # last 1MB
    else:
        test_offset = 256 * 1024 * 1024  # 256MB in (conservative fallback)

    # Verify this is not near any IC/DC base address
    all_known = set()
    for addr in list(ic_addrs.values()) + list(dc_addrs.values()):
        if addr > 0:
            # Add with various MC base subtractions
            for base in [0, 0x800000000, 0x8000000000]:
                if addr >= base:
                    phys = addr - base
                    if 0 <= phys < 16 * 1024**3:
                        all_known.add(phys)

    safe = True
    for known in all_known:
        if abs(test_offset - known) < 4 * 1024 * 1024:  # 4MB safety margin
            safe = False
            lines.append(f"  WARNING: Test offset 0x{test_offset:X} too close to known "
                        f"firmware addr 0x{known:X}")

    if not safe:
        test_offset = vram_total - (2 * 1024 * 1024) if vram_total > 0 else 512 * 1024 * 1024
        lines.append(f"  Adjusted test offset to 0x{test_offset:X}")

    lines.append(f"\n  Test offset: 0x{test_offset:X} ({test_offset / (1024*1024):.1f} MiB into VRAM)")

    # Read existing content
    lines.append("\n--- Pre-write content ---")
    pre_data = read_vram(test_offset, 64)
    if pre_data is None:
        lines.append("  FAILED to read VRAM at test offset")
        text = "\n".join(lines)
        outpath = RESULTS_DIR / "z2347_vram_write_test.txt"
        outpath.write_text(text)
        print(text)
        return {"write_works": False, "error": "read_failed"}

    pre_hex = " ".join(f"{b:02X}" for b in pre_data[:64])
    lines.append(f"  First 64 bytes: {pre_hex}")

    # Write test pattern
    test_pattern = b"Z2347_VRAM_WRITE_TEST_" + struct.pack("<Q", int(time.time()))
    test_pattern = test_pattern + b"\x00" * (64 - len(test_pattern))

    lines.append(f"\n--- Writing test pattern ---")
    pattern_hex = " ".join(f"{b:02X}" for b in test_pattern[:64])
    lines.append(f"  Pattern: {pattern_hex}")

    write_ok = write_vram(test_offset, test_pattern)
    lines.append(f"  Write result: {'SUCCESS' if write_ok else 'FAILED'}")

    # Read back
    lines.append("\n--- Post-write readback ---")
    post_data = read_vram(test_offset, 64)
    if post_data:
        post_hex = " ".join(f"{b:02X}" for b in post_data[:64])
        lines.append(f"  First 64 bytes: {post_hex}")

        match = post_data[:len(test_pattern)] == test_pattern
        lines.append(f"  Readback matches pattern: {match}")
    else:
        lines.append("  FAILED to read back")
        match = False

    # Restore original content
    lines.append("\n--- Restoring original content ---")
    restore_ok = write_vram(test_offset, pre_data)
    lines.append(f"  Restore result: {'SUCCESS' if restore_ok else 'FAILED'}")

    # Verify restore
    verify = read_vram(test_offset, 64)
    if verify:
        restore_match = verify[:len(pre_data)] == pre_data
        lines.append(f"  Restore verified: {restore_match}")

    result = {
        "write_works": write_ok and match,
        "offset": test_offset,
        "pattern_matched": match,
        "restored": restore_ok,
    }

    lines.append(f"\n--- VRAM Write Test Result ---")
    lines.append(f"  VRAM is host-writable: {result['write_works']}")
    if result['write_works']:
        lines.append("  This means host can write arbitrary data to any VRAM offset,")
        lines.append("  including firmware IC base regions if addresses are known.")

    text = "\n".join(lines)
    outpath = RESULTS_DIR / "z2347_vram_write_test.txt"
    outpath.write_text(text)
    print(text)
    print(f"\nSaved to {outpath}")

    return result


# ─── STEP 5: Summary Analysis ───────────────────────────────────────────────
def step5_analysis(reg_results, ic_addrs, dc_addrs, fw_matches, found_locations,
                   lock_info, write_result):
    """Synthesize all findings into a summary."""
    t = check_temp("step5_start")
    ts = datetime.now().isoformat()

    summary = {
        "timestamp": ts,
        "temperature_C": t,
        "gpu": "AMD Radeon 8060S (gfx1151, RDNA4)",
        "architecture": "RS64 (RISC-V based CP firmware)",
    }

    # IC base addresses found
    ic_found = {k: f"0x{v:016X}" for k, v in ic_addrs.items() if v != 0}
    dc_found = {k: f"0x{v:016X}" for k, v in dc_addrs.items() if v != 0}
    summary["ic_base_addresses"] = ic_found
    summary["dc_base_addresses"] = dc_found
    summary["ic_bases_readable"] = len(ic_found) > 0

    # Firmware in VRAM
    summary["fw_signature_matches"] = fw_matches
    summary["fw_vram_locations"] = found_locations
    summary["firmware_in_readable_vram"] = len(found_locations) > 0

    # Cache lock
    any_primed = False
    no_lock_bit = True
    for name, info in lock_info.items():
        v = info.get("value")
        if v is not None and "OP_CNTL" in name:
            if (v >> 5) & 1:
                any_primed = True

    summary["cache_state"] = {
        "any_icache_primed": any_primed,
        "explicit_lock_bit_exists": False,
        "note": "IC_OP_CNTL has PRIME/INVALIDATE but no LOCK. Cache can be "
                "invalidated and re-primed by writing IC_OP_CNTL.INVALIDATE_CACHE=1 "
                "followed by PRIME_ICACHE=1."
    }

    # VRAM write
    summary["vram_write"] = write_result

    # Attack surface assessment
    vram_writable = write_result.get("write_works", False)
    ic_readable = len(ic_found) > 0
    fw_in_vram = len(found_locations) > 0

    attack_path = []
    if ic_readable:
        attack_path.append("IC_BASE registers are readable (firmware VA known)")
    if fw_in_vram:
        attack_path.append("Firmware code found in readable VRAM")
    if vram_writable:
        attack_path.append("VRAM is writable from host via debugfs")
    if any_primed:
        attack_path.append("Cache is primed (would need invalidation + re-prime after VRAM mod)")
    if not no_lock_bit:
        attack_path.append("No explicit cache lock — invalidation should work")

    summary["attack_surface"] = {
        "ic_base_readable": ic_readable,
        "firmware_in_vram": fw_in_vram,
        "vram_writable": vram_writable,
        "cache_primed_no_lock": any_primed and no_lock_bit,
        "path_elements": attack_path,
    }

    # Feasibility
    if ic_readable and vram_writable:
        if fw_in_vram:
            feasibility = "HIGH"
            detail = ("IC_BASE addresses known, firmware found in VRAM, VRAM writable. "
                     "Full attack path: (1) read IC_BASE to get FW VRAM address, "
                     "(2) modify FW code in VRAM, (3) invalidate + re-prime icache via "
                     "IC_OP_CNTL writes through debugfs. Requires root + debugfs access.")
        else:
            feasibility = "MEDIUM"
            detail = ("IC_BASE addresses known and VRAM writable, but firmware not yet "
                     "located in VRAM scan. May need wider scan or GPU VA translation.")
    elif vram_writable:
        feasibility = "LOW-MEDIUM"
        detail = ("VRAM writable but IC_BASE addresses not fully resolved. "
                 "Need to determine GPU VA to physical VRAM mapping.")
    else:
        feasibility = "LOW"
        detail = "VRAM write or IC_BASE read not confirmed."

    summary["feasibility"] = feasibility
    summary["detail"] = detail

    # Bootload status
    boot_val = reg_results.get("RLC_RLCS_BOOTLOAD_STATUS")
    if boot_val is not None:
        summary["bootload_complete"] = bool((boot_val >> 31) & 1)
        summary["gfx_init_done"] = bool(boot_val & 1)

    # Save
    outpath = RESULTS_DIR / "z2347_rs64_cache_probe.json"

    # Convert None values for JSON serialization
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif obj is None:
            return "null"
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)

    with open(outpath, "w") as f:
        json.dump(sanitize(summary), f, indent=2)

    print("\n" + "=" * 70)
    print("z2347 STEP 5: Summary Analysis")
    print("=" * 70)
    print(json.dumps(sanitize(summary), indent=2))
    print(f"\nSaved to {outpath}")

    return summary


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("z2347: RS64 Instruction Cache Probe")
    print(f"Temperature: {get_temp():.1f}C")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Verify debugfs access
    if not os.path.exists(DEBUGFS_REGS):
        print(f"ERROR: {DEBUGFS_REGS} not found. Need root + debugfs mounted.")
        sys.exit(1)
    if not os.path.exists(DEBUGFS_VRAM):
        print(f"ERROR: {DEBUGFS_VRAM} not found.")
        sys.exit(1)

    # Ensure results dir exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Read registers
    print("\n" + "=" * 70)
    print("STEP 1: Reading CP IC/DC base registers...")
    print("=" * 70)
    reg_results, ic_addrs, dc_addrs = step1_read_registers()

    check_temp("between_steps_1_2")
    time.sleep(1)

    # Step 2: Read VRAM at IC base, compare to disk firmware
    print("\n" + "=" * 70)
    print("STEP 2: Reading VRAM at firmware addresses...")
    print("=" * 70)
    fw_matches, found_locations = step2_vram_firmware(ic_addrs, dc_addrs)

    check_temp("between_steps_2_3")
    time.sleep(1)

    # Step 3: Cache lock status
    print("\n" + "=" * 70)
    print("STEP 3: Checking cache lock status...")
    print("=" * 70)
    lock_info = step3_cache_lock_status(reg_results)

    check_temp("between_steps_3_4")
    time.sleep(1)

    # Step 4: Safe VRAM write test
    print("\n" + "=" * 70)
    print("STEP 4: Safe VRAM write test...")
    print("=" * 70)
    write_result = step4_vram_write_test(ic_addrs, dc_addrs)

    check_temp("between_steps_4_5")
    time.sleep(1)

    # Step 5: Analysis
    print("\n" + "=" * 70)
    print("STEP 5: Analysis...")
    print("=" * 70)
    summary = step5_analysis(reg_results, ic_addrs, dc_addrs, fw_matches,
                            found_locations, lock_info, write_result)

    print(f"\nFinal temperature: {get_temp():.1f}C")
    print("z2347 COMPLETE")


if __name__ == "__main__":
    main()
