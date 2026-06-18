#!/usr/bin/env python3
"""
z2347b: SAFE RS64 IC Base Probe — NO SRBM/GRBM bank switching
==============================================================
Logs after EVERY register read to pinpoint crashes.
Only uses direct read_reg() — no bank selection that could disturb MES.

Goal: Find where CP firmware lives in VRAM.
"""
import struct, os, sys, json, hashlib, time
from datetime import datetime
from pathlib import Path

DEBUGFS_REGS = "/sys/kernel/debug/dri/128/amdgpu_regs"
DEBUGFS_VRAM = "/sys/kernel/debug/dri/128/amdgpu_vram"
THERMAL      = "/sys/class/thermal/thermal_zone0/temp"
RESULTS_DIR  = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
FW_DIR       = Path("/lib/firmware/amdgpu")
LOG_FILE     = RESULTS_DIR / "z2347b_log.txt"

# Ensure unbuffered
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

def log(msg):
    """Write to both stdout and log file, flush immediately."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

def get_temp():
    try:
        with open(THERMAL) as f:
            return int(f.read().strip()) / 1000.0
    except:
        return -1.0

def check_temp(label=""):
    t = get_temp()
    log(f"TEMP {label}: {t:.1f}C")
    if t > 88.0:
        log(f"THERMAL ABORT at {t:.1f}C during {label}")
        sys.exit(1)
    return t

def read_reg_safe(dword_offset, name="?"):
    """Read single 32-bit MMIO register. NO PG_LOCK, NO bank switch."""
    byte_off = dword_offset * 4
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(byte_off)
            data = f.read(4)
            if len(data) < 4:
                log(f"  REG {name} (0x{dword_offset:04X}): SHORT READ ({len(data)} bytes)")
                return None
            val = struct.unpack("<I", data)[0]
            log(f"  REG {name} (0x{dword_offset:04X}): 0x{val:08X}")
            return val
    except Exception as e:
        log(f"  REG {name} (0x{dword_offset:04X}): EXCEPTION {e}")
        return None

def read_vram(offset, size):
    try:
        with open(DEBUGFS_VRAM, "rb") as f:
            f.seek(offset)
            return f.read(size)
    except Exception as e:
        log(f"  VRAM read at 0x{offset:X} size {size}: EXCEPTION {e}")
        return None

def write_vram(offset, data):
    try:
        with open(DEBUGFS_VRAM, "r+b") as f:
            f.seek(offset)
            f.write(data)
            return True
    except Exception as e:
        log(f"  VRAM write at 0x{offset:X}: EXCEPTION {e}")
        return False

# ─── STEP 0: Sanity check ────────────────────────────────────────────────────
def step0_sanity():
    log("=" * 60)
    log("STEP 0: Sanity check")
    log("=" * 60)

    for path in [DEBUGFS_REGS, DEBUGFS_VRAM]:
        exists = os.path.exists(path)
        log(f"  {path}: {'EXISTS' if exists else 'MISSING'}")
        if not exists:
            log("ABORT: debugfs not available")
            sys.exit(1)

    check_temp("step0")

    # Read one safe register (GRBM_STATUS) to verify access works
    log("  Reading GRBM_STATUS (0x2004) as sanity check...")
    val = read_reg_safe(0x2004, "GRBM_STATUS")
    if val is None:
        log("ABORT: Cannot read registers")
        sys.exit(1)
    log(f"  GRBM_STATUS = 0x{val:08X} — GPU is {'active' if val & 0x80000000 else 'idle'}")

    # Read 4 bytes from VRAM offset 0
    log("  Reading VRAM offset 0 (4 bytes) as sanity check...")
    data = read_vram(0, 4)
    if data is None:
        log("ABORT: Cannot read VRAM")
        sys.exit(1)
    log(f"  VRAM[0:4] = {data.hex()}")

    log("STEP 0: PASS")
    save_checkpoint("step0", {"status": "PASS", "grbm_status": f"0x{val:08X}"})

# ─── STEP 1: Read IC base registers (direct only, no bank switch) ────────────
def step1_ic_bases():
    log("=" * 60)
    log("STEP 1: Read IC/DC base registers (DIRECT only, no SRBM/GRBM)")
    log("=" * 60)
    check_temp("step1_start")

    # These are the IC base registers — should be safe to read directly
    ic_regs = [
        # (name, dword_offset)
        ("CP_PFP_IC_BASE_LO",   0x5840),
        ("CP_PFP_IC_BASE_HI",   0x5841),
        ("CP_PFP_IC_BASE_CNTL", 0x5842),
        ("CP_ME_IC_BASE_LO",    0x5844),
        ("CP_ME_IC_BASE_HI",    0x5845),
        ("CP_ME_IC_BASE_CNTL",  0x5846),
        ("CP_CPC_IC_BASE_LO",   0x584c),
        ("CP_CPC_IC_BASE_HI",   0x584d),
        ("CP_CPC_IC_BASE_CNTL", 0x584e),
        ("CP_MES_IC_BASE_LO",   0x5850),
        ("CP_MES_IC_BASE_HI",   0x5851),
        ("CP_MES_IC_BASE_CNTL", 0x5852),
    ]

    results = {}
    log("  --- IC Base Registers ---")
    for name, dword in ic_regs:
        time.sleep(0.05)  # 50ms between reads — gentle
        val = read_reg_safe(dword, name)
        results[name] = val

    check_temp("step1_ic_done")

    # DC base registers
    dc_regs = [
        ("CP_GFX_RS64_DC_BASE0_LO",  0x5863),
        ("CP_GFX_RS64_DC_BASE0_HI",  0x5865),
        ("CP_MEC_DC_BASE_LO",        0x5870),
        ("CP_MEC_DC_BASE_HI",        0x5871),
    ]

    log("  --- DC Base Registers ---")
    for name, dword in dc_regs:
        time.sleep(0.05)
        val = read_reg_safe(dword, name)
        results[name] = val

    check_temp("step1_dc_done")

    # Reconstruct addresses
    log("  --- Reconstructed IC Base Addresses ---")
    ic_addrs = {}
    for prefix in ["CP_PFP_IC", "CP_ME_IC", "CP_CPC_IC", "CP_MES_IC"]:
        lo = results.get(f"{prefix}_BASE_LO")
        hi = results.get(f"{prefix}_BASE_HI")
        if lo is not None and hi is not None:
            addr = ((hi & 0xFFFF) << 32) | (lo & 0xFFFFF000)
            ic_addrs[prefix] = addr
            log(f"  {prefix}: 0x{addr:012X} (lo=0x{lo:08X} hi=0x{hi:08X})")
        else:
            log(f"  {prefix}: UNREADABLE")

    dc_addrs = {}
    for lo_name, hi_name, label in [
        ("CP_GFX_RS64_DC_BASE0_LO", "CP_GFX_RS64_DC_BASE0_HI", "GFX_RS64_DC"),
        ("CP_MEC_DC_BASE_LO", "CP_MEC_DC_BASE_HI", "MEC_DC"),
    ]:
        lo = results.get(lo_name)
        hi = results.get(hi_name)
        if lo is not None and hi is not None:
            addr = ((hi & 0xFFFF) << 32) | lo
            dc_addrs[label] = addr
            log(f"  {label}: 0x{addr:012X}")

    log("STEP 1: DONE")
    save_checkpoint("step1", {
        "registers": {k: f"0x{v:08X}" if v is not None else None for k, v in results.items()},
        "ic_addrs": {k: f"0x{v:012X}" for k, v in ic_addrs.items()},
        "dc_addrs": {k: f"0x{v:012X}" for k, v in dc_addrs.items()},
    })
    return results, ic_addrs, dc_addrs


# ─── STEP 2: Read OP_CNTL (cache status) — ONE AT A TIME ─────────────────────
def step2_cache_status():
    log("=" * 60)
    log("STEP 2: Read OP_CNTL registers (cache prime/invalidate status)")
    log("  WARNING: These control cache behavior — reading ONE AT A TIME")
    log("=" * 60)
    check_temp("step2_start")

    # These are the potentially sensitive ones
    op_regs = [
        ("CP_PFP_IC_OP_CNTL", 0x5843),
        ("CP_ME_IC_OP_CNTL",  0x5847),
        # CPC and MES OP_CNTL are in different register range
        ("CP_CPC_IC_OP_CNTL", 0x297a),
        ("CP_MES_IC_OP_CNTL", 0x2820),
    ]

    results = {}
    for name, dword in op_regs:
        log(f"  About to read {name} at 0x{dword:04X}...")
        time.sleep(0.2)  # 200ms pause before each
        val = read_reg_safe(dword, name)
        results[name] = val
        if val is not None:
            inv = val & 1
            inv_complete = (val >> 1) & 1
            prime = (val >> 4) & 1
            primed = (val >> 5) & 1
            log(f"    INVALIDATE={inv} INV_COMPLETE={inv_complete} PRIME={prime} ICACHE_PRIMED={primed}")
        time.sleep(0.2)  # 200ms pause after each

    check_temp("step2_op_done")

    # DC OP_CNTL
    dc_op_regs = [
        ("CP_GFX_RS64_DC_OP_CNTL", 0x2a09),
        ("CP_GFX_RS64_DC_BASE_CNTL", 0x2a08),
        ("CP_MEC_DC_BASE_CNTL", 0x290b),
    ]

    log("  --- DC Control Registers ---")
    for name, dword in dc_op_regs:
        time.sleep(0.1)
        val = read_reg_safe(dword, name)
        results[name] = val

    check_temp("step2_dc_done")

    # Bootload status
    boot_regs = [
        ("RLC_RLCS_BOOTLOAD_STATUS", 0x4e82),
        ("RLC_RLCS_BOOTLOAD_ID_STATUS1", 0x4ecb),
        ("RLC_RLCS_BOOTLOAD_ID_STATUS2", 0x4ecc),
    ]

    log("  --- RLC Bootload Status ---")
    for name, dword in boot_regs:
        time.sleep(0.1)
        val = read_reg_safe(dword, name)
        results[name] = val
        if name == "RLC_RLCS_BOOTLOAD_STATUS" and val is not None:
            gfx_init = val & 1
            boot_complete = (val >> 31) & 1
            log(f"    GFX_INIT_DONE={gfx_init} BOOTLOAD_COMPLETE={boot_complete}")

    log("STEP 2: DONE")
    save_checkpoint("step2", {
        "registers": {k: f"0x{v:08X}" if v is not None else None for k, v in results.items()},
    })
    return results


# ─── STEP 3: Read VRAM at IC base addresses ──────────────────────────────────
def step3_vram_at_ic(ic_addrs, dc_addrs):
    log("=" * 60)
    log("STEP 3: Read VRAM at IC base addresses")
    log("=" * 60)
    check_temp("step3_start")

    # GPU VA → physical VRAM offset.
    # VRAM base from dmesg: 0x8000000000 (our GPU)
    VRAM_MC_BASE = 0x8000000000

    all_addrs = {}
    all_addrs.update(ic_addrs)
    all_addrs.update(dc_addrs)

    vram_findings = {}

    for name, gpu_va in sorted(all_addrs.items()):
        if gpu_va == 0:
            log(f"  {name}: GPU VA = 0x0 — not programmed, skip")
            continue

        log(f"  {name}: GPU VA = 0x{gpu_va:012X}")

        # Try subtracting MC base
        offsets = []
        if gpu_va >= VRAM_MC_BASE:
            phys = gpu_va - VRAM_MC_BASE
            offsets.append((f"VA - 0x{VRAM_MC_BASE:X}", phys))
        offsets.append(("raw", gpu_va))
        # Also try if the VA is just a direct offset
        if gpu_va < 16 * 1024**3:
            offsets.append(("direct", gpu_va))

        for method, phys_off in offsets:
            if phys_off > 16 * 1024**3:  # skip unreasonable
                continue

            log(f"    Trying [{method}] offset 0x{phys_off:X}...")
            time.sleep(0.1)
            data = read_vram(phys_off, 256)
            if data is None:
                log(f"    [{method}]: READ FAILED")
                continue

            if data == b'\x00' * len(data):
                log(f"    [{method}]: ALL ZEROS (256 bytes)")
                continue
            if data == b'\xff' * len(data):
                log(f"    [{method}]: ALL 0xFF (256 bytes)")
                continue

            non_zero = sum(1 for b in data if b != 0)
            sha = hashlib.sha256(data).hexdigest()[:16]
            log(f"    [{method}]: {non_zero}/256 non-zero, SHA256={sha}")
            log(f"    First 32 bytes: {data[:32].hex()}")

            # Check for ISA signatures
            # s_endpgm = 0xBF810000
            endpgm_count = 0
            for i in range(0, len(data) - 3, 4):
                dw = struct.unpack_from("<I", data, i)[0]
                if dw == 0xBF810000:
                    endpgm_count += 1
            if endpgm_count > 0:
                log(f"    *** FOUND {endpgm_count} s_endpgm instructions! Likely shader/FW code ***")

            vram_findings[f"{name}_{method}"] = {
                "gpu_va": f"0x{gpu_va:012X}",
                "phys_off": f"0x{phys_off:X}",
                "non_zero": non_zero,
                "sha256_16": sha,
                "first_32": data[:32].hex(),
                "endpgm_count": endpgm_count,
            }

        check_temp(f"step3_{name}")

    log("STEP 3: DONE")
    save_checkpoint("step3", {"vram_findings": vram_findings})
    return vram_findings


# ─── STEP 4: Compare VRAM content to disk firmware blobs ─────────────────────
def step4_fw_compare(ic_addrs, dc_addrs):
    log("=" * 60)
    log("STEP 4: Compare VRAM content to disk firmware blobs")
    log("=" * 60)
    check_temp("step4_start")

    VRAM_MC_BASE = 0x8000000000

    fw_blobs = {
        "MEC":  "gc_11_5_1_mec.bin",
        "PFP":  "gc_11_5_1_pfp.bin",
        "ME":   "gc_11_5_1_me.bin",
        "MES1": "gc_11_5_1_mes_2.bin",
        "MES0": "gc_11_5_1_mes1.bin",
        "RLC":  "gc_11_5_1_rlc.bin",
    }

    # Load firmware blobs from disk
    fw_data = {}
    for name, filename in fw_blobs.items():
        path = FW_DIR / filename
        if not path.exists():
            # Try .zst
            zst_path = FW_DIR / (filename + ".zst")
            if zst_path.exists():
                log(f"  {name}: found {zst_path}, decompressing...")
                try:
                    import zstandard
                    with open(zst_path, "rb") as f:
                        dctx = zstandard.ZstdDecompressor()
                        fw_data[name] = dctx.decompress(f.read())
                    log(f"  {name}: {len(fw_data[name])} bytes decompressed")
                except ImportError:
                    import subprocess
                    result = subprocess.run(["zstd", "-d", "-c", str(zst_path)],
                                          capture_output=True)
                    if result.returncode == 0:
                        fw_data[name] = result.stdout
                        log(f"  {name}: {len(fw_data[name])} bytes (via zstd CLI)")
                    else:
                        log(f"  {name}: CANNOT decompress {zst_path}")
            else:
                log(f"  {name}: {path} NOT FOUND")
        else:
            fw_data[name] = path.read_bytes()
            log(f"  {name}: {len(fw_data[name])} bytes from {path}")

    if not fw_data:
        log("  No firmware blobs loaded — skip comparison")
        save_checkpoint("step4", {"error": "no firmware blobs"})
        return {}

    check_temp("step4_blobs_loaded")

    # For each IC base address, read 4KB from VRAM and compare
    all_addrs = {}
    all_addrs.update(ic_addrs)
    all_addrs.update(dc_addrs)

    matches = {}

    for reg_name, gpu_va in sorted(all_addrs.items()):
        if gpu_va == 0:
            continue

        phys_off = gpu_va - VRAM_MC_BASE if gpu_va >= VRAM_MC_BASE else gpu_va
        if phys_off > 16 * 1024**3:
            continue

        log(f"  Reading 4KB at VRAM offset 0x{phys_off:X} for {reg_name}...")
        data = read_vram(phys_off, 4096)
        if data is None or data == b'\x00' * len(data) or data == b'\xff' * len(data):
            log(f"    Empty/unreadable — skip")
            continue

        vram_sha = hashlib.sha256(data).hexdigest()[:32]
        log(f"    VRAM SHA256={vram_sha}, {sum(1 for b in data if b)}/4096 non-zero")

        # Compare against each firmware blob
        for fw_name, blob in fw_data.items():
            # Parse firmware header to find code start
            if len(blob) < 64:
                continue

            header_size_dw = struct.unpack_from("<H", blob, 4)[0] if len(blob) > 5 else 0
            code_offsets = [0, header_size_dw * 4, 256, 512, 1024]

            for code_off in code_offsets:
                if code_off + 256 > len(blob):
                    continue
                fw_chunk = blob[code_off:code_off + 256]
                if fw_chunk == b'\x00' * 256:
                    continue

                # Compare first 256 bytes
                if data[:256] == fw_chunk:
                    log(f"    *** EXACT MATCH: {fw_name} at code_offset={code_off} ***")
                    matches[reg_name] = {
                        "fw": fw_name,
                        "code_offset": code_off,
                        "vram_offset": f"0x{phys_off:X}",
                    }
                    break

                # Partial match (first 32 bytes)
                if data[:32] == fw_chunk[:32]:
                    log(f"    ** PARTIAL MATCH (32B): {fw_name} at code_offset={code_off} **")
                    matches[reg_name] = {
                        "fw": fw_name,
                        "code_offset": code_off,
                        "vram_offset": f"0x{phys_off:X}",
                        "partial": True,
                    }
                    break

        check_temp(f"step4_{reg_name}")

    log("STEP 4: DONE")
    save_checkpoint("step4", {"matches": matches})
    return matches


# ─── STEP 5: Brute scan first 32MB of VRAM for firmware signatures ───────────
def step5_vram_scan():
    log("=" * 60)
    log("STEP 5: Brute-force VRAM scan for s_endpgm / $PS1 signatures")
    log("=" * 60)
    check_temp("step5_start")

    SCAN_MB = 32
    CHUNK = 256 * 1024  # 256KB chunks

    findings = []

    for offset in range(0, SCAN_MB * 1024 * 1024, CHUNK):
        if offset % (4 * 1024 * 1024) == 0:
            mb = offset / (1024 * 1024)
            log(f"  Scanning VRAM offset {mb:.0f}MB / {SCAN_MB}MB...")
            check_temp(f"step5_{mb:.0f}MB")

        data = read_vram(offset, CHUNK)
        if data is None:
            continue
        if data == b'\x00' * len(data):
            continue

        # Look for s_endpgm (0xBF810000)
        for i in range(0, len(data) - 3, 4):
            dw = struct.unpack_from("<I", data, i)[0]
            if dw == 0xBF810000:
                abs_off = offset + i
                # Read context around it
                ctx_start = max(0, i - 16)
                ctx = data[ctx_start:i + 20]
                findings.append({
                    "type": "s_endpgm",
                    "vram_offset": f"0x{abs_off:X}",
                    "context": ctx.hex(),
                })
                if len(findings) <= 10:
                    log(f"    s_endpgm at VRAM 0x{abs_off:X}")

        # Look for $PS1 marker
        ps1_pos = data.find(b"$PS1")
        if ps1_pos >= 0:
            abs_off = offset + ps1_pos
            log(f"    $PS1 header at VRAM 0x{abs_off:X}!")
            findings.append({
                "type": "$PS1",
                "vram_offset": f"0x{abs_off:X}",
            })

    log(f"  Total findings: {len(findings)} (s_endpgm: {sum(1 for f in findings if f['type']=='s_endpgm')}, $PS1: {sum(1 for f in findings if f['type']=='$PS1')})")
    log("STEP 5: DONE")
    save_checkpoint("step5", {
        "scan_mb": SCAN_MB,
        "total_findings": len(findings),
        "s_endpgm_count": sum(1 for f in findings if f['type'] == 's_endpgm'),
        "ps1_count": sum(1 for f in findings if f['type'] == '$PS1'),
        "first_10": findings[:10],
    })
    return findings


# ─── Checkpoint saving ────────────────────────────────────────────────────────
_checkpoints = {}
def save_checkpoint(step_name, data):
    """Save incremental JSON after each step."""
    _checkpoints[step_name] = data
    _checkpoints["last_completed_step"] = step_name
    _checkpoints["timestamp"] = datetime.now().isoformat()
    _checkpoints["temperature_C"] = get_temp()

    outpath = RESULTS_DIR / "z2347b_safe_ic_probe.json"
    with open(outpath, "w") as f:
        json.dump(_checkpoints, f, indent=2, default=str)
    log(f"  Checkpoint saved: {step_name}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    # Clear log
    LOG_FILE.write_text("")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("z2347b: SAFE RS64 IC Base Probe")
    log(f"PID: {os.getpid()}")
    log(f"Temperature: {get_temp():.1f}C")

    step0_sanity()
    time.sleep(1)

    _, ic_addrs, dc_addrs = step1_ic_bases()
    time.sleep(1)

    step2_cache_status()
    time.sleep(1)

    step3_vram_at_ic(ic_addrs, dc_addrs)
    time.sleep(1)

    step4_fw_compare(ic_addrs, dc_addrs)
    time.sleep(1)

    step5_vram_scan()

    log("=" * 60)
    log("ALL STEPS COMPLETE")
    log(f"Final temperature: {get_temp():.1f}C")
    log("=" * 60)


if __name__ == "__main__":
    main()
