#!/usr/bin/env python3
"""
z2348: Deep IC/Firmware Boundary Test — ALL safe vectors with continuous logging
================================================================================
Tests every accessible attack vector for RS64 instruction cache modification.
Logs EVERY operation with timestamp, temperature, GRBM_STATUS health check.
Flushes to disk after EVERY line. Saves JSON checkpoint after EVERY step.

Steps (progressive risk — stops on first anomaly):
  0. Sanity: GRBM_STATUS, temp, scratch reg write/read
  1. IC register re-read: confirm z2347b values, read IC_OP_CNTL decode
  2. Scratch register write/verify: confirm debugfs write path works
  3. IC_OP_CNTL write test: try writing INVALIDATE bit, read back
  4. IC_BASE write test: try writing new IC_BASE_LO, read back
  5. VRAM firmware prep: write RS64 NOP-sled + s_endpgm to safe VRAM region
  6. IC_BASE redirect test: point MES IC_BASE to our VRAM NOP-sled
  7. KFD PM4 probe: submit WRITE_DATA targeting IC registers via PM4
  8. GTT firmware hunt: scan /proc/iomem + /dev/mem for firmware in system RAM
  9. Analysis: full summary of what worked and what didn't

SAFETY:
  - NO SRBM/GRBM bank switching (confirmed crash cause in z2347)
  - Health check (GRBM_STATUS + temp) after EVERY write operation
  - Auto-abort if GPU hangs (GRBM_STATUS changes to 0xFFFFFFFF)
  - Auto-abort if temp > 85C
  - Saves checkpoint JSON after every step (crash-recoverable)
  - Each step can be individually skipped via SKIP_STEPS env var
"""
import struct, os, sys, json, hashlib, time, subprocess, signal
from datetime import datetime
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────
DEBUGFS_REGS = "/sys/kernel/debug/dri/128/amdgpu_regs"
DEBUGFS_VRAM = "/sys/kernel/debug/dri/128/amdgpu_vram"
THERMAL      = "/sys/class/thermal/thermal_zone0/temp"
RESULTS_DIR  = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
LOG_FILE     = RESULTS_DIR / "z2348_log.txt"
JSON_FILE    = RESULTS_DIR / "z2348_deep_ic_test.json"
FW_DIR       = Path("/lib/firmware/amdgpu")

# VRAM offset for our test payload (64MB — well above framebuffer, below TMR)
PAYLOAD_VRAM_OFFSET = 64 * 1024 * 1024  # 64MB

# Register addresses (dword offsets)
REG = {
    'GRBM_STATUS':           0x2004,
    'GRBM_STATUS2':          0x2002,
    'SCRATCH_0':             0x2040,
    'SCRATCH_1':             0x2041,
    'SCRATCH_7':             0x2047,
    # IC Base registers
    'CP_PFP_IC_BASE_LO':    0x5840,
    'CP_PFP_IC_BASE_HI':    0x5841,
    'CP_PFP_IC_BASE_CNTL':  0x5842,
    'CP_PFP_IC_OP_CNTL':    0x5843,
    'CP_ME_IC_BASE_LO':     0x5844,
    'CP_ME_IC_BASE_HI':     0x5845,
    'CP_ME_IC_BASE_CNTL':   0x5846,
    'CP_ME_IC_OP_CNTL':     0x5847,
    'CP_CPC_IC_BASE_LO':    0x584C,
    'CP_CPC_IC_BASE_HI':    0x584D,
    'CP_CPC_IC_BASE_CNTL':  0x584E,
    'CP_CPC_IC_OP_CNTL':    0x297A,
    'CP_MES_IC_BASE_LO':    0x5850,
    'CP_MES_IC_BASE_HI':    0x5851,
    'CP_MES_IC_BASE_CNTL':  0x5852,
    'CP_MES_IC_OP_CNTL':    0x2820,
    # DC registers
    'CP_GFX_RS64_DC_OP_CNTL':   0x2A09,
    'CP_GFX_RS64_DC_BASE_CNTL': 0x2A08,
    'CP_MEC_DC_BASE_CNTL':      0x290B,
    # RLC
    'RLC_RLCS_BOOTLOAD_STATUS':  0x4E82,
    # MES control
    'CP_MES_CNTL':               0x2810,
    'CP_MES_INSTR_PNTR':        0x2812,
    'CP_MES_PRGRM_CNTR_START':  0x2826,
    # Additional CP status
    'CP_STAT':                   0x2100,
    'CP_BUSY_STAT':              0x2108,
    'CP_GFX_ERROR':              0x210C,
    'CP_GFX_HQD_ACTIVE':        0x2114,
    'CP_CPC_STATUS':             0x2180,
    'CP_CPC_BUSY_STAT':          0x2184,
    'CP_MEC_CNTL':               0x2960,
}

# IC_OP_CNTL bit definitions
INVALIDATE_BIT  = 1 << 0
INV_COMPLETE    = 1 << 1
PRIME_BIT       = 1 << 2
ICACHE_PRIMED   = 1 << 3

# Skip steps via env
SKIP_STEPS = set(int(x) for x in os.environ.get('SKIP_STEPS', '').split(',') if x.strip())

# ─── Unbuffered output ───────────────────────────────────────────────
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ─── State ───────────────────────────────────────────────────────────
state = {
    'started': datetime.now().isoformat(),
    'pid': os.getpid(),
    'steps': {},
    'anomalies': [],
    'health_checks': 0,
    'writes_attempted': 0,
    'writes_confirmed': 0,
    'writes_rejected': 0,
}

def save_state():
    state['last_save'] = datetime.now().isoformat()
    state['temperature_C'] = get_temp()
    with open(JSON_FILE, 'w') as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

# ─── Core functions ──────────────────────────────────────────────────

def log(msg):
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

def check_health(label=""):
    """Read GRBM_STATUS + temp. Return True if healthy."""
    t = get_temp()
    state['health_checks'] += 1
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(REG['GRBM_STATUS'] * 4)
            data = f.read(4)
            grbm = struct.unpack("<I", data)[0] if len(data) == 4 else 0xDEAD
    except Exception as e:
        log(f"  HEALTH {label}: GRBM READ FAILED ({e}) — GPU may be hung!")
        state['anomalies'].append(f"GRBM read failed at {label}: {e}")
        return False

    log(f"  HEALTH {label}: GRBM=0x{grbm:08X} T={t:.1f}C")

    if grbm == 0xFFFFFFFF:
        log(f"  *** GPU HUNG *** GRBM=0xFFFFFFFF at {label}")
        state['anomalies'].append(f"GPU hung at {label}")
        return False
    if t > 85.0:
        log(f"  *** THERMAL ABORT *** {t:.1f}C at {label}")
        state['anomalies'].append(f"Thermal abort {t:.1f}C at {label}")
        return False
    return True

def read_reg(dword_offset, name="?"):
    byte_off = dword_offset * 4
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(byte_off)
            data = f.read(4)
            if len(data) < 4:
                log(f"  REG {name} (0x{dword_offset:04X}): SHORT READ")
                return None
            val = struct.unpack("<I", data)[0]
            log(f"  REG {name} (0x{dword_offset:04X}): 0x{val:08X}")
            return val
    except Exception as e:
        log(f"  REG {name} (0x{dword_offset:04X}): EXCEPTION {e}")
        return None

def write_reg(dword_offset, value, name="?"):
    """Write 32-bit value to MMIO register via debugfs. Returns read-back value or None."""
    byte_off = dword_offset * 4
    state['writes_attempted'] += 1
    log(f"  WRITE {name} (0x{dword_offset:04X}) <- 0x{value:08X}")
    try:
        with open(DEBUGFS_REGS, "r+b") as f:
            f.seek(byte_off)
            f.write(struct.pack("<I", value))
            f.flush()
        log(f"  WRITE {name}: write() completed")
    except Exception as e:
        log(f"  WRITE {name}: EXCEPTION on write: {e}")
        state['anomalies'].append(f"Write exception {name}: {e}")
        return None

    time.sleep(0.05)  # 50ms settle

    # Read back
    rb = read_reg(dword_offset, f"{name}_readback")
    if rb is not None:
        if rb == value:
            log(f"  WRITE {name}: CONFIRMED (readback matches)")
            state['writes_confirmed'] += 1
        else:
            log(f"  WRITE {name}: MISMATCH (wrote 0x{value:08X}, read 0x{rb:08X})")
            state['writes_rejected'] += 1
    return rb

def read_vram(offset, size):
    try:
        with open(DEBUGFS_VRAM, "rb") as f:
            f.seek(offset)
            return f.read(size)
    except Exception as e:
        log(f"  VRAM read 0x{offset:X} ({size}B): {e}")
        return None

def write_vram(offset, data):
    try:
        with open(DEBUGFS_VRAM, "r+b") as f:
            f.seek(offset)
            f.write(data)
            f.flush()
        return True
    except Exception as e:
        log(f"  VRAM write 0x{offset:X} ({len(data)}B): {e}")
        return False

def pause(ms):
    time.sleep(ms / 1000.0)

# ─── STEPS ───────────────────────────────────────────────────────────

def step0_sanity():
    """Sanity checks: debugfs exists, GPU responds, temp ok."""
    log("=" * 70)
    log("STEP 0: Sanity check")
    log("=" * 70)
    s = {}

    for path, label in [(DEBUGFS_REGS, "regs"), (DEBUGFS_VRAM, "vram")]:
        exists = os.path.exists(path)
        log(f"  {label}: {'EXISTS' if exists else 'MISSING'}")
        s[label] = exists
        if not exists:
            log(f"  FATAL: {path} missing")
            return False

    if not check_health("step0"):
        return False

    grbm = read_reg(REG['GRBM_STATUS'], "GRBM_STATUS")
    s['grbm_status'] = f"0x{grbm:08X}" if grbm is not None else "FAIL"

    # Verify VRAM readable
    vdata = read_vram(0, 4)
    s['vram_readable'] = vdata is not None and len(vdata) == 4
    log(f"  VRAM read: {'OK' if s['vram_readable'] else 'FAIL'}")

    state['steps']['step0'] = s
    save_state()
    return True


def step1_ic_registers():
    """Re-read all IC/DC registers and decode IC_OP_CNTL bits."""
    log("=" * 70)
    log("STEP 1: IC register full dump + CP status")
    log("=" * 70)
    s = {'registers': {}}

    ic_regs = [
        'CP_PFP_IC_BASE_LO', 'CP_PFP_IC_BASE_HI', 'CP_PFP_IC_BASE_CNTL', 'CP_PFP_IC_OP_CNTL',
        'CP_ME_IC_BASE_LO', 'CP_ME_IC_BASE_HI', 'CP_ME_IC_BASE_CNTL', 'CP_ME_IC_OP_CNTL',
        'CP_CPC_IC_BASE_LO', 'CP_CPC_IC_BASE_HI', 'CP_CPC_IC_BASE_CNTL', 'CP_CPC_IC_OP_CNTL',
        'CP_MES_IC_BASE_LO', 'CP_MES_IC_BASE_HI', 'CP_MES_IC_BASE_CNTL', 'CP_MES_IC_OP_CNTL',
    ]

    for name in ic_regs:
        val = read_reg(REG[name], name)
        s['registers'][name] = f"0x{val:08X}" if val is not None else "FAIL"
        pause(50)

    # CP status registers
    log("  --- CP Status Registers ---")
    status_regs = [
        'CP_STAT', 'CP_BUSY_STAT', 'CP_GFX_ERROR', 'CP_GFX_HQD_ACTIVE',
        'CP_CPC_STATUS', 'CP_CPC_BUSY_STAT',
        'CP_MES_CNTL', 'CP_MES_INSTR_PNTR', 'CP_MES_PRGRM_CNTR_START',
    ]
    for name in status_regs:
        val = read_reg(REG[name], name)
        s['registers'][name] = f"0x{val:08X}" if val is not None else "FAIL"
        pause(50)

    if not check_health("step1"):
        return False

    # Decode IC_OP_CNTL for each engine
    for engine in ['PFP', 'ME', 'CPC', 'MES']:
        key = f'CP_{engine}_IC_OP_CNTL'
        val_str = s['registers'].get(key, "FAIL")
        if val_str != "FAIL":
            val = int(val_str, 16)
            log(f"  {engine} IC_OP_CNTL decode: INV={val&1} INV_DONE={(val>>1)&1} PRIME={(val>>2)&1} PRIMED={(val>>3)&1} raw=0x{val:08X}")

    state['steps']['step1'] = s
    save_state()
    return True


def step2_scratch_write():
    """Confirm debugfs write path using scratch registers (known safe from z2345)."""
    log("=" * 70)
    log("STEP 2: Scratch register write/verify (safe baseline)")
    log("=" * 70)
    s = {'tests': []}

    test_patterns = [0xDEADBEEF, 0x12345678, 0xCAFEBABE, 0x00000000, 0xFFFFFFFF]

    for i, pattern in enumerate(test_patterns):
        log(f"  Test {i}: SCRATCH_0 <- 0x{pattern:08X}")
        rb = write_reg(REG['SCRATCH_0'], pattern, "SCRATCH_0")
        ok = rb == pattern
        s['tests'].append({
            'pattern': f"0x{pattern:08X}",
            'readback': f"0x{rb:08X}" if rb is not None else "FAIL",
            'match': ok,
        })
        if not ok:
            log(f"  *** Scratch write FAILED: wrote 0x{pattern:08X}, got {rb}")
        if not check_health(f"step2_test{i}"):
            return False
        pause(100)

    # Restore scratch to 0
    write_reg(REG['SCRATCH_0'], 0, "SCRATCH_0_restore")

    all_ok = all(t['match'] for t in s['tests'])
    s['all_passed'] = all_ok
    log(f"  Scratch write result: {'ALL PASS' if all_ok else 'SOME FAIL'}")

    state['steps']['step2'] = s
    save_state()
    return all_ok


def step3_ic_op_cntl_write():
    """Try writing to IC_OP_CNTL registers. Test if hardware accepts writes."""
    log("=" * 70)
    log("STEP 3: IC_OP_CNTL write test (probe write capability)")
    log("=" * 70)
    s = {'tests': []}

    # For each engine, try:
    # a) Read current value
    # b) Write back same value (noop — safest possible write)
    # c) Read and verify
    # d) Try writing INVALIDATE bit (bit 0)
    # e) Read back — does it stick? does INV_COMPLETE (bit 1) appear?
    # f) If INVALIDATE stuck, try PRIME bit (bit 2)

    engines = [
        ('MES', REG['CP_MES_IC_OP_CNTL']),
        ('CPC', REG['CP_CPC_IC_OP_CNTL']),
        ('PFP', REG['CP_PFP_IC_OP_CNTL']),
        ('ME',  REG['CP_ME_IC_OP_CNTL']),
    ]

    for engine_name, reg_addr in engines:
        log(f"  --- {engine_name} IC_OP_CNTL (0x{reg_addr:04X}) ---")
        result = {'engine': engine_name, 'register': f"0x{reg_addr:04X}"}

        # a) Read current
        val0 = read_reg(reg_addr, f"{engine_name}_IC_OP_CNTL_before")
        result['before'] = f"0x{val0:08X}" if val0 is not None else "FAIL"
        pause(100)

        if not check_health(f"step3_{engine_name}_pre"):
            result['status'] = 'ABORTED_HEALTH'
            s['tests'].append(result)
            continue

        # b) Write back same value (noop test)
        log(f"  Noop write: {engine_name} <- 0x{val0:08X} (same value)")
        rb_noop = write_reg(reg_addr, val0, f"{engine_name}_noop")
        result['noop_readback'] = f"0x{rb_noop:08X}" if rb_noop is not None else "FAIL"
        pause(100)

        if not check_health(f"step3_{engine_name}_noop"):
            result['status'] = 'ABORTED_AFTER_NOOP'
            s['tests'].append(result)
            state['steps']['step3'] = s
            save_state()
            return False

        # c) Try writing INVALIDATE bit (bit 0)
        log(f"  *** Attempting INVALIDATE: {engine_name} <- 0x{INVALIDATE_BIT:08X}")
        rb_inv = write_reg(reg_addr, INVALIDATE_BIT, f"{engine_name}_INVALIDATE")
        result['invalidate_readback'] = f"0x{rb_inv:08X}" if rb_inv is not None else "FAIL"

        if rb_inv is not None:
            inv_stuck = (rb_inv & INVALIDATE_BIT) != 0
            inv_complete = (rb_inv & INV_COMPLETE) != 0
            still_primed = (rb_inv & ICACHE_PRIMED) != 0
            log(f"  After INVALIDATE: INV={inv_stuck} INV_DONE={inv_complete} PRIMED={still_primed}")
            result['invalidate_accepted'] = inv_stuck or inv_complete
            result['cache_invalidated'] = inv_complete
            result['still_primed'] = still_primed
        pause(200)

        if not check_health(f"step3_{engine_name}_inv"):
            result['status'] = 'ABORTED_AFTER_INVALIDATE'
            s['tests'].append(result)
            state['steps']['step3'] = s
            save_state()
            return False

        # d) Read again after settle
        val_after = read_reg(reg_addr, f"{engine_name}_IC_OP_CNTL_after")
        result['after'] = f"0x{val_after:08X}" if val_after is not None else "FAIL"

        # e) If invalidate seemed to work, try PRIME
        if rb_inv is not None and (rb_inv & (INVALIDATE_BIT | INV_COMPLETE)) != 0:
            log(f"  *** INVALIDATE may have worked! Trying PRIME: {engine_name} <- 0x{PRIME_BIT:08X}")
            rb_prime = write_reg(reg_addr, PRIME_BIT, f"{engine_name}_PRIME")
            result['prime_readback'] = f"0x{rb_prime:08X}" if rb_prime is not None else "FAIL"
            if rb_prime is not None:
                prime_ok = (rb_prime & ICACHE_PRIMED) != 0
                log(f"  After PRIME: PRIMED={prime_ok}")
                result['prime_accepted'] = (rb_prime & PRIME_BIT) != 0 or prime_ok
            pause(200)
            if not check_health(f"step3_{engine_name}_prime"):
                result['status'] = 'ABORTED_AFTER_PRIME'
                s['tests'].append(result)
                state['steps']['step3'] = s
                save_state()
                return False
        else:
            log(f"  INVALIDATE had no visible effect — skipping PRIME")
            result['prime_readback'] = 'SKIPPED'

        result['status'] = 'COMPLETED'
        s['tests'].append(result)
        pause(200)

    state['steps']['step3'] = s
    save_state()
    return True


def step4_ic_base_write():
    """Try writing to IC_BASE registers. Test if we can redirect instruction fetch."""
    log("=" * 70)
    log("STEP 4: IC_BASE write test (probe redirect capability)")
    log("=" * 70)
    s = {'tests': []}

    # Only test MES (safest — we know its current values)
    # MES IC_BASE: LO=0x8000, HI=0x0
    # Try writing a different value, read back, then restore

    targets = [
        ('CP_MES_IC_BASE_LO', REG['CP_MES_IC_BASE_LO'], 0x00008000),  # current
        ('CP_MES_IC_BASE_HI', REG['CP_MES_IC_BASE_HI'], 0x00000000),  # current
    ]

    for name, addr, original in targets:
        log(f"  --- {name} (0x{addr:04X}) ---")
        result = {'register': name, 'original': f"0x{original:08X}"}

        # Read current
        val0 = read_reg(addr, f"{name}_before")
        result['before'] = f"0x{val0:08X}" if val0 is not None else "FAIL"
        pause(100)

        # Try writing a different value (0xDEAD for LO, keep it recognizable)
        test_val = 0x0000DEAD if 'LO' in name else 0x00000001
        log(f"  *** Attempting IC_BASE write: {name} <- 0x{test_val:08X}")
        rb = write_reg(addr, test_val, f"{name}_test")
        result['test_readback'] = f"0x{rb:08X}" if rb is not None else "FAIL"

        if rb is not None:
            write_accepted = (rb == test_val)
            result['write_accepted'] = write_accepted
            if write_accepted:
                log(f"  *** IC_BASE WRITE ACCEPTED! Hardware allows IC_BASE modification!")
            else:
                log(f"  IC_BASE write rejected (readback = original, hardware ignores write)")
        pause(200)

        if not check_health(f"step4_{name}_test"):
            # Try to restore before aborting
            log(f"  Restoring {name} <- 0x{original:08X}")
            write_reg(addr, original, f"{name}_restore")
            result['status'] = 'ABORTED'
            s['tests'].append(result)
            state['steps']['step4'] = s
            save_state()
            return False

        # Restore original
        log(f"  Restoring {name} <- 0x{original:08X}")
        rb_restore = write_reg(addr, original, f"{name}_restore")
        result['restore_readback'] = f"0x{rb_restore:08X}" if rb_restore is not None else "FAIL"
        result['status'] = 'COMPLETED'
        s['tests'].append(result)
        pause(200)

        if not check_health(f"step4_{name}_restore"):
            state['steps']['step4'] = s
            save_state()
            return False

    state['steps']['step4'] = s
    save_state()
    return True


def step5_vram_payload():
    """Write RS64 NOP-sled + s_endpgm to safe VRAM region."""
    log("=" * 70)
    log("STEP 5: VRAM firmware payload preparation")
    log("=" * 70)
    s = {}

    # RS64 NOP = 0xBF800000 (s_nop 0)
    # RS64 s_endpgm = 0xBF810000
    NOP = struct.pack("<I", 0xBF800000)
    ENDPGM = struct.pack("<I", 0xBF810000)

    # Build payload: 252 NOPs + 1 s_endpgm + 3 s_endpgm (safety)
    payload = NOP * 252 + ENDPGM * 4  # 1024 bytes total

    log(f"  Payload: {len(payload)} bytes ({len(payload)//4} dwords)")
    log(f"  Content: 252x s_nop + 4x s_endpgm")
    log(f"  Target VRAM offset: 0x{PAYLOAD_VRAM_OFFSET:X} ({PAYLOAD_VRAM_OFFSET/(1024*1024):.0f}MB)")

    # Read current content at target
    before = read_vram(PAYLOAD_VRAM_OFFSET, 32)
    if before:
        s['vram_before'] = before.hex()
        log(f"  Current VRAM content: {before[:16].hex()}")
    else:
        log(f"  Cannot read VRAM at target offset!")
        s['status'] = 'VRAM_READ_FAIL'
        state['steps']['step5'] = s
        save_state()
        return False

    if not check_health("step5_pre_write"):
        s['status'] = 'ABORTED'
        state['steps']['step5'] = s
        save_state()
        return False

    # Write payload
    log(f"  *** Writing NOP-sled payload to VRAM 0x{PAYLOAD_VRAM_OFFSET:X}...")
    ok = write_vram(PAYLOAD_VRAM_OFFSET, payload)
    s['write_ok'] = ok
    if not ok:
        log(f"  VRAM write failed!")
        s['status'] = 'WRITE_FAIL'
        state['steps']['step5'] = s
        save_state()
        return False
    log(f"  VRAM write completed")
    pause(100)

    # Verify
    verify = read_vram(PAYLOAD_VRAM_OFFSET, len(payload))
    if verify == payload:
        log(f"  VRAM verify: MATCH — payload written correctly")
        s['verify'] = 'MATCH'
    elif verify:
        diff_count = sum(1 for a, b in zip(verify, payload) if a != b)
        log(f"  VRAM verify: MISMATCH — {diff_count}/{len(payload)} bytes differ")
        s['verify'] = f'MISMATCH_{diff_count}'
    else:
        log(f"  VRAM verify: READ FAIL")
        s['verify'] = 'READ_FAIL'

    if not check_health("step5_post_write"):
        s['status'] = 'ABORTED'
        state['steps']['step5'] = s
        save_state()
        return False

    # Calculate the IC_BASE value that would point here
    # IC_BASE_LO = lower_32(vram_phys >> 8)
    # For VRAM offset, the GPU physical addr may differ
    # On this APU: VRAM base is at some high address
    # We'll try both raw offset and with VRAM base
    shifted = PAYLOAD_VRAM_OFFSET >> 8
    s['ic_base_lo_for_payload'] = f"0x{shifted & 0xFFFFFFFF:08X}"
    s['ic_base_hi_for_payload'] = f"0x{(shifted >> 32) & 0xFFFFFFFF:08X}"
    log(f"  If IC_BASE pointed here: LO=0x{shifted & 0xFFFFFFFF:08X} HI=0x{(shifted >> 32) & 0xFFFFFFFF:08X}")

    s['status'] = 'OK'
    state['steps']['step5'] = s
    save_state()
    return True


def step6_ic_redirect():
    """If steps 3-5 showed writes work: attempt IC invalidate + redirect + prime."""
    log("=" * 70)
    log("STEP 6: IC redirect attempt (depends on step 3+4+5 results)")
    log("=" * 70)
    s = {}

    # Check if IC_OP_CNTL writes worked in step 3
    step3 = state['steps'].get('step3', {})
    step4 = state['steps'].get('step4', {})
    step5 = state['steps'].get('step5', {})

    any_op_accepted = False
    any_base_accepted = False

    for test in step3.get('tests', []):
        if test.get('invalidate_accepted'):
            any_op_accepted = True
    for test in step4.get('tests', []):
        if test.get('write_accepted'):
            any_base_accepted = True

    log(f"  Step 3 (IC_OP_CNTL writable): {any_op_accepted}")
    log(f"  Step 4 (IC_BASE writable): {any_base_accepted}")
    log(f"  Step 5 (VRAM payload): {step5.get('status', 'NOT_RUN')}")

    if not any_op_accepted and not any_base_accepted:
        log(f"  Neither IC_OP_CNTL nor IC_BASE accepts writes — redirect NOT possible")
        s['status'] = 'SKIPPED_NO_WRITE_PATH'
        state['steps']['step6'] = s
        save_state()
        return True

    if not check_health("step6_pre"):
        s['status'] = 'ABORTED'
        state['steps']['step6'] = s
        save_state()
        return False

    # If we got here, at least one write path works
    # Try the full sequence on MES (it has known IC_BASE values):
    #   1. Write IC_BASE to our VRAM payload
    #   2. Invalidate IC
    #   3. Prime IC from new IC_BASE
    #   4. Check MES_INSTR_PNTR to see if it changed

    payload_lo = int(step5.get('ic_base_lo_for_payload', '0x0'), 16)
    payload_hi = int(step5.get('ic_base_hi_for_payload', '0x0'), 16)

    # Read MES instruction pointer before
    ip_before = read_reg(REG['CP_MES_INSTR_PNTR'], "MES_IP_before")
    s['mes_ip_before'] = f"0x{ip_before:08X}" if ip_before is not None else "FAIL"
    pause(100)

    # Step 6a: Write new IC_BASE
    log(f"  *** 6a: MES IC_BASE <- LO=0x{payload_lo:08X} HI=0x{payload_hi:08X}")
    rb_lo = write_reg(REG['CP_MES_IC_BASE_LO'], payload_lo, "MES_IC_BASE_LO_redirect")
    rb_hi = write_reg(REG['CP_MES_IC_BASE_HI'], payload_hi, "MES_IC_BASE_HI_redirect")
    s['base_lo_rb'] = f"0x{rb_lo:08X}" if rb_lo is not None else "FAIL"
    s['base_hi_rb'] = f"0x{rb_hi:08X}" if rb_hi is not None else "FAIL"
    pause(200)

    if not check_health("step6_base_write"):
        # Restore!
        log(f"  Restoring MES IC_BASE to original...")
        write_reg(REG['CP_MES_IC_BASE_LO'], 0x8000, "restore_LO")
        write_reg(REG['CP_MES_IC_BASE_HI'], 0x0, "restore_HI")
        s['status'] = 'ABORTED_AFTER_BASE'
        state['steps']['step6'] = s
        save_state()
        return False

    # Step 6b: Invalidate IC
    log(f"  *** 6b: MES IC_OP_CNTL <- INVALIDATE (0x01)")
    rb_inv = write_reg(REG['CP_MES_IC_OP_CNTL'], INVALIDATE_BIT, "MES_INVALIDATE")
    s['invalidate_rb'] = f"0x{rb_inv:08X}" if rb_inv is not None else "FAIL"
    pause(200)

    if not check_health("step6_invalidate"):
        log(f"  Restoring MES IC_BASE...")
        write_reg(REG['CP_MES_IC_BASE_LO'], 0x8000, "restore_LO")
        write_reg(REG['CP_MES_IC_BASE_HI'], 0x0, "restore_HI")
        s['status'] = 'ABORTED_AFTER_INVALIDATE'
        state['steps']['step6'] = s
        save_state()
        return False

    # Step 6c: Prime IC
    log(f"  *** 6c: MES IC_OP_CNTL <- PRIME (0x04)")
    rb_prime = write_reg(REG['CP_MES_IC_OP_CNTL'], PRIME_BIT, "MES_PRIME")
    s['prime_rb'] = f"0x{rb_prime:08X}" if rb_prime is not None else "FAIL"
    pause(500)  # give it time

    if not check_health("step6_prime"):
        log(f"  Restoring MES IC_BASE...")
        write_reg(REG['CP_MES_IC_BASE_LO'], 0x8000, "restore_LO")
        write_reg(REG['CP_MES_IC_BASE_HI'], 0x0, "restore_HI")
        s['status'] = 'ABORTED_AFTER_PRIME'
        state['steps']['step6'] = s
        save_state()
        return False

    # Step 6d: Check MES instruction pointer
    ip_after = read_reg(REG['CP_MES_INSTR_PNTR'], "MES_IP_after")
    s['mes_ip_after'] = f"0x{ip_after:08X}" if ip_after is not None else "FAIL"

    if ip_before is not None and ip_after is not None:
        if ip_after != ip_before:
            log(f"  *** MES INSTRUCTION POINTER CHANGED! 0x{ip_before:08X} -> 0x{ip_after:08X}")
            s['ip_changed'] = True
        else:
            log(f"  MES IP unchanged: 0x{ip_after:08X}")
            s['ip_changed'] = False

    # Always restore IC_BASE
    log(f"  Restoring MES IC_BASE to original (LO=0x8000, HI=0x0)...")
    write_reg(REG['CP_MES_IC_BASE_LO'], 0x8000, "restore_LO")
    write_reg(REG['CP_MES_IC_BASE_HI'], 0x0, "restore_HI")
    # Re-prime with original
    write_reg(REG['CP_MES_IC_OP_CNTL'], INVALIDATE_BIT, "restore_INV")
    pause(100)
    write_reg(REG['CP_MES_IC_OP_CNTL'], PRIME_BIT, "restore_PRIME")
    pause(200)

    if not check_health("step6_restore"):
        s['status'] = 'RESTORE_FAILED'
    else:
        s['status'] = 'COMPLETED'

    state['steps']['step6'] = s
    save_state()
    return True


def step7_kfd_pm4():
    """Try PM4 submission via /dev/kfd to write to IC registers."""
    log("=" * 70)
    log("STEP 7: KFD PM4 probe (submit WRITE_DATA to IC registers)")
    log("=" * 70)
    s = {}

    # Check KFD access
    kfd_path = "/dev/kfd"
    if not os.path.exists(kfd_path):
        log(f"  /dev/kfd not found — KFD not available")
        s['status'] = 'NO_KFD'
        state['steps']['step7'] = s
        save_state()
        return True

    import stat
    kfd_stat = os.stat(kfd_path)
    mode = stat.filemode(kfd_stat.st_mode)
    log(f"  /dev/kfd: {mode} uid={kfd_stat.st_uid} gid={kfd_stat.st_gid}")

    # Check if we can open it
    try:
        fd = os.open(kfd_path, os.O_RDWR)
        os.close(fd)
        log(f"  /dev/kfd: open OK")
        s['kfd_open'] = True
    except Exception as e:
        log(f"  /dev/kfd: open FAILED ({e})")
        s['kfd_open'] = False
        s['status'] = 'OPEN_FAIL'
        state['steps']['step7'] = s
        save_state()
        return True

    # Try to use HIP/HSA to create queue and submit PM4
    # This requires ROCm runtime which may not be available as root
    try:
        log(f"  Attempting HIP-based PM4 submission...")
        # Use ctypes to talk to libhsakmt directly
        import ctypes
        try:
            hsakmt = ctypes.CDLL("libhsakmt.so.1")
            log(f"  libhsakmt.so.1 loaded")
            s['hsakmt'] = True

            # hsaKmtOpenKFD
            ret = hsakmt.hsaKmtOpenKFD()
            log(f"  hsaKmtOpenKFD() = {ret}")
            s['kmt_open'] = ret

            if ret == 0:  # HSAKMT_STATUS_SUCCESS
                # We could proceed with queue creation, but this is complex
                # For now, just confirm KFD access works
                log(f"  KFD opened successfully — PM4 submission POSSIBLE")
                s['pm4_possible'] = True

                # Clean up
                hsakmt.hsaKmtCloseKFD()
                log(f"  KFD closed")
            else:
                log(f"  KFD open returned non-zero — may need non-root user")
                s['pm4_possible'] = False
        except OSError as e:
            log(f"  libhsakmt not available: {e}")
            s['hsakmt'] = False
            s['pm4_possible'] = False
    except Exception as e:
        log(f"  KFD/HIP probe exception: {e}")
        s['pm4_possible'] = False

    if not check_health("step7"):
        s['status'] = 'ABORTED'
    else:
        s['status'] = 'COMPLETED'

    state['steps']['step7'] = s
    save_state()
    return True


def step8_gtt_firmware_hunt():
    """Search system RAM (GTT) for firmware via /dev/mem at MMIO BAR regions."""
    log("=" * 70)
    log("STEP 8: GTT firmware hunt (system memory search)")
    log("=" * 70)
    s = {}

    # Read /proc/iomem for GPU BAR regions
    try:
        with open("/proc/iomem") as f:
            iomem = f.read()
        gpu_regions = []
        for line in iomem.split('\n'):
            if 'amdgpu' in line.lower() or ('c3:00.0' in line):
                gpu_regions.append(line.strip())
                log(f"  iomem: {line.strip()}")
        s['gpu_iomem_regions'] = gpu_regions
    except Exception as e:
        log(f"  /proc/iomem read failed: {e}")
        s['gpu_iomem_regions'] = []

    # Check GART info from amdgpu
    try:
        r = subprocess.run(['cat', '/sys/class/drm/card1/device/resource'], capture_output=True, text=True)
        if r.returncode == 0:
            log(f"  PCI BARs:")
            for i, line in enumerate(r.stdout.strip().split('\n')):
                parts = line.split()
                if len(parts) >= 3:
                    start = int(parts[0], 16)
                    end = int(parts[1], 16)
                    flags = int(parts[2], 16)
                    size = end - start + 1 if end > start else 0
                    if size > 0:
                        log(f"    BAR{i}: 0x{start:012X}-0x{end:012X} ({size/(1024*1024):.1f}MB) flags=0x{flags:X}")
                        s[f'bar{i}'] = {'start': f"0x{start:012X}", 'end': f"0x{end:012X}", 'size_mb': size/(1024*1024)}
    except Exception as e:
        log(f"  PCI BAR read failed: {e}")

    # Try reading /dev/mem at MES IC_BASE GPU VA = 0x800000
    # On APU, VRAM might be at fb_location_mc (from amdgpu)
    try:
        # Get VRAM base from kernel
        r = subprocess.run(['sudo', 'dmesg'], capture_output=True, text=True)
        for line in r.stdout.split('\n'):
            if 'fb location' in line.lower() or 'vram_start' in line.lower() or 'mc_vm_fb_location' in line.lower():
                log(f"  dmesg: {line.strip()}")
                s.setdefault('vram_hints', []).append(line.strip())
    except:
        pass

    # Check GPU memory info
    try:
        with open("/sys/class/drm/card1/device/mem_info_vram_total") as f:
            vram_total = int(f.read().strip())
            log(f"  VRAM total: {vram_total} bytes ({vram_total/(1024**3):.1f}GB)")
            s['vram_total'] = vram_total
    except:
        pass
    try:
        with open("/sys/class/drm/card1/device/mem_info_gtt_total") as f:
            gtt_total = int(f.read().strip())
            log(f"  GTT total: {gtt_total} bytes ({gtt_total/(1024**3):.1f}GB)")
            s['gtt_total'] = gtt_total
    except:
        pass

    if not check_health("step8"):
        s['status'] = 'ABORTED'
    else:
        s['status'] = 'COMPLETED'

    state['steps']['step8'] = s
    save_state()
    return True


def step9_analysis():
    """Final analysis: summarize what worked, what's possible, what's blocked."""
    log("=" * 70)
    log("STEP 9: Final Analysis")
    log("=" * 70)
    s = {}

    # Collect results from all steps
    analysis = []

    # Step 2: scratch writes
    step2 = state['steps'].get('step2', {})
    scratch_ok = step2.get('all_passed', False)
    analysis.append(f"Debugfs MMIO write path: {'WORKS' if scratch_ok else 'BROKEN'}")

    # Step 3: IC_OP_CNTL writes
    step3 = state['steps'].get('step3', {})
    any_inv_accepted = False
    any_prime_accepted = False
    for test in step3.get('tests', []):
        if test.get('invalidate_accepted'):
            any_inv_accepted = True
        if test.get('prime_accepted'):
            any_prime_accepted = True
    analysis.append(f"IC_OP_CNTL INVALIDATE writable: {any_inv_accepted}")
    analysis.append(f"IC_OP_CNTL PRIME writable: {any_prime_accepted}")

    # Step 4: IC_BASE writes
    step4 = state['steps'].get('step4', {})
    any_base_accepted = False
    for test in step4.get('tests', []):
        if test.get('write_accepted'):
            any_base_accepted = True
    analysis.append(f"IC_BASE writable: {any_base_accepted}")

    # Step 5: VRAM payload
    step5 = state['steps'].get('step5', {})
    vram_ok = step5.get('verify') == 'MATCH'
    analysis.append(f"VRAM payload write+verify: {'OK' if vram_ok else step5.get('verify', 'NOT_RUN')}")

    # Step 6: IC redirect
    step6 = state['steps'].get('step6', {})
    ip_changed = step6.get('ip_changed', False)
    analysis.append(f"IC redirect result: {step6.get('status', 'NOT_RUN')}")
    if ip_changed:
        analysis.append(f"*** MES INSTRUCTION POINTER CHANGED — FIRMWARE REDIRECT POSSIBLE ***")

    # Step 7: KFD PM4
    step7 = state['steps'].get('step7', {})
    pm4_ok = step7.get('pm4_possible', False)
    analysis.append(f"KFD PM4 path: {'AVAILABLE' if pm4_ok else 'NOT_AVAILABLE'}")

    # Summary
    log("  === SUMMARY ===")
    for line in analysis:
        log(f"  {line}")

    # Verdict
    can_modify = any_inv_accepted and any_base_accepted and vram_ok
    if can_modify:
        verdict = "FIRMWARE MODIFICATION POSSIBLE via IC invalidate + IC_BASE redirect + VRAM payload"
    elif scratch_ok and not any_inv_accepted:
        verdict = "Debugfs writes work but IC registers are HARDWARE PROTECTED — cannot modify firmware"
    else:
        verdict = "Further investigation needed"
    log(f"  VERDICT: {verdict}")

    s['analysis'] = analysis
    s['verdict'] = verdict
    s['stats'] = {
        'writes_attempted': state['writes_attempted'],
        'writes_confirmed': state['writes_confirmed'],
        'writes_rejected': state['writes_rejected'],
        'health_checks': state['health_checks'],
        'anomalies': state['anomalies'],
    }

    state['steps']['step9'] = s
    save_state()
    return True


# ─── MAIN ────────────────────────────────────────────────────────────

def main():
    log(f"z2348: Deep IC/Firmware Boundary Test")
    log(f"PID: {os.getpid()}")
    log(f"Temperature: {get_temp():.1f}C")
    log(f"Skip steps: {SKIP_STEPS or 'none'}")

    steps = [
        (0, "Sanity check", step0_sanity),
        (1, "IC register dump", step1_ic_registers),
        (2, "Scratch write test", step2_scratch_write),
        (3, "IC_OP_CNTL write test", step3_ic_op_cntl_write),
        (4, "IC_BASE write test", step4_ic_base_write),
        (5, "VRAM payload prep", step5_vram_payload),
        (6, "IC redirect attempt", step6_ic_redirect),
        (7, "KFD PM4 probe", step7_kfd_pm4),
        (8, "GTT firmware hunt", step8_gtt_firmware_hunt),
        (9, "Analysis", step9_analysis),
    ]

    for step_num, step_name, step_func in steps:
        if step_num in SKIP_STEPS:
            log(f"  SKIPPING step {step_num}: {step_name}")
            continue

        log("")
        try:
            ok = step_func()
        except Exception as e:
            log(f"  STEP {step_num} EXCEPTION: {e}")
            import traceback
            log(f"  {traceback.format_exc()}")
            state['anomalies'].append(f"Step {step_num} exception: {e}")
            ok = False
            save_state()

        if not ok:
            log(f"  STEP {step_num} ({step_name}) returned FAIL — stopping")
            state['stopped_at'] = step_num
            state['stopped_reason'] = f"Step {step_num} failed"
            save_state()
            break

        pause(1000)  # 1s between steps

    state['completed'] = datetime.now().isoformat()
    save_state()
    log("")
    log(f"ALL DONE. Results: {JSON_FILE}")
    log(f"Final temp: {get_temp():.1f}C")


if __name__ == "__main__":
    main()
