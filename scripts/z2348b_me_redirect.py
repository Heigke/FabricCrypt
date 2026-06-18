#!/usr/bin/env python3
"""
z2348b: ME (Micro Engine) IC Redirect Test
===========================================
z2348 found that CP_ME_IC_OP_CNTL accepts INVALIDATE and PRIME writes.
This script tests if we can also write ME IC_BASE and redirect firmware.

Steps:
  0. Sanity + health
  1. Read ALL ME registers thoroughly
  2. Test ME IC_BASE_LO write (z2348 only tested MES, not ME)
  3. Test ME IC_BASE_HI write
  4. Test ME IC_BASE_CNTL write
  5. If ME IC_BASE is writable: full redirect sequence
     a. Write NOP-sled to VRAM
     b. Change ME IC_BASE → VRAM NOP-sled
     c. INVALIDATE ME IC
     d. PRIME ME IC
     e. Read ME state (IP, status)
     f. Restore original IC_BASE + re-prime
  6. Also test: PFP IC_BASE writes (PFP_IC_BASE_CNTL had interesting bits)
  7. Also test: Write to all 4 IC_BASE_CNTL registers
  8. KFD PM4 WRITE_DATA to scratch register (prove PM4 write path works)
  9. Analysis

SAFETY: Same as z2348 — health check after every write, no SRBM.
"""
import struct, os, sys, json, time, subprocess, ctypes
from datetime import datetime
from pathlib import Path

DEBUGFS_REGS = "/sys/kernel/debug/dri/128/amdgpu_regs"
DEBUGFS_VRAM = "/sys/kernel/debug/dri/128/amdgpu_vram"
THERMAL      = "/sys/class/thermal/thermal_zone0/temp"
RESULTS_DIR  = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
LOG_FILE     = RESULTS_DIR / "z2348b_log.txt"
JSON_FILE    = RESULTS_DIR / "z2348b_me_redirect.json"

PAYLOAD_VRAM_OFFSET = 64 * 1024 * 1024  # 64MB

# ME IC registers
REG = {
    'GRBM_STATUS':           0x2004,
    'GRBM_STATUS2':          0x2002,
    # ME
    'CP_ME_IC_BASE_LO':     0x5844,
    'CP_ME_IC_BASE_HI':     0x5845,
    'CP_ME_IC_BASE_CNTL':   0x5846,
    'CP_ME_IC_OP_CNTL':     0x5847,
    # PFP
    'CP_PFP_IC_BASE_LO':    0x5840,
    'CP_PFP_IC_BASE_HI':    0x5841,
    'CP_PFP_IC_BASE_CNTL':  0x5842,
    'CP_PFP_IC_OP_CNTL':    0x5843,
    # CPC
    'CP_CPC_IC_BASE_LO':    0x584C,
    'CP_CPC_IC_BASE_HI':    0x584D,
    'CP_CPC_IC_BASE_CNTL':  0x584E,
    'CP_CPC_IC_OP_CNTL':    0x297A,
    # MES
    'CP_MES_IC_BASE_LO':    0x5850,
    'CP_MES_IC_BASE_HI':    0x5851,
    'CP_MES_IC_BASE_CNTL':  0x5852,
    'CP_MES_IC_OP_CNTL':    0x2820,
    # Status
    'CP_STAT':              0x2100,
    'CP_BUSY_STAT':         0x2108,
    'CP_GFX_ERROR':         0x210C,
    'SCRATCH_0':            0x2040,
    'SCRATCH_1':            0x2041,
    # ME specific
    'CP_ME_CNTL':           0x2963,
    'CP_ME_INSTR_PNTR':     0x2966,
    'CP_GFX_RS64_INSTR_PNTR0': 0x2900,
    'CP_GFX_RS64_INSTR_PNTR1': 0x2901,
}

INVALIDATE_BIT = 1 << 0
INV_COMPLETE   = 1 << 1
PRIME_BIT      = 1 << 2
ICACHE_PRIMED  = 1 << 3

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

state = {
    'started': datetime.now().isoformat(),
    'pid': os.getpid(),
    'steps': {},
    'anomalies': [],
    'writes_attempted': 0,
    'writes_confirmed': 0,
    'writes_rejected': 0,
    'writes_silent_fail': 0,  # write accepted but readback != written
    'health_checks': 0,
    'writable_registers': [],
    'non_writable_registers': [],
}

def save_state():
    state['last_save'] = datetime.now().isoformat()
    state['temperature_C'] = get_temp()
    with open(JSON_FILE, 'w') as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

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
    t = get_temp()
    state['health_checks'] += 1
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(REG['GRBM_STATUS'] * 4)
            data = f.read(4)
            grbm = struct.unpack("<I", data)[0] if len(data) == 4 else 0xDEAD
    except Exception as e:
        log(f"  HEALTH {label}: GRBM FAIL ({e})")
        state['anomalies'].append(f"Health fail at {label}: {e}")
        return False
    log(f"  HEALTH {label}: GRBM=0x{grbm:08X} T={t:.1f}C")
    if grbm == 0xFFFFFFFF:
        state['anomalies'].append(f"GPU hung at {label}")
        return False
    if t > 85.0:
        state['anomalies'].append(f"Thermal {t}C at {label}")
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
        log(f"  REG {name} (0x{dword_offset:04X}): {e}")
        return None

def write_reg(dword_offset, value, name="?"):
    byte_off = dword_offset * 4
    state['writes_attempted'] += 1
    log(f"  WRITE {name} (0x{dword_offset:04X}) <- 0x{value:08X}")
    try:
        with open(DEBUGFS_REGS, "r+b") as f:
            f.seek(byte_off)
            f.write(struct.pack("<I", value))
            f.flush()
    except Exception as e:
        log(f"  WRITE {name}: EXCEPTION {e}")
        state['anomalies'].append(f"Write fail {name}: {e}")
        return None
    time.sleep(0.05)
    rb = read_reg(dword_offset, f"{name}_rb")
    if rb is not None:
        if rb == value:
            log(f"  WRITE {name}: ✓ CONFIRMED")
            state['writes_confirmed'] += 1
            state['writable_registers'].append(f"{name}(0x{dword_offset:04X})")
        else:
            log(f"  WRITE {name}: ✗ REJECTED (wrote 0x{value:08X}, got 0x{rb:08X})")
            state['writes_rejected'] += 1
            if f"{name}(0x{dword_offset:04X})" not in state['non_writable_registers']:
                state['non_writable_registers'].append(f"{name}(0x{dword_offset:04X})")
    return rb

def write_reg_test(dword_offset, test_value, name="?"):
    """Write test_value, check if accepted, restore original. Returns (accepted, readback)."""
    original = read_reg(dword_offset, f"{name}_original")
    time.sleep(0.05)
    rb = write_reg(dword_offset, test_value, f"{name}_test")
    accepted = (rb == test_value)
    time.sleep(0.05)
    # Restore
    write_reg(dword_offset, original if original is not None else 0, f"{name}_restore")
    time.sleep(0.05)
    return accepted, rb, original

def read_vram(offset, size):
    try:
        with open(DEBUGFS_VRAM, "rb") as f:
            f.seek(offset)
            return f.read(size)
    except Exception as e:
        log(f"  VRAM read 0x{offset:X}: {e}")
        return None

def write_vram(offset, data):
    try:
        with open(DEBUGFS_VRAM, "r+b") as f:
            f.seek(offset)
            f.write(data)
            f.flush()
        return True
    except Exception as e:
        log(f"  VRAM write 0x{offset:X}: {e}")
        return False


# ─── STEPS ───────────────────────────────────────────────────────────

def step0():
    log("=" * 70)
    log("STEP 0: Sanity")
    log("=" * 70)
    if not check_health("step0"):
        return False
    state['steps']['step0'] = {'status': 'OK'}
    save_state()
    return True

def step1():
    """Read ALL ME-related registers."""
    log("=" * 70)
    log("STEP 1: ME register full dump")
    log("=" * 70)
    s = {}

    for name in ['CP_ME_IC_BASE_LO', 'CP_ME_IC_BASE_HI', 'CP_ME_IC_BASE_CNTL',
                  'CP_ME_IC_OP_CNTL', 'CP_ME_CNTL',
                  'CP_GFX_RS64_INSTR_PNTR0', 'CP_GFX_RS64_INSTR_PNTR1']:
        val = read_reg(REG[name], name)
        s[name] = f"0x{val:08X}" if val is not None else "FAIL"
        time.sleep(0.05)

    # Decode ME IC_BASE_CNTL
    cntl = int(s.get('CP_ME_IC_BASE_CNTL', '0x0'), 16)
    log(f"  ME IC_BASE_CNTL decode: 0x{cntl:08X}")
    log(f"    bits[12:0] = 0x{cntl & 0x1FFF:04X}")
    log(f"    bit 13 = {(cntl >> 13) & 1}")
    log(f"    bit 14 = {(cntl >> 14) & 1}")
    log(f"    bit 17 = {(cntl >> 17) & 1}")
    log(f"    bit 29 = {(cntl >> 29) & 1}")
    s['cntl_decode'] = {
        'vmid_or_cache': f"0x{cntl & 0x1FFF:04X}",
        'bit13': (cntl >> 13) & 1,
        'bit14': (cntl >> 14) & 1,
        'bit17': (cntl >> 17) & 1,
        'bit29': (cntl >> 29) & 1,
    }

    if not check_health("step1"):
        return False

    state['steps']['step1'] = s
    save_state()
    return True

def step2():
    """Test ME IC_BASE_LO write."""
    log("=" * 70)
    log("STEP 2: ME IC_BASE_LO write test")
    log("=" * 70)
    s = {}

    accepted, rb, orig = write_reg_test(REG['CP_ME_IC_BASE_LO'], 0x0000DEAD, "ME_IC_BASE_LO")
    s['test_value'] = "0x0000DEAD"
    s['accepted'] = accepted
    s['readback'] = f"0x{rb:08X}" if rb is not None else "FAIL"
    s['original'] = f"0x{orig:08X}" if orig is not None else "FAIL"

    if not check_health("step2"):
        state['steps']['step2'] = s
        save_state()
        return False

    # Also try 0xFFFFFFFF
    accepted2, rb2, _ = write_reg_test(REG['CP_ME_IC_BASE_LO'], 0xFFFFFFFF, "ME_IC_BASE_LO_ff")
    s['test2_accepted'] = accepted2
    s['test2_readback'] = f"0x{rb2:08X}" if rb2 is not None else "FAIL"

    if not check_health("step2b"):
        state['steps']['step2'] = s
        save_state()
        return False

    state['steps']['step2'] = s
    save_state()
    return True

def step3():
    """Test ME IC_BASE_HI write."""
    log("=" * 70)
    log("STEP 3: ME IC_BASE_HI write test")
    log("=" * 70)
    s = {}

    accepted, rb, orig = write_reg_test(REG['CP_ME_IC_BASE_HI'], 0x00000001, "ME_IC_BASE_HI")
    s['accepted'] = accepted
    s['readback'] = f"0x{rb:08X}" if rb is not None else "FAIL"
    s['original'] = f"0x{orig:08X}" if orig is not None else "FAIL"

    if not check_health("step3"):
        state['steps']['step3'] = s
        save_state()
        return False

    state['steps']['step3'] = s
    save_state()
    return True

def step4():
    """Test ME IC_BASE_CNTL write."""
    log("=" * 70)
    log("STEP 4: ME IC_BASE_CNTL write test")
    log("=" * 70)
    s = {}

    accepted, rb, orig = write_reg_test(REG['CP_ME_IC_BASE_CNTL'], 0x00000000, "ME_IC_BASE_CNTL_zero")
    s['zero_accepted'] = accepted
    s['zero_readback'] = f"0x{rb:08X}" if rb is not None else "FAIL"
    s['original'] = f"0x{orig:08X}" if orig is not None else "FAIL"

    if not check_health("step4a"):
        state['steps']['step4'] = s
        save_state()
        return False

    accepted2, rb2, _ = write_reg_test(REG['CP_ME_IC_BASE_CNTL'], 0xFFFFFFFF, "ME_IC_BASE_CNTL_ff")
    s['ff_accepted'] = accepted2
    s['ff_readback'] = f"0x{rb2:08X}" if rb2 is not None else "FAIL"

    if not check_health("step4b"):
        state['steps']['step4'] = s
        save_state()
        return False

    state['steps']['step4'] = s
    save_state()
    return True

def step5():
    """Test ALL IC_BASE/CNTL registers across all 4 engines."""
    log("=" * 70)
    log("STEP 5: Comprehensive IC register write sweep")
    log("=" * 70)
    s = {'results': []}

    targets = [
        ('PFP_IC_BASE_LO',   REG['CP_PFP_IC_BASE_LO'],   0x0000CAFE),
        ('PFP_IC_BASE_HI',   REG['CP_PFP_IC_BASE_HI'],   0x00000001),
        ('PFP_IC_BASE_CNTL', REG['CP_PFP_IC_BASE_CNTL'], 0x00000000),
        ('PFP_IC_OP_CNTL',   REG['CP_PFP_IC_OP_CNTL'],   INVALIDATE_BIT),
        ('ME_IC_BASE_LO',    REG['CP_ME_IC_BASE_LO'],    0x0000CAFE),
        ('ME_IC_BASE_HI',    REG['CP_ME_IC_BASE_HI'],    0x00000001),
        ('ME_IC_BASE_CNTL',  REG['CP_ME_IC_BASE_CNTL'],  0x00000000),
        ('ME_IC_OP_CNTL',    REG['CP_ME_IC_OP_CNTL'],    INVALIDATE_BIT),
        ('CPC_IC_BASE_LO',   REG['CP_CPC_IC_BASE_LO'],   0x0000CAFE),
        ('CPC_IC_BASE_HI',   REG['CP_CPC_IC_BASE_HI'],   0x00000001),
        ('CPC_IC_BASE_CNTL', REG['CP_CPC_IC_BASE_CNTL'], 0x00000000),
        ('CPC_IC_OP_CNTL',   REG['CP_CPC_IC_OP_CNTL'],   INVALIDATE_BIT),
        ('MES_IC_BASE_LO',   REG['CP_MES_IC_BASE_LO'],   0x0000CAFE),
        ('MES_IC_BASE_HI',   REG['CP_MES_IC_BASE_HI'],   0x00000001),
        ('MES_IC_BASE_CNTL', REG['CP_MES_IC_BASE_CNTL'], 0x00000001),
        ('MES_IC_OP_CNTL',   REG['CP_MES_IC_OP_CNTL'],   INVALIDATE_BIT),
    ]

    for name, addr, test_val in targets:
        log(f"  --- {name} (0x{addr:04X}) <- 0x{test_val:08X} ---")
        accepted, rb, orig = write_reg_test(addr, test_val, name)
        result = {
            'register': name,
            'address': f"0x{addr:04X}",
            'test_value': f"0x{test_val:08X}",
            'original': f"0x{orig:08X}" if orig is not None else "FAIL",
            'readback': f"0x{rb:08X}" if rb is not None else "FAIL",
            'accepted': accepted,
        }
        s['results'].append(result)
        if accepted:
            log(f"  >>> {name}: WRITABLE <<<")
        time.sleep(0.1)

        if not check_health(f"step5_{name}"):
            state['steps']['step5'] = s
            save_state()
            return False

    # Summary
    writable = [r['register'] for r in s['results'] if r['accepted']]
    non_writable = [r['register'] for r in s['results'] if not r['accepted']]
    log(f"  WRITABLE ({len(writable)}): {writable}")
    log(f"  NON-WRITABLE ({len(non_writable)}): {non_writable}")
    s['writable'] = writable
    s['non_writable'] = non_writable

    state['steps']['step5'] = s
    save_state()
    return True

def step6():
    """If ME has writable IC_BASE + IC_OP_CNTL: attempt full redirect."""
    log("=" * 70)
    log("STEP 6: ME IC redirect attempt")
    log("=" * 70)
    s = {}

    step5_data = state['steps'].get('step5', {})
    writable = step5_data.get('writable', [])

    me_base_ok = 'ME_IC_BASE_LO' in writable or 'ME_IC_BASE_HI' in writable
    me_op_ok = 'ME_IC_OP_CNTL' in writable

    log(f"  ME IC_BASE writable: {me_base_ok}")
    log(f"  ME IC_OP_CNTL writable: {me_op_ok}")

    if not (me_base_ok and me_op_ok):
        # Check if ANY engine has both
        for engine in ['PFP', 'ME', 'CPC', 'MES']:
            base_ok = f'{engine}_IC_BASE_LO' in writable or f'{engine}_IC_BASE_HI' in writable
            op_ok = f'{engine}_IC_OP_CNTL' in writable
            if base_ok and op_ok:
                log(f"  {engine} has BOTH IC_BASE and IC_OP_CNTL writable!")
                s['redirect_engine'] = engine
                # TODO: implement redirect for this engine
                break
        else:
            log(f"  No engine has both IC_BASE and IC_OP_CNTL writable")
            log(f"  Redirect NOT possible via debugfs MMIO alone")
            s['status'] = 'NO_COMBINED_WRITE_PATH'
            state['steps']['step6'] = s
            save_state()
            return True

    if not check_health("step6_pre"):
        s['status'] = 'ABORTED'
        state['steps']['step6'] = s
        save_state()
        return False

    # Prepare VRAM payload
    NOP = struct.pack("<I", 0xBF800000)     # s_nop 0
    ENDPGM = struct.pack("<I", 0xBF810000)  # s_endpgm
    payload = NOP * 252 + ENDPGM * 4
    log(f"  Writing NOP-sled to VRAM 0x{PAYLOAD_VRAM_OFFSET:X}...")
    write_vram(PAYLOAD_VRAM_OFFSET, payload)
    verify = read_vram(PAYLOAD_VRAM_OFFSET, len(payload))
    if verify != payload:
        log(f"  VRAM payload verify FAILED!")
        s['status'] = 'VRAM_FAIL'
        state['steps']['step6'] = s
        save_state()
        return True

    # Calculate IC_BASE for VRAM offset
    # IC_BASE stores gpu_addr >> 8
    # VRAM offset 64MB = 0x4000000, shifted >> 8 = 0x40000
    payload_addr_shifted = PAYLOAD_VRAM_OFFSET >> 8
    new_lo = payload_addr_shifted & 0xFFFFFFFF
    new_hi = (payload_addr_shifted >> 32) & 0xFFFFFFFF
    log(f"  Payload IC_BASE: LO=0x{new_lo:08X} HI=0x{new_hi:08X}")

    # Save original ME state
    orig_lo = read_reg(REG['CP_ME_IC_BASE_LO'], "ME_orig_LO")
    orig_hi = read_reg(REG['CP_ME_IC_BASE_HI'], "ME_orig_HI")
    orig_cntl = read_reg(REG['CP_ME_IC_BASE_CNTL'], "ME_orig_CNTL")
    s['original'] = {
        'lo': f"0x{orig_lo:08X}" if orig_lo is not None else "FAIL",
        'hi': f"0x{orig_hi:08X}" if orig_hi is not None else "FAIL",
        'cntl': f"0x{orig_cntl:08X}" if orig_cntl is not None else "FAIL",
    }

    # 6a: Set new IC_BASE
    log(f"  *** 6a: ME IC_BASE <- LO=0x{new_lo:08X} HI=0x{new_hi:08X}")
    rb_lo = write_reg(REG['CP_ME_IC_BASE_LO'], new_lo, "ME_redirect_LO")
    rb_hi = write_reg(REG['CP_ME_IC_BASE_HI'], new_hi, "ME_redirect_HI")
    s['redirect_lo'] = f"0x{rb_lo:08X}" if rb_lo is not None else "FAIL"
    s['redirect_hi'] = f"0x{rb_hi:08X}" if rb_hi is not None else "FAIL"
    s['lo_accepted'] = (rb_lo == new_lo)
    s['hi_accepted'] = (rb_hi == new_hi)
    time.sleep(0.2)

    if not check_health("step6_base"):
        log(f"  Restoring...")
        write_reg(REG['CP_ME_IC_BASE_LO'], orig_lo or 0, "restore_LO")
        write_reg(REG['CP_ME_IC_BASE_HI'], orig_hi or 0, "restore_HI")
        s['status'] = 'ABORTED_AFTER_BASE'
        state['steps']['step6'] = s
        save_state()
        return False

    # 6b: INVALIDATE ME IC
    log(f"  *** 6b: ME IC_OP_CNTL <- INVALIDATE (0x01)")
    rb_inv = write_reg(REG['CP_ME_IC_OP_CNTL'], INVALIDATE_BIT, "ME_INVALIDATE")
    s['invalidate_rb'] = f"0x{rb_inv:08X}" if rb_inv is not None else "FAIL"
    time.sleep(0.2)

    if not check_health("step6_inv"):
        log(f"  Restoring...")
        write_reg(REG['CP_ME_IC_BASE_LO'], orig_lo or 0, "restore_LO")
        write_reg(REG['CP_ME_IC_BASE_HI'], orig_hi or 0, "restore_HI")
        s['status'] = 'ABORTED_AFTER_INV'
        state['steps']['step6'] = s
        save_state()
        return False

    # 6c: PRIME ME IC from new base
    log(f"  *** 6c: ME IC_OP_CNTL <- PRIME (0x04)")
    rb_prime = write_reg(REG['CP_ME_IC_OP_CNTL'], PRIME_BIT, "ME_PRIME")
    s['prime_rb'] = f"0x{rb_prime:08X}" if rb_prime is not None else "FAIL"
    time.sleep(0.5)

    if not check_health("step6_prime"):
        log(f"  Restoring...")
        write_reg(REG['CP_ME_IC_BASE_LO'], orig_lo or 0, "restore_LO")
        write_reg(REG['CP_ME_IC_BASE_HI'], orig_hi or 0, "restore_HI")
        write_reg(REG['CP_ME_IC_OP_CNTL'], INVALIDATE_BIT, "restore_inv")
        time.sleep(0.1)
        write_reg(REG['CP_ME_IC_OP_CNTL'], PRIME_BIT, "restore_prime")
        s['status'] = 'ABORTED_AFTER_PRIME'
        state['steps']['step6'] = s
        save_state()
        return False

    # 6d: Check ME state
    log(f"  --- ME state after redirect ---")
    for name in ['CP_GFX_RS64_INSTR_PNTR0', 'CP_GFX_RS64_INSTR_PNTR1',
                  'CP_STAT', 'CP_BUSY_STAT', 'CP_GFX_ERROR', 'GRBM_STATUS']:
        read_reg(REG[name], name)
        time.sleep(0.05)

    # Read IC_OP_CNTL to see if PRIMED bit appeared
    op_after = read_reg(REG['CP_ME_IC_OP_CNTL'], "ME_OP_CNTL_after")
    s['op_cntl_after'] = f"0x{op_after:08X}" if op_after is not None else "FAIL"
    if op_after is not None:
        log(f"  IC_OP_CNTL: INV={(op_after>>0)&1} DONE={(op_after>>1)&1} PRIME={(op_after>>2)&1} PRIMED={(op_after>>3)&1}")

    # 6e: Restore
    log(f"  *** Restoring ME IC_BASE to original ***")
    write_reg(REG['CP_ME_IC_BASE_LO'], orig_lo or 0, "restore_LO")
    write_reg(REG['CP_ME_IC_BASE_HI'], orig_hi or 0, "restore_HI")
    write_reg(REG['CP_ME_IC_BASE_CNTL'], orig_cntl or 0x531F, "restore_CNTL")
    time.sleep(0.1)
    write_reg(REG['CP_ME_IC_OP_CNTL'], INVALIDATE_BIT, "restore_inv")
    time.sleep(0.1)
    write_reg(REG['CP_ME_IC_OP_CNTL'], PRIME_BIT, "restore_prime")
    time.sleep(0.2)

    if not check_health("step6_restore"):
        s['status'] = 'RESTORE_FAILED'
    else:
        s['status'] = 'COMPLETED'

    state['steps']['step6'] = s
    save_state()
    return True


def step7():
    """Test IC_BASE_CNTL writes for all engines."""
    log("=" * 70)
    log("STEP 7: IC_BASE_CNTL write sweep")
    log("=" * 70)
    s = {'results': []}

    cntl_regs = [
        ('PFP_IC_BASE_CNTL', REG['CP_PFP_IC_BASE_CNTL']),
        ('ME_IC_BASE_CNTL',  REG['CP_ME_IC_BASE_CNTL']),
        ('CPC_IC_BASE_CNTL', REG['CP_CPC_IC_BASE_CNTL']),
        ('MES_IC_BASE_CNTL', REG['CP_MES_IC_BASE_CNTL']),
    ]

    for name, addr in cntl_regs:
        orig = read_reg(addr, f"{name}_orig")
        results = []

        # Test bit-by-bit: which bits are writable?
        for bit in range(32):
            test_val = (orig or 0) ^ (1 << bit)  # flip one bit
            rb = write_reg(addr, test_val, f"{name}_bit{bit}")
            accepted = (rb == test_val)
            if accepted:
                results.append(bit)
            # Restore
            write_reg(addr, orig or 0, f"{name}_restore")
            time.sleep(0.02)

        log(f"  {name}: writable bits = {results}")
        s['results'].append({
            'register': name,
            'original': f"0x{orig:08X}" if orig is not None else "FAIL",
            'writable_bits': results,
        })

        if not check_health(f"step7_{name}"):
            state['steps']['step7'] = s
            save_state()
            return False

    state['steps']['step7'] = s
    save_state()
    return True


def step8():
    """Try PM4 WRITE_DATA to scratch register via KFD to prove PM4 works."""
    log("=" * 70)
    log("STEP 8: KFD PM4 WRITE_DATA probe")
    log("=" * 70)
    s = {}

    # This requires a more complex setup using hsakmt
    # For now, test if we can at least allocate a queue
    try:
        hsakmt = ctypes.CDLL("libhsakmt.so.1")
        ret = hsakmt.hsaKmtOpenKFD()
        log(f"  hsaKmtOpenKFD() = {ret}")
        if ret != 0:
            s['status'] = 'KFD_OPEN_FAIL'
            state['steps']['step8'] = s
            save_state()
            return True

        # Try to get system properties
        # hsaKmtGetSystemProperties(HsaSystemProperties*)
        class HsaSystemProperties(ctypes.Structure):
            _fields_ = [
                ("NumNodes", ctypes.c_uint32),
                ("PlatformOem", ctypes.c_uint32),
                ("PlatformId", ctypes.c_uint32),
                ("PlatformRev", ctypes.c_uint32),
            ]
        props = HsaSystemProperties()
        ret = hsakmt.hsaKmtGetSystemProperties(ctypes.byref(props))
        log(f"  hsaKmtGetSystemProperties() = {ret}")
        if ret == 0:
            log(f"  NumNodes = {props.NumNodes}")
            s['num_nodes'] = props.NumNodes

        # For a real PM4 submission we'd need to:
        # 1. hsaKmtAcquireSystemProperties
        # 2. hsaKmtCreateQueue (with PM4 ring buffer)
        # 3. Write PM4 packets to ring buffer
        # 4. Ring doorbell
        # This is complex but possible — for now confirm KFD works
        log(f"  KFD confirmed working — PM4 submission requires queue setup")
        s['kfd_works'] = True

        hsakmt.hsaKmtCloseKFD()
        s['status'] = 'KFD_CONFIRMED'

    except Exception as e:
        log(f"  KFD probe exception: {e}")
        s['status'] = f'EXCEPTION: {e}'

    state['steps']['step8'] = s
    save_state()
    return True


def step9():
    """Final analysis."""
    log("=" * 70)
    log("STEP 9: Final Analysis")
    log("=" * 70)
    s = {}

    findings = []

    # Collect writable registers from step 5
    step5 = state['steps'].get('step5', {})
    writable = step5.get('writable', [])
    non_writable = step5.get('non_writable', [])

    findings.append(f"WRITABLE IC registers via debugfs: {writable}")
    findings.append(f"NON-WRITABLE IC registers: {non_writable}")

    # Check per-engine capability
    for engine in ['PFP', 'ME', 'CPC', 'MES']:
        base_w = any(engine in r for r in writable if 'BASE' in r and 'CNTL' not in r)
        cntl_w = any(engine in r for r in writable if 'CNTL' in r and 'OP' not in r)
        op_w = any(engine in r for r in writable if 'OP_CNTL' in r)
        findings.append(f"  {engine}: BASE={'W' if base_w else '-'} CNTL={'W' if cntl_w else '-'} OP={'W' if op_w else '-'}")
        if base_w and op_w:
            findings.append(f"  *** {engine}: FULL REDIRECT POSSIBLE (base + op_cntl writable) ***")

    # Step 6 results
    step6 = state['steps'].get('step6', {})
    findings.append(f"Redirect test: {step6.get('status', 'NOT_RUN')}")

    # Step 7 results
    step7 = state['steps'].get('step7', {})
    for r in step7.get('results', []):
        bits = r.get('writable_bits', [])
        if bits:
            findings.append(f"  {r['register']}: {len(bits)} writable bits: {bits}")

    # Step 8
    step8 = state['steps'].get('step8', {})
    findings.append(f"KFD PM4: {step8.get('status', 'NOT_RUN')}")

    # Verdict
    any_full_redirect = False
    for engine in ['PFP', 'ME', 'CPC', 'MES']:
        base_w = any(engine in r for r in writable if 'BASE' in r and 'CNTL' not in r)
        op_w = any(engine in r for r in writable if 'OP_CNTL' in r)
        if base_w and op_w:
            any_full_redirect = True

    if any_full_redirect:
        verdict = "FIRMWARE REDIRECT ACHIEVABLE — at least one engine has writable IC_BASE + IC_OP_CNTL"
    elif writable:
        verdict = f"PARTIAL WRITE ACCESS — {len(writable)} registers writable but no complete redirect chain"
    else:
        verdict = "NO DEBUGFS WRITE ACCESS — all IC registers are hardware-protected"

    findings.append(f"VERDICT: {verdict}")

    for line in findings:
        log(f"  {line}")

    s['findings'] = findings
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


def main():
    log(f"z2348b: ME IC Redirect Test")
    log(f"PID: {os.getpid()}")
    log(f"Temperature: {get_temp():.1f}C")

    # Clear old log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    steps = [
        (0, step0), (1, step1), (2, step2), (3, step3), (4, step4),
        (5, step5), (6, step6), (7, step7), (8, step8), (9, step9),
    ]

    for num, func in steps:
        log("")
        try:
            ok = func()
        except Exception as e:
            import traceback
            log(f"  STEP {num} EXCEPTION: {e}")
            log(traceback.format_exc())
            state['anomalies'].append(f"Step {num}: {e}")
            save_state()
            ok = False

        if not ok:
            log(f"  STEP {num} FAILED — stopping")
            state['stopped_at'] = num
            save_state()
            break
        time.sleep(1)

    state['completed'] = datetime.now().isoformat()
    save_state()
    log(f"\nALL DONE. Results: {JSON_FILE}")
    log(f"Final temp: {get_temp():.1f}C")


if __name__ == "__main__":
    main()
