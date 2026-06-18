#!/usr/bin/env python3
"""z2351: PM4 WRITE_DATA with dst_sel=register to IC registers.

debugfs MMIO writes to IC registers are shadow/decoupled (z2350 finding).
PM4 WRITE_DATA with dst_sel=0 (register) goes through ME's internal
register write path — may have different privilege/routing.

Also tests COPY_DATA (opcode 0x40) and SET_UCONFIG_REG packets.

Three approaches:
1. WRITE_DATA dst_sel=0 (register) — ME writes to register file directly
2. COPY_DATA src=immediate dst=register — another PM4 register write path
3. WRITE_DATA dst_sel=0 to RLC registers — check RLC accessibility

Must run as root with LD_PRELOAD hook for PM4 queue.
"""
import struct, time, os, json, ctypes

REGS = '/sys/kernel/debug/dri/0/amdgpu_regs'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

hsa = ctypes.CDLL('/opt/rocm-7.1.1/lib/libhsa-runtime64.so.1', use_errno=True)

class hsa_agent_t(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint64)]
class hsa_queue_t(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint32), ('features', ctypes.c_uint32),
        ('base_address', ctypes.POINTER(ctypes.c_void_p)),
        ('doorbell_signal', ctypes.c_uint64),
        ('size', ctypes.c_uint32), ('reserved1', ctypes.c_uint32),
        ('id', ctypes.c_uint64),
    ]
class hsa_signal_t(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint64)]
class hsa_region_t(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint64)]

AGENT_CB = ctypes.CFUNCTYPE(ctypes.c_int, hsa_agent_t, ctypes.c_void_p)
REGION_CB = ctypes.CFUNCTYPE(ctypes.c_int, hsa_region_t, ctypes.c_void_p)

hsa.hsa_init.restype = ctypes.c_int
hsa.hsa_iterate_agents.argtypes = [AGENT_CB, ctypes.c_void_p]
hsa.hsa_iterate_agents.restype = ctypes.c_int
hsa.hsa_agent_get_info.argtypes = [hsa_agent_t, ctypes.c_uint32, ctypes.c_void_p]
hsa.hsa_agent_get_info.restype = ctypes.c_int
hsa.hsa_queue_create.argtypes = [
    hsa_agent_t, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(hsa_queue_t))
]
hsa.hsa_queue_create.restype = ctypes.c_int
hsa.hsa_signal_store_relaxed.argtypes = [hsa_signal_t, ctypes.c_int64]
hsa.hsa_agent_iterate_regions.argtypes = [hsa_agent_t, REGION_CB, ctypes.c_void_p]
hsa.hsa_agent_iterate_regions.restype = ctypes.c_int
hsa.hsa_region_get_info.argtypes = [hsa_region_t, ctypes.c_uint32, ctypes.c_void_p]
hsa.hsa_region_get_info.restype = ctypes.c_int
hsa.hsa_memory_allocate.argtypes = [hsa_region_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
hsa.hsa_memory_allocate.restype = ctypes.c_int

def PACKET3(op, n):
    return (3 << 30) | ((n & 0x3FFF) << 16) | ((op & 0xFF) << 8)

def read_reg(off):
    with open(REGS, 'rb') as f:
        f.seek(off * 4)
        return struct.unpack('<I', f.read(4))[0]

def write_reg(off, val):
    with open(REGS, 'r+b') as f:
        f.seek(off * 4)
        f.write(struct.pack('<I', val))
        f.flush()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- PM4 packet builders ---

def build_nop():
    """PM4 NOP packet."""
    return struct.pack("<II", PACKET3(0x10, 0), 0)

def build_write_data_mem(gpu_va, value):
    """PM4 WRITE_DATA to memory (dst_sel=5, confirmed working)."""
    header = PACKET3(0x37, 3)
    control = (5 << 8) | (1 << 20)  # dst_sel=5(mem-async), wr_confirm=1
    return struct.pack("<IIIII", header, control,
                       gpu_va & 0xFFFFFFFF, (gpu_va >> 32) & 0xFFFFFFFF, value)

def build_write_data_reg(reg_offset, value, engine_sel=0):
    """PM4 WRITE_DATA to register (dst_sel=0).

    reg_offset: register offset in DWORDs (e.g. 0x5846)
    engine_sel: 0=ME, 1=PFP, 2=CE

    For dst_sel=0: DW2 = register byte offset = reg_offset * 4
    """
    header = PACKET3(0x37, 3)
    # dst_sel=0 (register), wr_confirm=1, engine_sel
    control = (0 << 8) | (1 << 20) | (engine_sel << 30)
    reg_byte = reg_offset * 4
    return struct.pack("<IIIII", header, control, reg_byte, 0, value)

def build_copy_data_imm_to_reg(reg_offset, value):
    """PM4 COPY_DATA: immediate value → register.

    Opcode 0x40. Different internal path than WRITE_DATA.
    src_sel=0 (register/immediate), dst_sel=0 (register)
    """
    header = PACKET3(0x40, 4)
    # src_sel=0 (reg), dst_sel=0 (reg), but we want immediate...
    # Actually COPY_DATA: bits[3:0]=src_sel, bits[11:8]=dst_sel
    # src_sel=5 (immediate data), dst_sel=0 (register)
    control = (5 << 0) | (0 << 8)
    src_lo = value  # immediate value
    src_hi = 0
    dst_lo = reg_offset * 4  # register byte offset
    dst_hi = 0
    return struct.pack("<IIIIII", header, control, src_lo, src_hi, dst_lo, dst_hi)

# IC register offsets
ME_IC_BASE_LO   = 0x5844
ME_IC_BASE_HI   = 0x5845
ME_IC_BASE_CNTL = 0x5846
ME_IC_OP_CNTL   = 0x5847

PFP_IC_BASE_LO   = 0x5840
PFP_IC_BASE_CNTL = 0x5842
PFP_IC_OP_CNTL   = 0x5843

# Some RLC registers (GFX11)
RLC_CNTL          = 0x4C00
RLC_CGCG_CGLS_CTRL = 0x4C48
RLC_CP_SCHEDULERS  = 0x4CA4
RLC_SAFE_MODE      = 0x4C50

def submit_pm4(ring, doorbell, wptr_ptr, pkt_bytes, wptr_offset):
    """Submit PM4 packet and ring doorbell."""
    ring_off = wptr_offset * 4
    ctypes.memmove(ring + ring_off, pkt_bytes, len(pkt_bytes))
    new_wptr = wptr_offset + len(pkt_bytes) // 4

    if wptr_ptr:
        wptr_ptr.contents.value = new_wptr
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(new_wptr))

    return new_wptr

def save_results(results, tag=""):
    """Save intermediate results after each test to prevent data loss on crash."""
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2351_pm4_reg_write.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    if tag:
        log(f"  [saved intermediate: {tag}]")

def main():
    import sys
    skip_rlc = '--skip-rlc' in sys.argv
    skip_pfp = '--skip-pfp' in sys.argv
    results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}
    log("=== z2351: PM4 Register Write to IC Registers ===")

    # Init HSA
    status = hsa.hsa_init()
    if status != 0:
        log(f"hsa_init failed: {status}")
        return

    # Find GPU
    gpu = hsa_agent_t(0)
    def find_gpu(a, d):
        dt = ctypes.c_uint32(0)
        hsa.hsa_agent_get_info(a, 17, ctypes.byref(dt))
        if dt.value == 1:
            gpu.handle = a.handle
            return 1
        return 0
    hsa.hsa_iterate_agents(AGENT_CB(find_gpu), None)

    # Find kernarg region
    kernarg = hsa_region_t(0)
    def find_regions(r, d):
        seg = ctypes.c_uint32(0)
        hsa.hsa_region_get_info(r, 0, ctypes.byref(seg))
        if seg.value == 0:
            flags = ctypes.c_uint32(0)
            hsa.hsa_region_get_info(r, 1, ctypes.byref(flags))
            if flags.value & 0x1:
                kernarg.handle = r.handle
        return 0
    hsa.hsa_agent_iterate_regions(gpu, REGION_CB(find_regions), None)

    # Allocate test buffer for PM4 memory writes (verification)
    test_buf = ctypes.c_void_p(0)
    hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(test_buf))
    buf_addr = test_buf.value
    arr = (ctypes.c_uint32 * 1024).from_address(buf_addr)
    arr[0] = 0xDEADBEEF
    log(f"  Test buffer at 0x{buf_addr:x}")

    # Create PM4 queue
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        log(f"Queue create failed: {status}")
        os._exit(1)

    q = qp.contents
    ring = ctypes.cast(q.base_address, ctypes.c_void_p).value
    doorbell = hsa_signal_t(q.doorbell_signal)

    wptr_ptr = None
    try:
        with open('/tmp/pm4_queue_info', 'r') as f:
            for line in f:
                if line.startswith('wptr='):
                    wptr_addr = int(line.strip().split('=')[1], 16)
                    wptr_ptr = ctypes.cast(wptr_addr, ctypes.POINTER(ctypes.c_uint64))
    except:
        pass

    log(f"  Queue: type={q.type} ring=0x{ring:x}")
    wptr = 0

    # === BASELINE: Verify PM4 memory write works ===
    log("\n=== BASELINE: PM4 WRITE_DATA to memory ===")
    marker = 0xBEEF0001
    pkt = build_nop() + build_write_data_mem(buf_addr, marker)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.3)
    result = arr[0]
    ok = (result == marker)
    log(f"  Wrote 0x{marker:08X}, read 0x{result:08X} {'PASS' if ok else 'FAIL'}")
    results['baseline_mem_write'] = ok
    save_results(results, "baseline")
    if not ok:
        log("  PM4 queue not working. Aborting.")
        os._exit(1)

    # === TEST 1: WRITE_DATA dst_sel=0 to ME_IC_BASE_CNTL ===
    log("\n=== TEST 1: PM4 WRITE_DATA dst_sel=register to ME_IC_BASE_CNTL ===")
    pre_cntl = read_reg(ME_IC_BASE_CNTL)
    log(f"  Pre:  ME_IC_BASE_CNTL = 0x{pre_cntl:08X}")

    # Write a known value — change VMID from 15→0 via PM4
    test_val = (pre_cntl & ~0xF) | 0  # VMID=0
    log(f"  PM4 WRITE_DATA reg → 0x{test_val:08X} (VMID=0)")
    pkt = build_write_data_reg(ME_IC_BASE_CNTL, test_val, engine_sel=0)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.3)

    post_cntl = read_reg(ME_IC_BASE_CNTL)
    changed = (post_cntl != pre_cntl)
    log(f"  Post: ME_IC_BASE_CNTL = 0x{post_cntl:08X} {'CHANGED!' if changed else 'unchanged'}")
    results['wd_reg_ic_base_cntl'] = {
        'pre': f"0x{pre_cntl:08X}",
        'wrote': f"0x{test_val:08X}",
        'post': f"0x{post_cntl:08X}",
        'changed': changed,
    }

    # Verify ME still alive
    arr[0] = 0xDEADBEEF
    marker2 = 0xBEEF0002
    pkt = build_nop() + build_write_data_mem(buf_addr, marker2)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.3)
    me_alive = (arr[0] == marker2)
    log(f"  ME alive: {me_alive} (0x{arr[0]:08X})")
    results['wd_reg_me_alive'] = me_alive
    save_results(results, "test1_wd_reg")

    # Restore via PM4 if changed
    if changed:
        pkt = build_write_data_reg(ME_IC_BASE_CNTL, pre_cntl, engine_sel=0)
        wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
        time.sleep(0.1)

    # === TEST 2: WRITE_DATA dst_sel=0 to ME_IC_OP_CNTL (invalidate) ===
    log("\n=== TEST 2: PM4 WRITE_DATA dst_sel=register to ME_IC_OP_CNTL ===")
    pre_op = read_reg(ME_IC_OP_CNTL)
    log(f"  Pre:  ME_IC_OP_CNTL = 0x{pre_op:08X}")

    # Try invalidate via PM4 register write
    log(f"  PM4 WRITE_DATA reg → IC_INVALIDATE (bit 0)")
    pkt = build_write_data_reg(ME_IC_OP_CNTL, 0x01, engine_sel=0)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.3)

    post_op = read_reg(ME_IC_OP_CNTL)
    inv_complete = bool(post_op & 0x02)
    log(f"  Post: ME_IC_OP_CNTL = 0x{post_op:08X} INV_COMPLETE={inv_complete}")
    results['wd_reg_ic_op_invalidate'] = {
        'pre': f"0x{pre_op:08X}",
        'post': f"0x{post_op:08X}",
        'inv_complete': inv_complete,
    }

    # ME alive after invalidate?
    arr[0] = 0xDEADBEEF
    marker3 = 0xBEEF0003
    pkt = build_nop() + build_write_data_mem(buf_addr, marker3)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.3)
    me_alive2 = (arr[0] == marker3)
    log(f"  ME alive after invalidate: {me_alive2}")
    results['wd_reg_inv_me_alive'] = me_alive2

    save_results(results, "test2_invalidate")

    # Try prime via PM4
    if me_alive2:
        log(f"  PM4 WRITE_DATA reg → IC_PRIME (bit 4)")
        pkt = build_write_data_reg(ME_IC_OP_CNTL, 0x10, engine_sel=0)
        wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
        time.sleep(0.3)

        post_op2 = read_reg(ME_IC_OP_CNTL)
        primed = bool(post_op2 & 0x20)
        log(f"  Post: ME_IC_OP_CNTL = 0x{post_op2:08X} PRIMED={primed}")
        results['wd_reg_ic_prime'] = {
            'post': f"0x{post_op2:08X}",
            'primed': primed,
        }

    # === TEST 3: COPY_DATA immediate→register for IC_BASE_CNTL ===
    log("\n=== TEST 3: PM4 COPY_DATA immediate→register IC_BASE_CNTL ===")
    pre_cntl2 = read_reg(ME_IC_BASE_CNTL)
    test_val2 = (pre_cntl2 & ~0xF) | 0x07  # VMID=7

    log(f"  Pre:  ME_IC_BASE_CNTL = 0x{pre_cntl2:08X}")
    log(f"  COPY_DATA imm→reg → 0x{test_val2:08X} (VMID=7)")
    pkt = build_copy_data_imm_to_reg(ME_IC_BASE_CNTL, test_val2)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.3)

    post_cntl2 = read_reg(ME_IC_BASE_CNTL)
    changed2 = (post_cntl2 != pre_cntl2)
    log(f"  Post: ME_IC_BASE_CNTL = 0x{post_cntl2:08X} {'CHANGED!' if changed2 else 'unchanged'}")
    results['copy_data_ic_base_cntl'] = {
        'pre': f"0x{pre_cntl2:08X}",
        'wrote': f"0x{test_val2:08X}",
        'post': f"0x{post_cntl2:08X}",
        'changed': changed2,
    }

    # Restore if changed
    if changed2:
        pkt = build_copy_data_imm_to_reg(ME_IC_BASE_CNTL, pre_cntl2)
        wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
        time.sleep(0.1)

    # ME alive?
    arr[0] = 0xDEADBEEF
    marker4 = 0xBEEF0004
    pkt = build_nop() + build_write_data_mem(buf_addr, marker4)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.3)
    me_alive3 = (arr[0] == marker4)
    log(f"  ME alive: {me_alive3}")
    results['copy_data_me_alive'] = me_alive3
    save_results(results, "test3_copy_data")

    # === TEST 4: PM4 WRITE_DATA to PFP IC registers (via engine_sel=1/PFP) ===
    if skip_pfp:
        log("\n=== TEST 4: SKIPPED (--skip-pfp) ===")
        results['pfp_skipped'] = True
        save_results(results, "test4_skipped")
    else:
        log("\n=== TEST 4: PM4 WRITE_DATA to PFP_IC_BASE_CNTL (engine_sel=PFP) ===")
        pre_pfp = read_reg(PFP_IC_BASE_CNTL)
        log(f"  Pre:  PFP_IC_BASE_CNTL = 0x{pre_pfp:08X}")

        # engine_sel=1 routes through PFP instead of ME
        test_pfp = pre_pfp | (1 << 23)  # Set EXE_DISABLE
        log(f"  PM4 WRITE_DATA reg (engine=PFP) → 0x{test_pfp:08X}")
        pkt = build_write_data_reg(PFP_IC_BASE_CNTL, test_pfp, engine_sel=1)
        wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
        time.sleep(0.3)

        post_pfp = read_reg(PFP_IC_BASE_CNTL)
        pfp_changed = (post_pfp != pre_pfp)
        log(f"  Post: PFP_IC_BASE_CNTL = 0x{post_pfp:08X} {'CHANGED!' if pfp_changed else 'unchanged'}")
        results['wd_pfp_ic_base_cntl'] = {
            'pre': f"0x{pre_pfp:08X}",
            'wrote': f"0x{test_pfp:08X}",
            'post': f"0x{post_pfp:08X}",
            'changed': pfp_changed,
        }

        # Restore
        if pfp_changed:
            pkt = build_write_data_reg(PFP_IC_BASE_CNTL, pre_pfp, engine_sel=1)
            wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
            time.sleep(0.1)
        save_results(results, "test4_pfp")

    # === TEST 5: Read RLC registers to check accessibility (READ-ONLY, safe) ===
    log("\n=== TEST 5: RLC register accessibility (read-only) ===")
    rlc_regs = {
        'RLC_CNTL': 0x4C00,
        'RLC_CGCG_CGLS_CTRL': 0x4C48,
        'RLC_CP_SCHEDULERS': 0x4CA4,
        'RLC_SAFE_MODE': 0x4C50,
        'RLC_GPM_GENERAL_0': 0x4C80,
        'RLC_GPM_GENERAL_1': 0x4C81,
        'RLC_GPM_GENERAL_2': 0x4C82,
        'RLC_GPM_GENERAL_3': 0x4C83,
        'RLC_GPM_GENERAL_4': 0x4C84,
        'RLC_GPM_GENERAL_7': 0x4C87,
        'RLC_GPM_GENERAL_12': 0x4C8C,
    }

    rlc_vals = {}
    for name, off in rlc_regs.items():
        try:
            val = read_reg(off)
            log(f"  {name:30s} (0x{off:04X}) = 0x{val:08X}")
            rlc_vals[name] = f"0x{val:08X}"
        except Exception as e:
            log(f"  {name:30s} (0x{off:04X}) = ERROR: {e}")
            rlc_vals[name] = f"ERROR: {e}"
    results['rlc_registers'] = rlc_vals
    save_results(results, "test5_rlc_read")

    # === TEST 6: PM4 WRITE_DATA to RLC_SAFE_MODE — DANGEROUS, skip by default ===
    if skip_rlc:
        log("\n=== TEST 6: SKIPPED (--skip-rlc) — RLC writes may crash ===")
        results['rlc_write_skipped'] = True
        save_results(results, "test6_skipped")
    else:
        log("\n=== TEST 6: PM4 WRITE_DATA to RLC_SAFE_MODE ===")
        log("  WARNING: RLC safe mode write — may cause crash")
        pre_safe = read_reg(RLC_SAFE_MODE)
        log(f"  Pre:  RLC_SAFE_MODE = 0x{pre_safe:08X}")

        # Write 1 to enter safe mode
        pkt = build_write_data_reg(RLC_SAFE_MODE, 0x01, engine_sel=0)
        wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
        time.sleep(0.5)

        post_safe = read_reg(RLC_SAFE_MODE)
        log(f"  Post: RLC_SAFE_MODE = 0x{post_safe:08X}")
        results['rlc_safe_mode_write'] = {
            'pre': f"0x{pre_safe:08X}",
            'post': f"0x{post_safe:08X}",
            'changed': (post_safe != pre_safe),
        }
        save_results(results, "test6_rlc_safe")

        # If safe mode entered, try IC write again
        if post_safe != pre_safe:
            log("  Safe mode changed! Trying IC_BASE_CNTL write...")
            pre_ic = read_reg(ME_IC_BASE_CNTL)
            test_ic = (pre_ic & ~0xF) | 0
            pkt = build_write_data_reg(ME_IC_BASE_CNTL, test_ic, engine_sel=0)
            wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
            time.sleep(0.3)
            post_ic = read_reg(ME_IC_BASE_CNTL)
            log(f"  IC_BASE_CNTL: 0x{pre_ic:08X} → 0x{post_ic:08X}")
            results['safe_mode_ic_write'] = {
                'pre': f"0x{pre_ic:08X}",
                'post': f"0x{post_ic:08X}",
                'changed': (post_ic != pre_ic),
            }

        # Exit safe mode if entered
        if post_safe != pre_safe:
            pkt = build_write_data_reg(RLC_SAFE_MODE, pre_safe, engine_sel=0)
            wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
            time.sleep(0.2)

    # ME still alive?
    arr[0] = 0xDEADBEEF
    marker5 = 0xBEEF0005
    pkt = build_nop() + build_write_data_mem(buf_addr, marker5)
    wptr = submit_pm4(ring, doorbell, wptr_ptr, pkt, wptr)
    time.sleep(0.5)
    me_alive4 = (arr[0] == marker5)
    log(f"  ME alive after all tests: {me_alive4}")
    results['final_me_alive'] = me_alive4

    # Final register state
    log("\n=== Final IC register state ===")
    for name, off in [('ME_IC_BASE_CNTL', ME_IC_BASE_CNTL), ('ME_IC_OP_CNTL', ME_IC_OP_CNTL),
                       ('PFP_IC_BASE_CNTL', PFP_IC_BASE_CNTL)]:
        val = read_reg(off)
        log(f"  {name} = 0x{val:08X}")

    results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_results(results, "FINAL")
    log(f"\nAll results saved.")

    log("\nExiting without queue destroy.")
    os._exit(0)

if __name__ == "__main__":
    main()
