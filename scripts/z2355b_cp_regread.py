#!/usr/bin/env python3
"""z2355b: Read GPU registers from INSIDE the CP via COPY_DATA PM4.

Uses COPY_DATA (opcode 0x40) with src_sel=0 (register) and
dst_sel=5 (memory-async) to copy register values to GPU-visible
memory. This reads registers from the GPU's perspective, bypassing
CPU-side banking issues.

Then submits survivor opcodes and reads the same registers again
to detect changes.

Usage:
  sudo LD_PRELOAD=scripts/_tmp_backup/hook_pm4.so \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONUNBUFFERED=1 \
    python3 scripts/z2355b_cp_regread.py
"""
import os, sys, struct, ctypes, time, json

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'z2355b_cp_regread.json')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=== z2355b: CP-Internal Register Read ===")

    if os.geteuid() != 0:
        log("ERROR: Must run as root (sudo)")
        sys.exit(1)

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
    hsa.hsa_memory_allocate.argtypes = [hsa_region_t, ctypes.c_size_t,
                                         ctypes.POINTER(ctypes.c_void_p)]
    hsa.hsa_memory_allocate.restype = ctypes.c_int

    def PACKET3(op, n):
        return (3 << 30) | ((n & 0x3FFF) << 16) | ((op & 0xFF) << 8)

    if hsa.hsa_init() != 0:
        log("ERROR: hsa_init failed")
        sys.exit(1)

    gpu = hsa_agent_t(0)
    def find_gpu(a, d):
        dt = ctypes.c_uint32(0)
        hsa.hsa_agent_get_info(a, 17, ctypes.byref(dt))
        if dt.value == 1:
            gpu.handle = a.handle
            return 1
        return 0
    hsa.hsa_iterate_agents(AGENT_CB(find_gpu), None)
    if not gpu.handle:
        log("ERROR: no GPU")
        sys.exit(1)

    kernarg = hsa_region_t(0)
    def find_ka(r, d):
        seg = ctypes.c_uint32(0)
        hsa.hsa_region_get_info(r, 0, ctypes.byref(seg))
        if seg.value == 0:
            flags = ctypes.c_uint32(0)
            hsa.hsa_region_get_info(r, 1, ctypes.byref(flags))
            if flags.value & 0x1:
                kernarg.handle = r.handle
        return 0
    hsa.hsa_agent_iterate_regions(gpu, REGION_CB(find_ka), None)

    # Alloc fence + readback buffers
    fence_buf = ctypes.c_void_p(0)
    hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(fence_buf))
    fence_va = fence_buf.value
    fence_arr = (ctypes.c_uint32 * 1024).from_address(fence_va)

    read_buf = ctypes.c_void_p(0)
    hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(read_buf))
    read_va = read_buf.value
    read_arr = (ctypes.c_uint32 * 1024).from_address(read_va)

    # Create PM4 queue
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 4096, 1, None, None,
                                   0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        log(f"ERROR: queue create failed: {status}")
        sys.exit(1)

    q = qp.contents
    ring_base = ctypes.cast(q.base_address, ctypes.c_void_p).value
    doorbell = hsa_signal_t(q.doorbell_signal)

    wptr_ptr = None
    try:
        with open('/tmp/pm4_queue_info', 'r') as f:
            for line in f:
                if line.startswith('wptr='):
                    wa = int(line.strip().split('=')[1], 16)
                    wptr_ptr = ctypes.cast(wa, ctypes.POINTER(ctypes.c_uint64))
    except:
        pass

    def submit(pkt, wptr):
        ctypes.memmove(ring_base + wptr * 4, pkt, len(pkt))
        nw = wptr + len(pkt) // 4
        if wptr_ptr:
            wptr_ptr.contents.value = nw
        hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(nw))
        return nw

    wptr = 0

    nop = struct.pack("<II", PACKET3(0x10, 0), 0)
    def mk_fence(val):
        h = PACKET3(0x37, 3)
        c = (5 << 8) | (1 << 20)
        return struct.pack("<IIIII", h, c, fence_va & 0xFFFFFFFF,
                           (fence_va >> 32) & 0xFFFFFFFF, val)

    # Baseline check
    fence_arr[0] = 0
    wptr = submit(nop + mk_fence(0xBA5E0001), wptr)
    time.sleep(0.3)
    if fence_arr[0] != 0xBA5E0001:
        log(f"ERROR: baseline fence fail: 0x{fence_arr[0]:08X}")
        sys.exit(1)
    log("PM4 queue alive")

    # COPY_DATA: src_sel=0 (register), dst_sel=5 (memory-async), wr_confirm=1
    # Header: PACKET3(0x40, 4)
    # DW1: src_sel[3:0] | (dst_sel[3:0] << 8) | (wr_confirm << 20)
    # DW2: src_reg_offset (DWORD offset within GC block, NOT including base!)
    # DW3: 0 (src high, unused for register source)
    # DW4: dst_addr_lo
    # DW5: dst_addr_hi

    def mk_copy_data(reg_offset, dst_va_offset):
        """Build COPY_DATA packet: read register → write to memory."""
        dst_va = read_va + dst_va_offset * 4
        hdr = PACKET3(0x40, 4)
        # src_sel=0 (reg), dst_sel=5 (mem-async), wr_confirm=1
        control = 0 | (5 << 8) | (1 << 20)
        return struct.pack("<IIIIII", hdr, control,
                           reg_offset & 0xFFFFFFFF, 0,
                           dst_va & 0xFFFFFFFF,
                           (dst_va >> 32) & 0xFFFFFFFF)

    # Registers to read — using SOC15 offsets that the firmware uses.
    # The CP's register namespace uses the raw register offsets (within GC block),
    # which for BASE_IDX=0 registers is just the offset itself.
    # For BASE_IDX=1 registers, the CP uses the same offsets internally.

    # Start with SAFE registers (BASE_IDX=0, known accessible)
    safe_regs = [
        ('GRBM_STATUS',    0x0DA4),
        ('GRBM_GFX_INDEX', 0x0013),
        ('CP_CPC_STALLED', 0x0E26),
        ('CP_CPF_STATUS',  0x0E27),
        ('CP_CPC_STATUS',  0x0E24),
        ('CP_ME_CNTL',     0x0078),
    ]

    # Potentially dangerous (BASE_IDX=1) — the CP may or may not be able
    # to read these via COPY_DATA on a user compute queue.
    risky_regs = [
        ('CP_HQD_ACTIVE',     0x3247),
        ('CP_HQD_VMID',       0x3229),
        ('CP_HQD_PQ_BASE_LO', 0x320A),
        ('CP_HQD_PQ_BASE_HI', 0x3209),
        ('CP_HQD_PQ_CONTROL', 0x322B),
        ('CP_MEC_DC_AP0_BASE',  0x2948),
        ('CP_MEC_DC_AP0_MASK',  0x2949),
        ('CP_MEC_DC_AP15_BASE', 0x2977),
        ('CP_MEC_DC_AP15_MASK', 0x2978),
        ('CP_MEC_DC_AP15_CNTL', 0x2979),
        ('CP_CPC_IC_OP_CNTL',  0x297A),
        ('CP_MEC_INSTR_BASE_LO', 0x292C),
        ('CP_MEC_INSTR_BASE_HI', 0x292D),
        ('CP_MEC_DC_BASE_CNTL',  0x290B),
    ]

    results = {
        'experiment': 'z2355b',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'read_va': f'0x{read_va:016X}',
        'phases': [],
    }

    def read_regs_via_copy_data(reg_list, label):
        """Read registers via COPY_DATA and return dict of values."""
        nonlocal wptr

        # Clear readback buffer
        for i in range(len(reg_list)):
            read_arr[i] = 0xDEADDEAD

        # Build chain of COPY_DATA packets + final fence
        pkt_chain = b''
        for i, (name, offset) in enumerate(reg_list):
            pkt_chain += mk_copy_data(offset, i)

        fence_val = 0xCD000000 | (hash(label) & 0xFFFFFF)
        fence_arr[0] = 0
        wptr = submit(pkt_chain + mk_fence(fence_val & 0xFFFFFFFF), wptr)

        deadline = time.time() + 3.0
        while fence_arr[0] != (fence_val & 0xFFFFFFFF) and time.time() < deadline:
            time.sleep(0.02)

        if fence_arr[0] != (fence_val & 0xFFFFFFFF):
            log(f"  {label}: COPY_DATA FENCE FAIL (0x{fence_arr[0]:08X}) — PRIV VIOLATION?")
            # Check if queue survived with a recovery NOP+fence
            fence_arr[0] = 0
            wptr = submit(nop * 4 + mk_fence(0xAEC00099), wptr)
            time.sleep(1.0)
            recovered = (fence_arr[0] == 0xAEC00099)
            log(f"  Queue {'survived' if recovered else 'DEAD'}")
            return None, recovered

        vals = {}
        for i, (name, offset) in enumerate(reg_list):
            v = read_arr[i]
            vals[name] = f'0x{v:08X}'
        return vals, True

    # ---- Phase 1: Read SAFE registers ----
    log("Phase 1: Reading safe (BASE_IDX=0) registers via COPY_DATA...")
    safe_vals, alive = read_regs_via_copy_data(safe_regs, 'safe_baseline')
    if safe_vals:
        for name, val in safe_vals.items():
            log(f"  {name:24s} = {val}")
        results['phases'].append({'label': 'safe_baseline', 'registers': safe_vals})
    else:
        log("  FAILED — aborting")
        sys.exit(1)

    # ---- Phase 2: Read RISKY registers one at a time ----
    log("\nPhase 2: Reading risky (BASE_IDX=1) registers via COPY_DATA, one at a time...")
    risky_vals = {}
    for name, offset in risky_regs:
        if not alive:
            log(f"  Queue dead, stopping")
            break

        read_arr[0] = 0xDEADDEAD
        pkt = mk_copy_data(offset, 0)
        fence_val = 0xCD010000 | (offset & 0xFFFF)
        fence_arr[0] = 0
        wptr = submit(pkt + mk_fence(fence_val), wptr)

        deadline = time.time() + 2.0
        while fence_arr[0] != fence_val and time.time() < deadline:
            time.sleep(0.02)

        if fence_arr[0] == fence_val:
            v = read_arr[0]
            risky_vals[name] = f'0x{v:08X}'
            status = 'OK'
            if v == 0xDEADDEAD:
                status = 'UNCHANGED (reg not copied?)'
        else:
            # Check if queue survived
            fence_arr[0] = 0
            wptr = submit(nop * 4 + mk_fence(0xAEC00088), wptr)
            time.sleep(1.0)
            alive = (fence_arr[0] == 0xAEC00088)
            risky_vals[name] = 'PRIV_VIOLATION' if not alive else 'TIMEOUT'
            status = 'PRIV_VIOLATION (queue dead)' if not alive else 'TIMEOUT'

        log(f"  {name:28s} [0x{offset:04X}] = {risky_vals.get(name, '?'):12s}  {status}")

    results['phases'].append({'label': 'risky_baseline', 'registers': risky_vals})

    if not alive:
        log("\nQueue dead after risky reads — cannot test opcodes")
        results['note'] = 'queue_dead_after_risky_reads'
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        log(f"Results: {RESULTS_FILE}")
        return

    # ---- Phase 3: Submit survivor opcodes and re-read registers ----
    log("\nPhase 3: Submit survivor opcodes, then re-read registers...")

    survivors_to_test = [0xA6, 0x95, 0xD0]  # Skip 0x04 (known queue-killer)
    readable_risky = [(n, o) for n, o in risky_regs if risky_vals.get(n) not in ('PRIV_VIOLATION', 'TIMEOUT')]

    for opcode in survivors_to_test:
        if not alive:
            break

        log(f"\n  --- Opcode 0x{opcode:02X} ---")

        # Build body
        if opcode == 0xD0:
            body = [0] * 36
        elif opcode == 0xA6:
            body = [0, 0, 0, 0]
        elif opcode == 0x95:
            body = [0, 0, 0, 0]
        else:
            body = [0, 0, 0, 0]

        n = max(0, len(body) - 1)
        header = PACKET3(opcode, n)
        probe_pkt = struct.pack(f'<I{len(body)}I', header, *body)

        fence_val = 0xF1000000 | (opcode << 8)
        fence_arr[0] = 0
        wptr = submit(probe_pkt + nop + mk_fence(fence_val), wptr)

        deadline = time.time() + 2.0
        while fence_arr[0] != fence_val and time.time() < deadline:
            time.sleep(0.02)

        if fence_arr[0] != fence_val:
            log(f"  Opcode 0x{opcode:02X}: FENCE FAIL")
            fence_arr[0] = 0
            wptr = submit(nop * 8 + mk_fence(0xAEC00077), wptr)
            time.sleep(1.0)
            alive = (fence_arr[0] == 0xAEC00077)
            results['phases'].append({
                'label': f'post_0x{opcode:02X}',
                'fence_ok': False,
                'alive': alive
            })
            continue

        log(f"  Opcode 0x{opcode:02X}: fence OK")

        # Re-read all readable risky registers
        if readable_risky:
            post_vals, alive = read_regs_via_copy_data(readable_risky, f'post_{opcode:02X}')
            if post_vals:
                # Diff against baseline
                changes = []
                for name, _ in readable_risky:
                    b = risky_vals.get(name)
                    a = post_vals.get(name)
                    if b != a and b is not None and a is not None:
                        changes.append({'reg': name, 'before': b, 'after': a})
                        log(f"  *** CHANGED: {name}: {b} → {a}")

                if not changes:
                    log(f"  No risky register changes")

                results['phases'].append({
                    'label': f'post_0x{opcode:02X}',
                    'fence_ok': True,
                    'registers': post_vals,
                    'changes': changes,
                })
            else:
                log(f"  Post-opcode register read failed")
                results['phases'].append({
                    'label': f'post_0x{opcode:02X}',
                    'fence_ok': True,
                    'reg_read_failed': True,
                })

        # Also re-read safe regs
        safe_post, alive = read_regs_via_copy_data(safe_regs, f'safe_post_{opcode:02X}')
        if safe_post:
            for name, offset in safe_regs:
                b = safe_vals.get(name)
                a = safe_post.get(name)
                if b != a:
                    log(f"  *** SAFE CHANGED: {name}: {b} → {a}")

        time.sleep(0.3)

    log("\n=== z2355b COMPLETE ===")
    results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Results: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
