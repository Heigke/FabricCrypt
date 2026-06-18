#!/usr/bin/env python3
"""z2354: Targeted register probe of survivor PM4 opcodes

Tests whether survivor opcodes (0x04, 0x95, 0xA1, 0xA6, 0xD0) actually
modify registers on user compute queues by reading registers via debugfs
before and after each opcode submission.

KEY FINDINGS from z2353 handler analysis:
  0xA6: Writes CP_MEC_DC_APERTURE15_BASE (0x2977) and MASK (0x2978)
  0x95: Writes CP_CPC_STALLED_STAT1 (0x0E26), polls CP_CPF_STATUS (0x0E27)
        Has sub-dispatch mechanism (5+ sub-functions)
  0xD0: Reads PM4 body[35], writes to HQD registers (0x322B, 0x3247)
  0x04: Reads 7 DWORDs from body, writes 5 consecutive regs (0x07D8-0x07E8)
  0xA1: Reads/writes CPC regs (0x0768, 0x0762, 0x0758, 0x075D)

Usage:
  sudo LD_PRELOAD=scripts/_tmp_backup/hook_pm4.so \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONUNBUFFERED=1 \
    python3 scripts/z2354_register_probe.py
"""
import os, sys, struct, ctypes, time, json, subprocess

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'z2354_register_probe.json')

# Registers to monitor (word offset → name)
MONITOR_REGS = {
    # MEC DC aperture 15 (written by 0xA6)
    0x2977: 'CP_MEC_DC_APERTURE15_BASE',
    0x2978: 'CP_MEC_DC_APERTURE15_MASK',
    0x2979: 'CP_MEC_DC_APERTURE15_CNTL',
    # CPC stall/status (written by 0x95)
    0x0E26: 'CP_CPC_STALLED_STAT1',
    0x0E27: 'CP_CPF_STATUS',
    # IC control (adjacent to what 0xA6 writes!)
    0x297a: 'CP_CPC_IC_OP_CNTL',
    # MEC instruction base
    0x292c: 'CP_MEC_LOCAL_INSTR_BASE_LO',
    0x292d: 'CP_MEC_LOCAL_INSTR_BASE_HI',
    0x2930: 'CP_MEC_LOCAL_INSTR_APERTURE',
    # DC base
    0x290b: 'CP_MEC_DC_BASE_CNTL',
    0x290c: 'CP_MEC_DC_OP_CNTL',
    # MEC control
    0x2900: 'CP_MEC_RS64_PRGRM_CNTR_START',
    0x2903: 'CP_MEC_ISA_CNTL',
    # HQD registers (written by 0xD0)
    0x322B: 'CP_HQD_or_SX_related_322B',
    0x3247: 'SX_PERFCOUNTER3_HI',
    # Other context regs (written by 0xA6)
    0x2997: 'UNDOCUMENTED_2997',
    0x2998: 'UNDOCUMENTED_2998',
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_thermal():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0

def read_debugfs_reg(reg_offset):
    """Read a register via debugfs amdgpu_regs. Returns value or None."""
    mmio_offset = reg_offset * 4
    try:
        with open('/sys/kernel/debug/dri/1/amdgpu_regs', 'r+') as f:
            f.write(f'0x{mmio_offset:x} 0x{mmio_offset:x}')
            f.seek(0)
            line = f.readline().strip()
            # Format: "0xOFFSET 0xVALUE"
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1], 16)
    except Exception as e:
        return None
    return None

def read_all_monitored():
    """Read all monitored registers."""
    vals = {}
    for reg in sorted(MONITOR_REGS.keys()):
        v = read_debugfs_reg(reg)
        vals[reg] = v
    return vals

def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def child_probe(opcode, body_dws_list):
    """Run in subprocess. Submit one opcode and check fence."""
    import ctypes, struct, time, json

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

    result = {'opcode': f'0x{opcode:02X}', 'body_count': len(body_dws_list)}

    if hsa.hsa_init() != 0:
        result['error'] = 'hsa_init failed'
        print(json.dumps(result), flush=True)
        return

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
        result['error'] = 'no GPU'
        print(json.dumps(result), flush=True)
        return

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

    fence_buf = ctypes.c_void_p(0)
    hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(fence_buf))
    fence_va = fence_buf.value
    fence_arr = (ctypes.c_uint32 * 1024).from_address(fence_va)

    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None,
                                   0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        result['error'] = f'queue failed: {status}'
        print(json.dumps(result), flush=True)
        return

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

    # Baseline
    fence_arr[0] = 0
    wptr = submit(nop + mk_fence(0xBA5E0001), wptr)
    time.sleep(0.3)
    if fence_arr[0] != 0xBA5E0001:
        result['error'] = f'baseline fail: 0x{fence_arr[0]:08X}'
        print(json.dumps(result), flush=True)
        return

    result['baseline'] = 'ok'

    # Submit probe
    n = max(0, len(body_dws_list) - 1)
    header = PACKET3(opcode, n)
    probe_pkt = struct.pack(f'<I{len(body_dws_list)}I', header, *body_dws_list)

    fence_val = 0xF1000000 | (opcode << 8) | len(body_dws_list)
    fence_arr[0] = 0
    wptr = submit(probe_pkt + nop + mk_fence(fence_val), wptr)

    deadline = time.time() + 2.0
    while fence_arr[0] != fence_val and time.time() < deadline:
        time.sleep(0.02)

    result['fence_ok'] = (fence_arr[0] == fence_val)
    result['fence_val'] = f'0x{fence_arr[0]:08X}'

    if result['fence_ok']:
        result['status'] = 'ok'
    else:
        fence_arr[0] = 0
        wptr = submit(nop * 8 + mk_fence(0xAEC00001), wptr)
        time.sleep(1.0)
        result['recovered'] = (fence_arr[0] == 0xAEC00001)
        result['status'] = 'dw_mismatch' if result['recovered'] else 'mec_stall'

    print(json.dumps(result), flush=True)


def main():
    log("=== z2354: Register Probe of Survivor Opcodes ===")

    results = {
        'experiment': 'z2354',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'probes': [],
    }

    hook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '_tmp_backup', 'hook_pm4.so')
    script_path = os.path.abspath(__file__)
    env = os.environ.copy()
    env['LD_PRELOAD'] = hook_path
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    env['PYTHONUNBUFFERED'] = '1'

    def run_child(opcode, body_dws_str):
        cmd = [sys.executable, script_path, '--child',
               str(opcode), body_dws_str]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=15, env=env)
            for line in proc.stdout.strip().split('\n'):
                if line.strip().startswith('{'):
                    try:
                        return json.loads(line.strip())
                    except:
                        pass
            return {'error': 'no json', 'stdout': proc.stdout[:200],
                    'stderr': proc.stderr[:200]}
        except subprocess.TimeoutExpired:
            return {'error': 'timeout'}
        except Exception as e:
            return {'error': str(e)}

    # Phase 1: Read baseline registers
    log("\n--- Phase 1: Baseline Register Read ---")
    baseline = read_all_monitored()
    for reg, val in sorted(baseline.items()):
        name = MONITOR_REGS[reg]
        log(f"  0x{reg:04X} {name}: {f'0x{val:08X}' if val is not None else 'UNREADABLE'}")
    results['baseline_regs'] = {f'0x{k:04X}': (f'0x{v:08X}' if v is not None else None)
                                 for k, v in baseline.items()}

    # Phase 2: Test each survivor with register reads before/after
    test_cases = [
        # (opcode, body_dws_str, description)
        (0xA6, '0,0,0,0', 'A6_zeros — should write DC aperture regs'),
        (0xA6, '0,0,0,0,0,0,0,0', 'A6_8dw — larger body'),
        (0x95, '0,0,0,0', '95_zeros — stall/status writer'),
        (0x95, '1,0,0,0', '95_sub1 — try sub-function 1'),
        (0x95, '2,0,0,0', '95_sub2 — try sub-function 2'),
        (0x95, '4,0,0,0', '95_sub4 — try sub-function 4'),
        (0x95, '5,0,0,0', '95_sub5 — try sub-function 5'),
        (0xD0, '0,0,0,0', 'D0_4dw — HQD writer (small body)'),
        # 0xD0 reads body[35], so give it 36 DWORDs with marker at offset 35
        (0xD0, ','.join(['0']*35 + ['305419896']), 'D0_36dw — body[35]=0x12345678'),
        (0x04, '0,0,0,0', '04_zeros — multi-reg writer'),
        (0x04, ','.join(['0']*8), '04_8dw — larger body'),
        (0xA1, '0,0,0,0', 'A1_zeros — CPC reg writer'),
    ]

    log(f"\n--- Phase 2: {len(test_cases)} Targeted Probes ---")
    for opcode, body_str, desc in test_cases:
        temp = get_thermal()
        if temp > 85:
            log(f"THERMAL ABORT: {temp:.1f}°C")
            break
        if temp > 75:
            log(f"Cooling ({temp:.1f}°C)...")
            while get_thermal() > 55:
                time.sleep(1)

        # Read registers BEFORE
        pre = read_all_monitored()

        # Submit opcode
        r = run_child(opcode, body_str)
        status = r.get('status', r.get('error', '?'))

        # Read registers AFTER
        post = read_all_monitored()

        # Find changes
        changes = {}
        for reg in sorted(MONITOR_REGS.keys()):
            pre_v = pre.get(reg)
            post_v = post.get(reg)
            if pre_v != post_v:
                changes[f'0x{reg:04X}'] = {
                    'name': MONITOR_REGS[reg],
                    'before': f'0x{pre_v:08X}' if pre_v is not None else None,
                    'after': f'0x{post_v:08X}' if post_v is not None else None,
                }

        change_count = len(changes)
        log(f"  {desc}: {status}  reg_changes={change_count}")
        if changes:
            for regname, ch in changes.items():
                log(f"    CHANGED {regname} {ch['name']}: {ch['before']} -> {ch['after']}")

        probe_result = {
            'test': desc,
            'opcode': f'0x{opcode:02X}',
            'body': body_str,
            'child_result': r,
            'reg_changes': changes,
        }
        results['probes'].append(probe_result)
        save_results(results)

        if status == 'mec_stall':
            time.sleep(4)
        else:
            time.sleep(0.5)

    # Phase 3: Final register state
    log("\n--- Phase 3: Final Register State ---")
    final = read_all_monitored()
    for reg, val in sorted(final.items()):
        name = MONITOR_REGS[reg]
        base_v = baseline.get(reg)
        changed = '*' if val != base_v else ' '
        log(f"  {changed} 0x{reg:04X} {name}: {f'0x{val:08X}' if val is not None else 'UNREADABLE'}")

    results['final_regs'] = {f'0x{k:04X}': (f'0x{v:08X}' if v is not None else None)
                              for k, v in final.items()}

    log("\n=== z2354 REGISTER PROBE COMPLETE ===")
    save_results(results)


if __name__ == '__main__':
    if '--child' in sys.argv:
        idx = sys.argv.index('--child')
        opcode = int(sys.argv[idx + 1])
        body_str = sys.argv[idx + 2]
        body_dws = [int(x) for x in body_str.split(',')]
        child_probe(opcode, body_dws)
    else:
        main()
