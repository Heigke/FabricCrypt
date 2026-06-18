#!/usr/bin/env python3
"""z2353b: Deep probe of survivor PM4 opcodes — subprocess-isolated

Each test runs in its own subprocess to survive GPU resets.
Tests survivors (0x04, 0x95, 0xA1, 0xA6, 0xD0) and re-tests
NO_OUTPUT opcodes from first run.

Usage:
  sudo LD_PRELOAD=scripts/_tmp_backup/hook_pm4.so \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONUNBUFFERED=1 \
    python3 scripts/z2353_deep_probe.py
"""
import os, sys, struct, ctypes, time, json, subprocess

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'z2353_deep_probe.json')

# Opcodes that survived ALL body sizes in z2353
SURVIVORS = [0x04, 0x95, 0xA1, 0xA6, 0xD0]

# Opcodes that showed NO OUTPUT (likely subprocess crashed before printing)
NO_OUTPUT = [0x62, 0x96, 0xA9, 0xAA, 0xAD, 0xB0, 0xD1, 0xFF]

# Body data patterns to test with each survivor
PATTERNS = {
    'zeros_1': [0],
    'zeros_4': [0, 0, 0, 0],
    'ones_4': [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
    'counting_4': [1, 2, 3, 4],
    'large_8': [0, 0, 0, 0, 0, 0, 0, 0],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_thermal():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0

def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def child_probe(opcode, body_dws_list):
    """Run in subprocess. Test one opcode with specific body DWs.
    Also checks a sentinel buffer for memory writes."""
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

    result = {'opcode': f'0x{opcode:02X}'}

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

    # Alloc fence + sentinel
    fence_buf = ctypes.c_void_p(0)
    hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(fence_buf))
    fence_va = fence_buf.value
    fence_arr = (ctypes.c_uint32 * 1024).from_address(fence_va)

    sentinel_buf = ctypes.c_void_p(0)
    hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(sentinel_buf))
    sentinel_va = sentinel_buf.value
    sentinel_arr = (ctypes.c_uint32 * 1024).from_address(sentinel_va)

    # Fill sentinel with known pattern
    for i in range(256):
        sentinel_arr[i] = 0xCAFE0000 | i

    # Create PM4 queue
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

    # Baseline
    nop = struct.pack("<II", PACKET3(0x10, 0), 0)
    def mk_fence(val):
        h = PACKET3(0x37, 3)
        c = (5 << 8) | (1 << 20)
        return struct.pack("<IIIII", h, c, fence_va & 0xFFFFFFFF,
                           (fence_va >> 32) & 0xFFFFFFFF, val)

    fence_arr[0] = 0
    wptr = submit(nop + mk_fence(0xBA5E0001), wptr)
    time.sleep(0.3)
    if fence_arr[0] != 0xBA5E0001:
        result['error'] = f'baseline fail: 0x{fence_arr[0]:08X}'
        print(json.dumps(result), flush=True)
        return

    result['baseline'] = 'ok'
    result['sentinel_va'] = f'0x{sentinel_va:016X}'

    # Build and submit probe
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

    # Check sentinel for changes
    changes = []
    for i in range(256):
        expected = 0xCAFE0000 | i
        if sentinel_arr[i] != expected:
            changes.append({'idx': i, 'was': f'0x{expected:08X}',
                            'now': f'0x{sentinel_arr[i]:08X}'})
    result['sentinel_changes'] = len(changes)
    if changes:
        result['sentinel_detail'] = changes[:10]  # First 10

    if result['fence_ok']:
        result['status'] = 'ok'
    else:
        # Try recovery
        fence_arr[0] = 0
        wptr = submit(nop * 8 + mk_fence(0xAEC00001), wptr)
        time.sleep(1.0)
        result['recovered'] = (fence_arr[0] == 0xAEC00001)
        result['status'] = 'dw_mismatch' if result['recovered'] else 'mec_stall'

    print(json.dumps(result), flush=True)


def main():
    log("=== z2353b: Deep Probe (subprocess-isolated) ===")

    results = {
        'experiment': 'z2353b',
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

    # TEST 1: Re-verify survivors with zero body
    log("\n--- Phase 1: Re-verify 5 survivors ---")
    for opcode in SURVIVORS:
        temp = get_thermal()
        if temp > 85:
            log(f"THERMAL ABORT: {temp:.1f}°C")
            break
        if temp > 75:
            log(f"Cooling ({temp:.1f}°C)...")
            while get_thermal() > 55:
                time.sleep(1)

        r = run_child(opcode, '0,0,0,0')
        status = r.get('status', r.get('error', '?'))
        sentinel = r.get('sentinel_changes', 0)
        log(f"  0x{opcode:02X} [zeros_4]: {status}  sentinel_changes={sentinel}")
        results['probes'].append({'phase': 'reverify', **r})
        save_results(results)

        if status == 'mec_stall':
            time.sleep(4)
        else:
            time.sleep(0.5)

    # TEST 2: Re-test NO_OUTPUT opcodes (they may work after clean GPU state)
    log("\n--- Phase 2: Re-test NO_OUTPUT opcodes ---")
    for opcode in NO_OUTPUT:
        temp = get_thermal()
        if temp > 85:
            log(f"THERMAL ABORT: {temp:.1f}°C")
            break
        if temp > 75:
            log(f"Cooling ({temp:.1f}°C)...")
            while get_thermal() > 55:
                time.sleep(1)

        r = run_child(opcode, '0,0,0,0')
        status = r.get('status', r.get('error', '?'))
        log(f"  0x{opcode:02X}: {status}")
        results['probes'].append({'phase': 'retest_nooutput', **r})
        save_results(results)

        if status == 'mec_stall':
            time.sleep(4)
        else:
            time.sleep(0.5)

    # TEST 3: Survivors with sentinel_va in body (check memory write)
    log("\n--- Phase 3: Survivors with sentinel VA in body ---")
    for opcode in SURVIVORS:
        temp = get_thermal()
        if temp > 85:
            log(f"THERMAL ABORT: {temp:.1f}°C")
            break
        if temp > 75:
            log(f"Cooling ({temp:.1f}°C)...")
            while get_thermal() > 55:
                time.sleep(1)

        # Can't pass sentinel_va since child allocates it — pass special marker
        r = run_child(opcode, '1,2,3,4')
        status = r.get('status', r.get('error', '?'))
        sentinel = r.get('sentinel_changes', 0)
        log(f"  0x{opcode:02X} [1,2,3,4]: {status}  sentinel_changes={sentinel}")
        results['probes'].append({'phase': 'counting', **r})
        save_results(results)

        if status == 'mec_stall':
            time.sleep(4)
        else:
            time.sleep(0.5)

    log("\n=== DEEP PROBE COMPLETE ===")
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
