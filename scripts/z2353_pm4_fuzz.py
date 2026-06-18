#!/usr/bin/env python3
"""z2353: PM4 Undocumented Opcode Fuzzer for MEC (gfx11.5.1) — v2

Tests 33 undocumented PM4 opcodes found in MEC firmware (z2339).
Each opcode tested in a SEPARATE SUBPROCESS to survive GPU resets.

Key insight from v1: first unknown opcode triggered gfx_v11_0_bad_op_irq
→ MEC stall → MODE2 GPU reset. Same queue can't recover.

Strategy v2:
  - Parent process spawns one child per opcode
  - Child creates fresh HSA queue, submits probe, checks fence, exits
  - Parent reads child's stdout for result
  - 5s timeout per child (MODE2 reset takes ~2s)
  - Waits 3s between probes for GPU reset to settle

Usage:
  sudo LD_PRELOAD=scripts/_tmp_backup/hook_pm4.so \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONUNBUFFERED=1 \
    python3 scripts/z2353_pm4_fuzz.py
"""
import os, sys, struct, ctypes, time, json, subprocess, signal

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'z2353_pm4_fuzz.json')
THERMAL_LIMIT = 85

# 33 undocumented PM4 opcodes from z2339 firmware analysis
UNDOCUMENTED_OPCODES = [
    0x02, 0x04, 0x05, 0x07, 0x09, 0x0A, 0x0B, 0x0D, 0x0E, 0x0F,
    0x17, 0x5C, 0x5D, 0x62, 0x63, 0x70, 0x93, 0x95, 0x96, 0x97,
    0x9A, 0x9B, 0xA1, 0xA5, 0xA6, 0xA9, 0xAA, 0xAD, 0xB0, 0xBA,
    0xD0, 0xD1, 0xFF,
]

PROBE_BODY_SIZES = [1, 2, 4, 8]

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


# ============================================================
# CHILD MODE: Test a single opcode+body_size
# ============================================================
def child_probe(opcode, body_dws):
    """Run in subprocess. Tests one opcode at one body size.
    Prints JSON result to stdout."""
    import ctypes, struct, time

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

    result = {'opcode': f'0x{opcode:02X}', 'body_dws': body_dws}

    # Init HSA
    if hsa.hsa_init() != 0:
        result['error'] = 'hsa_init failed'
        print(json.dumps(result), flush=True)
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
    if not gpu.handle:
        result['error'] = 'no GPU'
        print(json.dumps(result), flush=True)
        return

    # Find kernarg region
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

    # Alloc fence
    fence_buf = ctypes.c_void_p(0)
    if hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(fence_buf)) != 0:
        result['error'] = 'fence alloc failed'
        print(json.dumps(result), flush=True)
        return
    fence_va = fence_buf.value
    fence_arr = (ctypes.c_uint32 * 1024).from_address(fence_va)

    # Create PM4 queue
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None,
                                   0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        result['error'] = f'queue create failed: {status}'
        print(json.dumps(result), flush=True)
        return

    q = qp.contents
    ring_base = ctypes.cast(q.base_address, ctypes.c_void_p).value
    doorbell = hsa_signal_t(q.doorbell_signal)

    # wptr from hook
    wptr_ptr = None
    try:
        with open('/tmp/pm4_queue_info', 'r') as f:
            for line in f:
                if line.startswith('wptr='):
                    wptr_addr = int(line.strip().split('=')[1], 16)
                    wptr_ptr = ctypes.cast(wptr_addr, ctypes.POINTER(ctypes.c_uint64))
    except:
        pass

    def submit(pkt_bytes, wptr):
        ring_off = wptr * 4
        ctypes.memmove(ring_base + ring_off, pkt_bytes, len(pkt_bytes))
        new_wptr = wptr + len(pkt_bytes) // 4
        if wptr_ptr:
            wptr_ptr.contents.value = new_wptr
        hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(new_wptr))
        return new_wptr

    wptr = 0

    # Baseline fence
    fence_arr[0] = 0
    baseline_val = 0xBA5E0001
    nop = struct.pack("<II", PACKET3(0x10, 0), 0)
    fence_hdr = PACKET3(0x37, 3)
    fence_ctrl = (5 << 8) | (1 << 20)
    fence_pkt = struct.pack("<IIIII", fence_hdr, fence_ctrl,
                             fence_va & 0xFFFFFFFF, (fence_va >> 32) & 0xFFFFFFFF, baseline_val)
    wptr = submit(nop + fence_pkt, wptr)
    time.sleep(0.3)
    if fence_arr[0] != baseline_val:
        result['error'] = f'baseline failed: 0x{fence_arr[0]:08X}'
        print(json.dumps(result), flush=True)
        return

    result['baseline'] = 'ok'

    # Build probe packet
    n = max(0, body_dws - 1)
    probe_header = PACKET3(opcode, n)
    probe_pkt = struct.pack('<I', probe_header) + b'\x00\x00\x00\x00' * max(1, body_dws)

    # Fence for probe
    probe_fence_val = 0xF0000000 | (opcode << 8) | body_dws
    fence_arr[0] = 0
    fence_pkt2 = struct.pack("<IIIII", fence_hdr, fence_ctrl,
                              fence_va & 0xFFFFFFFF, (fence_va >> 32) & 0xFFFFFFFF, probe_fence_val)

    full_pkt = probe_pkt + nop + fence_pkt2
    wptr = submit(full_pkt, wptr)

    # Wait for fence
    deadline = time.time() + 2.0
    while fence_arr[0] != probe_fence_val and time.time() < deadline:
        time.sleep(0.02)

    fval = fence_arr[0]
    result['fence_ok'] = (fval == probe_fence_val)
    result['fence_val'] = f'0x{fval:08X}'
    result['expected'] = f'0x{probe_fence_val:08X}'

    if not result['fence_ok']:
        # Try recovery with more NOPs
        recovery_val = 0xAEC00001
        fence_arr[0] = 0
        rec_fence = struct.pack("<IIIII", fence_hdr, fence_ctrl,
                                 fence_va & 0xFFFFFFFF, (fence_va >> 32) & 0xFFFFFFFF, recovery_val)
        rec_pkt = nop * 8 + rec_fence
        wptr = submit(rec_pkt, wptr)
        time.sleep(1.0)
        result['recovered'] = (fence_arr[0] == recovery_val)
        if result['recovered']:
            # MEC didn't stall — opcode consumed extra DWs (misparse)
            result['status'] = 'dw_mismatch'
        else:
            result['status'] = 'mec_stall'
    else:
        result['status'] = 'ok'

    print(json.dumps(result), flush=True)


# ============================================================
# PARENT MODE: Orchestrate per-opcode subprocesses
# ============================================================
def main():
    # Support --start N to resume from opcode index N
    start_idx = 0
    for i, arg in enumerate(sys.argv):
        if arg == '--start' and i + 1 < len(sys.argv):
            start_idx = int(sys.argv[i + 1])

    log("=== z2353: PM4 Undocumented Opcode Fuzzer v2 ===")
    log(f"Testing opcodes {start_idx}..{len(UNDOCUMENTED_OPCODES)-1} × {len(PROBE_BODY_SIZES)} sizes")
    log(f"Each opcode in separate subprocess (survives GPU resets)")

    # Load existing results if resuming
    results = {'experiment': 'z2353', 'version': 2,
               'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
               'probes': [], 'summary': {}}
    if start_idx > 0 and os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                results = json.load(f)
            log(f"Loaded {len(results['probes'])} existing probes")
        except:
            pass
    save_results(results)

    hook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '_tmp_backup', 'hook_pm4.so')
    script_path = os.path.abspath(__file__)

    env = os.environ.copy()
    env['LD_PRELOAD'] = hook_path
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    env['PYTHONUNBUFFERED'] = '1'

    for idx, opcode in enumerate(UNDOCUMENTED_OPCODES):
        if idx < start_idx:
            continue
        temp = get_thermal()
        if temp > THERMAL_LIMIT:
            log(f"THERMAL ABORT: {temp:.1f}°C")
            results['aborted'] = 'thermal'
            break

        if temp > 75:
            log(f"Cooling ({temp:.1f}°C)...")
            while get_thermal() > 55:
                time.sleep(1)

        probe_result = {
            'opcode': f'0x{opcode:02X}',
            'timestamp': time.strftime('%H:%M:%S'),
            'sizes': [],
            'status': 'unknown',
        }

        mec_dead = False
        for body_dws in PROBE_BODY_SIZES:
            if mec_dead:
                # Skip remaining sizes — MEC died on this opcode
                break

            # Run child subprocess
            cmd = [sys.executable, script_path, '--child',
                   str(opcode), str(body_dws)]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=10, env=env)
                stdout = proc.stdout.strip()
                # Find JSON line (skip hook output)
                child_result = None
                for line in stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('{'):
                        try:
                            child_result = json.loads(line)
                            break
                        except:
                            pass

                if child_result is None:
                    size_info = {'body_dws': body_dws, 'error': 'no output',
                                 'stdout': stdout[:200], 'stderr': proc.stderr[:200]}
                    probe_result['sizes'].append(size_info)
                    log(f"  0x{opcode:02X} body={body_dws}: NO OUTPUT")
                    continue

                size_info = {'body_dws': body_dws}
                size_info.update(child_result)
                probe_result['sizes'].append(size_info)

                status = child_result.get('status', 'unknown')
                fence_ok = child_result.get('fence_ok', False)
                error = child_result.get('error', '')

                if fence_ok:
                    log(f"  0x{opcode:02X} body={body_dws}: OK (fence passed!)")
                elif status == 'mec_stall':
                    log(f"  0x{opcode:02X} body={body_dws}: MEC STALL → GPU reset")
                    mec_dead = True
                    # Wait for GPU reset to complete
                    time.sleep(4)
                elif status == 'dw_mismatch':
                    log(f"  0x{opcode:02X} body={body_dws}: DW MISMATCH (ate extra DWs)")
                elif error:
                    log(f"  0x{opcode:02X} body={body_dws}: ERROR: {error}")
                else:
                    log(f"  0x{opcode:02X} body={body_dws}: {status}")

            except subprocess.TimeoutExpired:
                size_info = {'body_dws': body_dws, 'error': 'timeout (10s)'}
                probe_result['sizes'].append(size_info)
                log(f"  0x{opcode:02X} body={body_dws}: TIMEOUT (child killed)")
                mec_dead = True
                time.sleep(4)

            except Exception as e:
                size_info = {'body_dws': body_dws, 'error': str(e)}
                probe_result['sizes'].append(size_info)
                log(f"  0x{opcode:02X} body={body_dws}: EXCEPTION: {e}")

            # Brief pause between sizes
            time.sleep(0.5)

        # Determine overall status for this opcode
        statuses = [s.get('status', s.get('error', 'unknown'))
                     for s in probe_result['sizes']]
        if all(s == 'ok' for s in statuses):
            probe_result['status'] = 'survived_all'
        elif any(s == 'mec_stall' for s in statuses):
            probe_result['status'] = 'mec_stall'
        elif any(s == 'dw_mismatch' for s in statuses):
            probe_result['status'] = 'dw_mismatch'
        elif any('error' in s for s in probe_result['sizes']):
            probe_result['status'] = 'error'
        else:
            probe_result['status'] = 'mixed'

        results['probes'].append(probe_result)
        results['opcodes_tested'] = idx + 1
        save_results(results)

        # After MEC stall, wait extra for GPU to fully recover
        if mec_dead:
            log(f"  Waiting for GPU recovery...")
            time.sleep(3)

    # === SUMMARY ===
    log("\n=== SUMMARY ===")
    survived = [p for p in results['probes'] if p['status'] == 'survived_all']
    stalled = [p for p in results['probes'] if p['status'] == 'mec_stall']
    mismatch = [p for p in results['probes'] if p['status'] == 'dw_mismatch']

    log(f"Tested: {len(results['probes'])}/{len(UNDOCUMENTED_OPCODES)} opcodes")
    log(f"Survived (fence OK): {len(survived)} — {[p['opcode'] for p in survived]}")
    log(f"MEC stall (bad opcode): {len(stalled)} — {[p['opcode'] for p in stalled]}")
    log(f"DW mismatch (ate extra): {len(mismatch)} — {[p['opcode'] for p in mismatch]}")

    if survived:
        log("\n*** INTERESTING: These opcodes were ACCEPTED by MEC! ***")
        for p in survived:
            log(f"  {p['opcode']}: all {len(PROBE_BODY_SIZES)} body sizes passed")

    if mismatch:
        log("\n*** INTERESTING: These opcodes consumed wrong # of DWs! ***")
        for p in mismatch:
            log(f"  {p['opcode']}: MEC recovered after extra NOPs")

    results['summary'] = {
        'total': len(results['probes']),
        'survived': len(survived),
        'mec_stall': len(stalled),
        'dw_mismatch': len(mismatch),
        'survived_opcodes': [p['opcode'] for p in survived],
        'stall_opcodes': [p['opcode'] for p in stalled],
        'mismatch_opcodes': [p['opcode'] for p in mismatch],
    }
    save_results(results)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == '__main__':
    if '--child' in sys.argv:
        # Child mode: test single opcode
        idx = sys.argv.index('--child')
        opcode = int(sys.argv[idx + 1])
        body_dws = int(sys.argv[idx + 2])
        child_probe(opcode, body_dws)
    else:
        main()
