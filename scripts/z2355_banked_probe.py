#!/usr/bin/env python3
"""z2355: Banked register probe — snapshot MEC registers while PM4 queue is active.

Creates a PM4 queue via HSA, takes kernel register snapshot (baseline),
submits survivor opcode, takes another snapshot, diffs.

Usage:
  sudo LD_PRELOAD=scripts/_tmp_backup/hook_pm4.so \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONUNBUFFERED=1 \
    python3 scripts/z2355_banked_probe.py
"""
import os, sys, struct, ctypes, time, json, subprocess, re

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'z2355_banked_probe.json')
KO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_bases.ko')

SURVIVORS = [0xA6, 0x95, 0xD0, 0x04, 0xA1]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def take_snapshot(label):
    """Load probe_bases.ko and parse register values from dmesg."""
    subprocess.run(['dmesg', '-C'], capture_output=True)
    subprocess.run(['rmmod', 'probe_bases'], capture_output=True)
    time.sleep(0.1)
    subprocess.run(['insmod', KO_PATH], capture_output=True)
    time.sleep(0.2)

    result = subprocess.run(['dmesg'], capture_output=True, text=True)
    regs = {}
    for line in result.stdout.split('\n'):
        if 'z2355:' not in line:
            continue
        # Parse "ACTIVE ME1 P0 Q3: VMID=2 ..."
        m = re.search(r'ACTIVE ME(\d+) P(\d+) Q(\d+): VMID=(\d+) BASE=0x([0-9A-F]+):([0-9A-F]+) CTRL=0x([0-9A-F]+)', line)
        if m:
            key = f'HQD_ME{m.group(1)}_P{m.group(2)}_Q{m.group(3)}'
            regs[key] = {
                'vmid': int(m.group(4)),
                'base': f'0x{m.group(5)}:{m.group(6)}',
                'ctrl': f'0x{m.group(7)}'
            }
        # Parse "  DC_AP15: base=0x... mask=0x... cntl=0x..."
        m = re.search(r'DC_AP(\d+):\s+base=0x([0-9A-Fa-f]+)\s+mask=0x([0-9A-Fa-f]+)\s+cntl=0x([0-9A-Fa-f]+)', line)
        if m:
            regs[f'DC_AP{m.group(1)}'] = {
                'base': f'0x{m.group(2)}',
                'mask': f'0x{m.group(3)}',
                'cntl': f'0x{m.group(4)}'
            }
        # Parse "  IC: op=0x... base=0x...:... cntl=0x..."
        m = re.search(r'IC: op=0x([0-9A-Fa-f]+) base=0x([0-9A-Fa-f]+):([0-9A-Fa-f]+) cntl=0x([0-9A-Fa-f]+)', line)
        if m:
            regs['IC'] = {
                'op': f'0x{m.group(1)}',
                'base_hi': f'0x{m.group(2)}',
                'base_lo': f'0x{m.group(3)}',
                'cntl': f'0x{m.group(4)}'
            }
        # Parse "  INSTR_BASE: 0x...:... aperture=0x..."
        m = re.search(r'INSTR_BASE: 0x([0-9A-Fa-f]+):([0-9A-Fa-f]+) aperture=0x([0-9A-Fa-f]+)', line)
        if m:
            regs['INSTR_BASE'] = {
                'hi': f'0x{m.group(1)}',
                'lo': f'0x{m.group(2)}',
                'aperture': f'0x{m.group(3)}'
            }
        # Parse "  DC_BASE_CNTL=0x... SET_ID=0x... SET_MASK=0x..."
        m = re.search(r'DC_BASE_CNTL=0x([0-9A-Fa-f]+) SET_ID=0x([0-9A-Fa-f]+) SET_MASK=0x([0-9A-Fa-f]+)', line)
        if m:
            regs['DC_BASE'] = {
                'cntl': f'0x{m.group(1)}',
                'set_id': f'0x{m.group(2)}',
                'set_mask': f'0x{m.group(3)}'
            }
        # Parse "  UNK_28EC=0x..."
        m = re.search(r'UNK_28EC=0x([0-9A-Fa-f]+)', line)
        if m:
            regs['UNK_28EC'] = f'0x{m.group(1)}'
        # Parse "  PQ_WPTR=0x... PQ_RPTR=0x..."
        m = re.search(r'PQ_WPTR=0x([0-9A-Fa-f]+) PQ_RPTR=0x([0-9A-Fa-f]+)', line)
        if m:
            regs['PQ_WPTR'] = f'0x{m.group(1)}'
            regs['PQ_RPTR'] = f'0x{m.group(2)}'
        # Parse "Found N active queues"
        m = re.search(r'Found (\d+) active queues', line)
        if m:
            regs['active_queues'] = int(m.group(1))
        # Parse global registers
        for name in ['CP_MEC_CNTL', 'CP_CPC_STALLED', 'CP_CPF_STATUS', 'CP_CPC_STATUS', 'CPC_PSP_DEBUG']:
            m = re.search(rf'{name}\s+=\s+0x([0-9A-Fa-f]+)', line)
            if m:
                regs[name] = f'0x{m.group(1)}'

    # Capture raw dmesg too
    regs['_raw'] = [l for l in result.stdout.split('\n') if 'z2355:' in l]
    return regs


def diff_snapshots(before, after, label):
    """Compare two snapshots, report changes."""
    changes = []
    all_keys = set(list(before.keys()) + list(after.keys()))
    all_keys.discard('_raw')

    for k in sorted(all_keys):
        b = before.get(k)
        a = after.get(k)
        if b != a:
            changes.append({'register': k, 'before': b, 'after': a})
    return changes


def main():
    log("=== z2355: Banked Register Probe ===")

    # Verify we're root
    if os.geteuid() != 0:
        log("ERROR: Must run as root (sudo)")
        sys.exit(1)

    # Verify probe module exists
    if not os.path.exists(KO_PATH):
        log(f"ERROR: {KO_PATH} not found. Build with make first.")
        sys.exit(1)

    results = {
        'experiment': 'z2355',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'probes': [],
    }

    # ---- HSA setup ----
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
        log("ERROR: no GPU agent")
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

    # Alloc fence buffer
    fence_buf = ctypes.c_void_p(0)
    hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(fence_buf))
    fence_va = fence_buf.value
    fence_arr = (ctypes.c_uint32 * 1024).from_address(fence_va)

    # Create PM4 queue
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None,
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
    log("PM4 queue alive, baseline fence OK")

    # ---- SNAPSHOT 1: With queue active, before any opcode ----
    log("Taking BASELINE register snapshot...")
    snap_before = take_snapshot('baseline')
    active = snap_before.get('active_queues', 0)
    log(f"  Found {active} active queues")
    if active == 0:
        log("WARNING: No active queues visible — banked registers won't be populated")

    # Print key baseline values
    for k in ['IC', 'INSTR_BASE', 'DC_BASE']:
        if k in snap_before:
            log(f"  {k}: {snap_before[k]}")

    results['baseline'] = snap_before

    # ---- TEST EACH SURVIVOR OPCODE ----
    for opcode in SURVIVORS:
        log(f"\n--- Testing opcode 0x{opcode:02X} ---")

        # Re-verify queue is still alive
        fence_arr[0] = 0
        wptr = submit(nop + mk_fence(0xAA000000 | opcode), wptr)
        time.sleep(0.3)
        if fence_arr[0] != (0xAA000000 | opcode):
            log(f"  Queue DEAD before 0x{opcode:02X} — fence=0x{fence_arr[0]:08X}")
            results['probes'].append({
                'opcode': f'0x{opcode:02X}',
                'error': 'queue_dead_before'
            })
            break

        # Build probe packet
        if opcode == 0xD0:
            # 0xD0 handler reads body[35] — give it a 36-DWORD body
            body = [0] * 36
            body[0] = 0xFEE10000  # marker
            body[35] = 0xFEE1FFFF
        elif opcode == 0x04:
            body = [0, 0, 0, 0, 0, 0, 0]  # 7 DWORDs
        elif opcode == 0xA6:
            body = [0, 0, 0, 0]
        elif opcode == 0x95:
            body = [0, 0, 0, 0]
        elif opcode == 0xA1:
            body = [0, 0, 0, 0]
        else:
            body = [0, 0, 0, 0]

        n = max(0, len(body) - 1)
        header = PACKET3(opcode, n)
        probe_pkt = struct.pack(f'<I{len(body)}I', header, *body)

        fence_val = 0xF1000000 | (opcode << 8) | len(body)
        fence_arr[0] = 0
        wptr = submit(probe_pkt + nop + mk_fence(fence_val), wptr)

        deadline = time.time() + 2.0
        while fence_arr[0] != fence_val and time.time() < deadline:
            time.sleep(0.02)

        fence_ok = (fence_arr[0] == fence_val)
        log(f"  Opcode 0x{opcode:02X}: fence={'OK' if fence_ok else f'FAIL(0x{fence_arr[0]:08X})'}")

        if not fence_ok:
            # Try to recover
            fence_arr[0] = 0
            wptr = submit(nop * 8 + mk_fence(0xAEC00001), wptr)
            time.sleep(1.0)
            recovered = (fence_arr[0] == 0xAEC00001)
            log(f"  Recovery: {'OK' if recovered else 'FAILED — queue dead'}")
            results['probes'].append({
                'opcode': f'0x{opcode:02X}',
                'fence_ok': False,
                'recovered': recovered,
                'status': 'recovered' if recovered else 'queue_dead'
            })
            if not recovered:
                break
            continue

        # Take post-opcode snapshot
        log(f"  Taking POST-opcode snapshot...")
        snap_after = take_snapshot(f'post_0x{opcode:02X}')

        changes = diff_snapshots(snap_before, snap_after, f'0x{opcode:02X}')
        if changes:
            log(f"  *** {len(changes)} REGISTER CHANGES DETECTED ***")
            for c in changes:
                log(f"    {c['register']}: {c['before']} → {c['after']}")
        else:
            log(f"  No register changes detected")

        results['probes'].append({
            'opcode': f'0x{opcode:02X}',
            'fence_ok': True,
            'body_size': len(body),
            'changes': changes,
            'post_snapshot': {k: v for k, v in snap_after.items() if k != '_raw'},
        })

        # Save incrementally
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

        time.sleep(0.5)

    # ---- SPECIAL: Test 0x95 with different sub-function selectors ----
    log("\n--- Testing 0x95 sub-functions ---")
    for subfn in [0x00, 0x01, 0x02, 0x03, 0x04, 0x10, 0x20]:
        fence_arr[0] = 0
        wptr = submit(nop + mk_fence(0xBB000000 | subfn), wptr)
        time.sleep(0.3)
        if fence_arr[0] != (0xBB000000 | subfn):
            log(f"  Queue dead before sub-fn 0x{subfn:02X}")
            break

        body = [subfn, 0, 0, 0]
        header = PACKET3(0x95, 3)
        probe_pkt = struct.pack('<IIIII', header, *body)

        fence_val = 0xF2950000 | subfn
        fence_arr[0] = 0
        wptr = submit(probe_pkt + nop + mk_fence(fence_val), wptr)

        deadline = time.time() + 2.0
        while fence_arr[0] != fence_val and time.time() < deadline:
            time.sleep(0.02)

        fence_ok = (fence_arr[0] == fence_val)
        log(f"  0x95 sub=0x{subfn:02X}: fence={'OK' if fence_ok else 'FAIL'}")

        if fence_ok:
            snap = take_snapshot(f'post_0x95_sub{subfn:02X}')
            changes = diff_snapshots(snap_before, snap, f'0x95_sub{subfn:02X}')
            if changes:
                log(f"    *** {len(changes)} CHANGES ***")
                for c in changes:
                    log(f"      {c['register']}: {c['before']} → {c['after']}")
            results['probes'].append({
                'opcode': '0x95',
                'sub_fn': f'0x{subfn:02X}',
                'fence_ok': True,
                'changes': changes,
            })
        else:
            fence_arr[0] = 0
            wptr = submit(nop * 8 + mk_fence(0xAEC00002), wptr)
            time.sleep(1.0)
            recovered = (fence_arr[0] == 0xAEC00002)
            log(f"    Recovery: {'OK' if recovered else 'FAILED'}")
            results['probes'].append({
                'opcode': '0x95',
                'sub_fn': f'0x{subfn:02X}',
                'fence_ok': False,
                'recovered': recovered,
            })
            if not recovered:
                break

        time.sleep(0.5)

    log("\n=== z2355 COMPLETE ===")

    # Final save
    results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    log(f"Results: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
