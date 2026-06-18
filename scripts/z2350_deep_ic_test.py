#!/usr/bin/env python3
"""z2350: Deep IC redirect test with PM4 verification.

Tests if IC invalidate+prime actually causes ME to re-fetch firmware:
1. Dispatch PM4 NOP → verify ME processes it (baseline)
2. Invalidate ME IC
3. Wait for inv_complete
4. Prime ME IC
5. Wait for primed status
6. Dispatch PM4 NOP → verify ME still processes it
7. If ME still works: IC reloaded from same firmware (no effect)
8. Change VMID, write NOPs to VRAM, invalidate+prime, test PM4

Must run as root with LD_PRELOAD hook for PM4 queue.
"""
import struct, time, os, json, mmap, ctypes

REGS = '/sys/kernel/debug/dri/0/amdgpu_regs'
BAR0 = '/sys/bus/pci/devices/0000:c3:00.0/resource0'

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

def build_nop():
    return struct.pack("<II", PACKET3(0x10, 0), 0)

def build_write_mem(gpu_va, value):
    header = PACKET3(0x37, 3)
    control = (5 << 8) | (1 << 20)
    return struct.pack("<IIIII", header, control,
                       gpu_va & 0xFFFFFFFF, (gpu_va >> 32) & 0xFFFFFFFF, value)

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

ME_IC_BASE_CNTL = 0x5846
ME_IC_OP_CNTL   = 0x5847

IC_INVALIDATE    = 1 << 0
IC_INV_COMPLETE  = 1 << 1
IC_PRIME         = 1 << 4
IC_PRIMED        = 1 << 5

def ic_invalidate_and_prime():
    """Full invalidate + prime with status polling."""
    # Invalidate
    write_reg(ME_IC_OP_CNTL, IC_INVALIDATE)

    # Poll for inv_complete (bit 1)
    for i in range(100):
        val = read_reg(ME_IC_OP_CNTL)
        if val & IC_INV_COMPLETE:
            log(f"    IC invalidated (took {i} polls)")
            break
        time.sleep(0.001)
    else:
        log(f"    IC invalidate: no INV_COMPLETE after 100 polls (last=0x{val:08X})")

    # Clear invalidate, then prime
    write_reg(ME_IC_OP_CNTL, 0)
    time.sleep(0.01)
    write_reg(ME_IC_OP_CNTL, IC_PRIME)

    # Poll for primed (bit 5)
    for i in range(100):
        val = read_reg(ME_IC_OP_CNTL)
        if val & IC_PRIMED:
            log(f"    IC primed (took {i} polls)")
            return True
        time.sleep(0.001)

    log(f"    IC prime: no PRIMED status after 100 polls (last=0x{val:08X})")
    return False

def test_pm4(ring, doorbell, wptr_ptr, buf_addr, arr, wptr_offset, label):
    """Submit PM4 NOP + WRITE_DATA and check if ME processes it."""
    marker = 0xBEEF0000 | (int(time.time()) & 0xFFFF)

    # Write test value to buffer
    pkt = build_nop() + build_write_mem(buf_addr, marker)

    ring_off = wptr_offset * 4
    ctypes.memmove(ring + ring_off, pkt, len(pkt))
    new_wptr = wptr_offset + len(pkt) // 4

    if wptr_ptr:
        wptr_ptr.contents.value = new_wptr
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(new_wptr))

    time.sleep(0.5)

    result = arr[0]
    ok = (result == marker)
    log(f"  {label}: wrote marker 0x{marker:08X}, read 0x{result:08X} {'PASS' if ok else 'FAIL'}")
    return ok, new_wptr

def main():
    results = {}

    log("=== z2350: Deep IC Test with PM4 Verification ===")

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

    # Allocate test buffer
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

    # Read wptr from hook
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

    # === TEST 1: Baseline PM4 dispatch ===
    log("\n=== TEST 1: Baseline PM4 dispatch ===")
    ok1, wptr = test_pm4(ring, doorbell, wptr_ptr, buf_addr, arr, 0, "Baseline")
    results['baseline'] = ok1

    if not ok1:
        log("  Baseline failed — PM4 queue not working. Aborting.")
        os._exit(1)

    # === TEST 2: Invalidate + Prime (no changes) + PM4 ===
    log("\n=== TEST 2: IC invalidate+prime (no changes) + PM4 ===")
    log("  Performing IC invalidate+prime...")
    primed = ic_invalidate_and_prime()
    results['safe_prime'] = primed

    arr[0] = 0xDEADBEEF
    ok2, wptr = test_pm4(ring, doorbell, wptr_ptr, buf_addr, arr, wptr, "After safe prime")
    results['after_safe_prime'] = ok2

    if not ok2:
        log("  *** ME STOPPED PROCESSING PM4 AFTER PRIME! ***")
        log("  This means IC prime actually DOES reload firmware!")
        results['ic_prime_effect'] = 'ME_STOPPED'
    else:
        log("  ME still processing PM4 after prime")
        results['ic_prime_effect'] = 'NO_EFFECT'

    # === TEST 3: Change VMID + prime (without VRAM modification) ===
    log("\n=== TEST 3: VMID change + IC prime ===")
    orig_cntl = read_reg(ME_IC_BASE_CNTL)
    log(f"  Original IC_BASE_CNTL = 0x{orig_cntl:08X} (VMID={orig_cntl & 0xF})")

    # Change VMID from 15 to 0
    new_cntl = (orig_cntl & ~0xF) | 0
    write_reg(ME_IC_BASE_CNTL, new_cntl)
    log(f"  Changed IC_BASE_CNTL  = 0x{new_cntl:08X} (VMID=0)")

    # Invalidate + prime
    primed3 = ic_invalidate_and_prime()

    arr[0] = 0xDEADBEEF
    ok3, wptr = test_pm4(ring, doorbell, wptr_ptr, buf_addr, arr, wptr, "After VMID change")
    results['after_vmid_change'] = ok3

    if not ok3:
        log("  *** VMID CHANGE CAUSED ME TO STOP! ***")
        log("  IC is now fetching from VMID=0 address space")
        results['vmid_effect'] = 'ME_STOPPED'
    else:
        log("  ME still processing — VMID change had no effect on IC fetch")
        results['vmid_effect'] = 'NO_EFFECT'

    # Restore VMID
    write_reg(ME_IC_BASE_CNTL, orig_cntl)
    ic_invalidate_and_prime()
    log(f"  Restored IC_BASE_CNTL = 0x{read_reg(ME_IC_BASE_CNTL):08X}")

    # === TEST 4: Clear suspicious bits (8, 9, 12, 14) one at a time ===
    log("\n=== TEST 4: IC_BASE_CNTL bit clearing tests ===")
    for bit in [8, 9, 12, 14]:
        orig = read_reg(ME_IC_BASE_CNTL)
        modified = orig & ~(1 << bit)
        write_reg(ME_IC_BASE_CNTL, modified)
        ic_invalidate_and_prime()

        arr[0] = 0xDEADBEEF
        ok, wptr = test_pm4(ring, doorbell, wptr_ptr, buf_addr, arr, wptr,
                           f"Clear bit[{bit}] (0x{modified:08X})")
        results[f'clear_bit_{bit}'] = ok

        if not ok:
            log(f"  *** BIT[{bit}] IS CRITICAL — ME STOPPED! ***")

        # Restore
        write_reg(ME_IC_BASE_CNTL, orig)
        ic_invalidate_and_prime()

    # Final state
    log("\n=== Final state ===")
    final_cntl = read_reg(ME_IC_BASE_CNTL)
    log(f"  IC_BASE_CNTL = 0x{final_cntl:08X}")

    # Save results
    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2350_deep_ic_test.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {outpath}")

    log("\nExiting without queue destroy.")
    os._exit(0)

if __name__ == "__main__":
    main()
