#!/usr/bin/env python3
"""z2350: Probe VRAM buffer allocation and GPU VA addresses.

Tests whether we can allocate buffers with GPU VA < 4GB,
which is required for IC_BASE_LO redirect (IC_BASE_HI is read-only).

Also verifies PM4 WRITE_DATA works to VRAM (coarse-grained) buffers,
not just kernarg (fine-grained) buffers.
"""
import os, sys, struct, ctypes, time

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

def build_write_mem(gpu_va, value):
    header = PACKET3(0x37, 3)
    control = (5 << 8) | (1 << 20)  # dst_sel=5 (async), wr_confirm
    return struct.pack("<IIIII", header, control,
                       gpu_va & 0xFFFFFFFF, (gpu_va >> 32) & 0xFFFFFFFF, value)

def build_nop():
    return struct.pack("<II", PACKET3(0x10, 0), 0)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=== z2350: VRAM Buffer Probe ===")

    status = hsa.hsa_init()
    if status != 0:
        log(f"hsa_init failed: {status}")
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

    # Find ALL regions and their properties
    regions = []
    def find_regions(r, d):
        seg = ctypes.c_uint32(0)
        hsa.hsa_region_get_info(r, 0, ctypes.byref(seg))  # HSA_REGION_INFO_SEGMENT
        flags = ctypes.c_uint32(0)
        hsa.hsa_region_get_info(r, 1, ctypes.byref(flags))  # HSA_REGION_INFO_GLOBAL_FLAGS
        sz = ctypes.c_size_t(0)
        hsa.hsa_region_get_info(r, 2, ctypes.byref(sz))  # HSA_REGION_INFO_SIZE
        alloc = ctypes.c_bool(False)
        hsa.hsa_region_get_info(r, 3, ctypes.byref(alloc))  # HSA_REGION_INFO_ALLOC_MAX_SIZE...
        regions.append({
            'handle': r.handle,
            'segment': seg.value,
            'flags': flags.value,
            'size': sz.value,
            'allocatable': alloc.value,
        })
        return 0
    hsa.hsa_agent_iterate_regions(gpu, REGION_CB(find_regions), None)

    log(f"Found {len(regions)} regions:")
    for i, r in enumerate(regions):
        seg_name = {0: 'GLOBAL', 1: 'READONLY', 2: 'PRIVATE', 3: 'GROUP', 4: 'KERNARG'}.get(r['segment'], f"?{r['segment']}")
        flag_parts = []
        if r['flags'] & 0x1: flag_parts.append('KERNARG')
        if r['flags'] & 0x2: flag_parts.append('FINE_GRAINED')
        if r['flags'] & 0x4: flag_parts.append('COARSE_GRAINED')
        log(f"  [{i}] handle=0x{r['handle']:x} seg={seg_name} flags={'+'.join(flag_parts) or hex(r['flags'])} size={r['size']//1024//1024}MB alloc={r['allocatable']}")

    # Allocate from each global region and check addresses
    log("\n=== Allocation VA test ===")
    kernarg_region = None
    coarse_region = None
    for r in regions:
        if r['segment'] == 0:
            if r['flags'] & 0x1:
                kernarg_region = hsa_region_t(r['handle'])
            if r['flags'] & 0x4:
                coarse_region = hsa_region_t(r['handle'])

    for name, region in [('kernarg', kernarg_region), ('coarse_vram', coarse_region)]:
        if region is None or region.handle == 0:
            log(f"  {name}: no region available")
            continue
        buf = ctypes.c_void_p(0)
        status = hsa.hsa_memory_allocate(region, 65536, ctypes.byref(buf))
        if status != 0:
            log(f"  {name}: alloc failed ({status})")
            continue
        addr = buf.value
        in_4gb = addr < 0x100000000
        log(f"  {name}: VA=0x{addr:012x} {'< 4GB ✓' if in_4gb else '> 4GB ✗'}")

    # Create PM4 queue and test write to coarse VRAM buffer
    if coarse_region and coarse_region.handle:
        log("\n=== PM4 write to VRAM (coarse) buffer ===")

        # Allocate VRAM buffer
        vram_buf = ctypes.c_void_p(0)
        status = hsa.hsa_memory_allocate(coarse_region, 65536, ctypes.byref(vram_buf))
        if status != 0:
            log(f"  VRAM alloc failed: {status}")
            os._exit(1)
        vram_addr = vram_buf.value
        log(f"  VRAM buffer: 0x{vram_addr:012x}")

        # Fill with pattern via CPU
        arr = (ctypes.c_uint32 * (65536 // 4)).from_address(vram_addr)
        for i in range(8):
            arr[i] = 0xDEADBEEF

        # Create PM4 queue
        qp = ctypes.POINTER(hsa_queue_t)()
        status = hsa.hsa_queue_create(gpu, 1024, 1, None, None, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
        if status != 0:
            log(f"  Queue create failed: {status}")
            os._exit(1)

        q = qp.contents
        ring = ctypes.cast(q.base_address, ctypes.c_void_p).value
        log(f"  Queue: id={q.id} type={q.type} ring=0x{ring:x}")

        # Read wptr from hook
        wptr_addr = None
        try:
            with open('/tmp/pm4_queue_info', 'r') as f:
                for line in f:
                    if line.startswith('wptr='):
                        wptr_addr = int(line.strip().split('=')[1], 16)
        except:
            pass

        wptr_ptr = None
        if wptr_addr:
            wptr_ptr = ctypes.cast(wptr_addr, ctypes.POINTER(ctypes.c_uint64))

        doorbell = hsa_signal_t(q.doorbell_signal)

        # Build PM4 packets to write to VRAM buffer
        pkt = build_nop()  # Start with NOP
        for i in range(8):
            pkt += build_write_mem(vram_addr + i * 4, 0xFACE0000 + i)

        # Submit
        ctypes.memmove(ring, pkt, len(pkt))
        new_wptr = len(pkt) // 4
        if wptr_ptr:
            wptr_ptr.contents.value = new_wptr
        hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(new_wptr))
        time.sleep(1.0)

        # Check results
        changed = 0
        for i in range(8):
            expected = 0xFACE0000 + i
            val = arr[i]
            mark = " ✓" if val == expected else ""
            log(f"    [{i}] = 0x{val:08X}{mark}")
            if val == expected:
                changed += 1

        log(f"\n  VRAM write result: {changed}/8")
        if changed == 8:
            log("  *** PM4 WRITES TO VRAM CONFIRMED! ***")

        if wptr_ptr:
            rptr_addr = wptr_addr + (0x80 - 0x38)  # rptr is at +0x80, wptr at +0x38
            rptr_ptr = ctypes.cast(rptr_addr, ctypes.POINTER(ctypes.c_uint64))
            log(f"  wptr={wptr_ptr.contents.value} rptr={rptr_ptr.contents.value}")

    log("\nDone. Exiting without queue destroy.")
    os._exit(0)

if __name__ == "__main__":
    main()
