#!/usr/bin/env python3
"""PM4 WRITE_DATA to memory — FIXED packet format.

Fixes from v3:
1. PM4 header count must be in bits[29:16], not bits[7:0]
   PACKET3(op, n) = (3<<30) | (n<<16) | (op<<8)
2. Properly update wptr memory before doorbell ring
3. Read wptr_addr from hook output file

CONFIRMED RESULT (pre-crash run):
  dst_sel=5 (memory-async): 4/4 writes WORKED (0xCAFE0001-4)
  dst_sel=1 (memory-sync): causes "Illegal opcode" — DO NOT USE
"""
import os, sys, struct, ctypes, ctypes.util, time

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
hsa.hsa_shut_down.restype = ctypes.c_int
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
    """Build PM4 PKT3 header: type=3 in [31:30], count=n in [29:16], opcode in [15:8]"""
    return (3 << 30) | ((n & 0x3FFF) << 16) | ((op & 0xFF) << 8)

def build_nop():
    """NOP packet: opcode=0x10, count=0 (1 padding dword follows)"""
    return struct.pack("<II", PACKET3(0x10, 0), 0)

def build_write_mem(gpu_va, value, dst_sel=5):
    """WRITE_DATA to memory.
    count=3 means 4 dwords follow header: control, addr_lo, addr_hi, data
    dst_sel: 5=memory-async (WORKS), 1=memory-sync (BROKEN on GFX11 compute)
    """
    header = PACKET3(0x37, 3)  # WRITE_DATA, 4 body dwords
    control = (dst_sel << 8) | (1 << 20)  # dst_sel + wr_confirm
    addr_lo = gpu_va & 0xFFFFFFFF
    addr_hi = (gpu_va >> 32) & 0xFFFFFFFF
    return struct.pack("<IIIII", header, control, addr_lo, addr_hi, value)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=== PM4 Memory Write Test v4 (FIXED headers) ===")

    # Verify header format
    h = PACKET3(0x37, 3)
    log(f"  WRITE_DATA header: 0x{h:08X} (expect 0xC0033700)")
    h2 = PACKET3(0x10, 0)
    log(f"  NOP header:        0x{h2:08X} (expect 0xC0001000)")

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
    log(f"Region: 0x{kernarg.handle:x}")

    # Allocate test buffer (GPU-visible, CPU-coherent)
    test_buf = ctypes.c_void_p(0)
    status = hsa.hsa_memory_allocate(kernarg, 4096, ctypes.byref(test_buf))
    if status != 0:
        log(f"Alloc failed: {status}")
        os._exit(1)
    buf_addr = test_buf.value
    log(f"Buffer: 0x{buf_addr:x}")

    # Fill with known pattern
    arr = (ctypes.c_uint32 * 1024).from_address(buf_addr)
    for i in range(16):
        arr[i] = 0xDEADBEEF

    # Create PM4 queue (hook patches type 2->0)
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        log(f"Queue create failed: {status}")
        os._exit(1)

    q = qp.contents
    ring = ctypes.cast(q.base_address, ctypes.c_void_p).value
    log(f"Queue: id={q.id} type={q.type} size={q.size} ring=0x{ring:x}")

    # Read wptr/rptr addresses from hook output
    wptr_addr = None
    rptr_addr = None
    try:
        with open('/tmp/pm4_queue_info', 'r') as f:
            for line in f:
                if line.startswith('wptr='):
                    wptr_addr = int(line.strip().split('=')[1], 16)
                elif line.startswith('rptr='):
                    rptr_addr = int(line.strip().split('=')[1], 16)
    except:
        pass

    wptr_ptr = None
    if wptr_addr:
        wptr_ptr = ctypes.cast(wptr_addr, ctypes.POINTER(ctypes.c_uint64))
        cur_wptr = wptr_ptr.contents.value
        log(f"wptr_addr=0x{wptr_addr:x} current_wptr={cur_wptr}")
    else:
        log("WARNING: No wptr_addr from hook, using doorbell-only approach")

    rptr_ptr = None
    if rptr_addr:
        rptr_ptr = ctypes.cast(rptr_addr, ctypes.POINTER(ctypes.c_uint64))
        cur_rptr = rptr_ptr.contents.value
        log(f"rptr_addr=0x{rptr_addr:x} current_rptr={cur_rptr}")

    doorbell = hsa_signal_t(q.doorbell_signal)

    # ==============================
    # TEST 1: NOP only — verify no illegal opcode
    # ==============================
    log("\n=== TEST 1: NOP packet ===")
    nop = build_nop()

    ctypes.memmove(ring, nop, len(nop))
    new_wptr = 2  # 2 dwords
    if wptr_ptr:
        wptr_ptr.contents.value = new_wptr

    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(new_wptr))
    time.sleep(0.5)
    log("  NOP dispatched")

    # ==============================
    # TEST 2: WRITE_DATA to memory (dst_sel=5 ONLY — dst_sel=1 crashes!)
    # ==============================
    log("\n=== TEST 2: WRITE_DATA dst_sel=5 (memory-async) ===")
    log(f"  Before:")
    for i in range(4):
        log(f"    [+{i*4}] = 0x{arr[i]:08X}")

    pkt = b''
    for i, val in enumerate([0xCAFE0001, 0xCAFE0002, 0xCAFE0003, 0xCAFE0004]):
        addr = buf_addr + i * 4
        pkt += build_write_mem(addr, val, dst_sel=5)

    log(f"  Packet size: {len(pkt)} bytes ({len(pkt)//4} dwords)")

    ring_offset = new_wptr * 4
    ctypes.memmove(ring + ring_offset, pkt, len(pkt))

    new_wptr2 = new_wptr + len(pkt) // 4
    if wptr_ptr:
        wptr_ptr.contents.value = new_wptr2

    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(new_wptr2))
    time.sleep(1.0)

    log("  After:")
    changed = 0
    for i in range(4):
        val = arr[i]
        expected = 0xCAFE0001 + i
        mark = " *** WRITTEN ***" if val == expected else ""
        log(f"    [+{i*4}] = 0x{val:08X}{mark}")
        if val == expected:
            changed += 1

    if changed > 0:
        log(f"\n*** PM4 MEMORY WRITE WORKS! {changed}/4 values written ***")
    else:
        log("  No writes landed")

    # ==============================
    # TEST 3: Bulk write (16 values) — stress test
    # ==============================
    log("\n=== TEST 3: Bulk write (16 values) ===")
    pkt3 = b''
    for i in range(16):
        addr = buf_addr + (i + 4) * 4  # offset +16 to +76
        pkt3 += build_write_mem(addr, 0xAAAA0000 + i, dst_sel=5)

    ring_offset3 = new_wptr2 * 4
    ctypes.memmove(ring + ring_offset3, pkt3, len(pkt3))
    new_wptr3 = new_wptr2 + len(pkt3) // 4
    if wptr_ptr:
        wptr_ptr.contents.value = new_wptr3
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(new_wptr3))
    time.sleep(1.0)

    bulk_ok = 0
    for i in range(16):
        expected = 0xAAAA0000 + i
        if arr[4 + i] == expected:
            bulk_ok += 1
    log(f"  Bulk: {bulk_ok}/16 values written")

    # Final state
    if wptr_ptr:
        log(f"\nFinal wptr={wptr_ptr.contents.value}")
    if rptr_ptr:
        log(f"Final rptr={rptr_ptr.contents.value}")

    log("\nDone. Exiting without queue destroy.")
    os._exit(0)

if __name__ == "__main__":
    main()
