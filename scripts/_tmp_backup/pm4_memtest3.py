#!/usr/bin/env python3
"""PM4 WRITE_DATA to memory — simple version.

Uses same doorbell approach as the working register test
(hsa_signal_store_relaxed), but targets memory instead of registers.
Also tests NOP packet (no privilege issue) to confirm dispatch.
"""
import os, sys, struct, ctypes, ctypes.util, time

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

DEBUGFS_REGS = "/sys/kernel/debug/dri/0/amdgpu_regs"

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
hsa.hsa_queue_destroy.argtypes = [ctypes.POINTER(hsa_queue_t)]
hsa.hsa_queue_destroy.restype = ctypes.c_int
hsa.hsa_signal_store_relaxed.argtypes = [hsa_signal_t, ctypes.c_int64]
hsa.hsa_agent_iterate_regions.argtypes = [hsa_agent_t, REGION_CB, ctypes.c_void_p]
hsa.hsa_agent_iterate_regions.restype = ctypes.c_int
hsa.hsa_region_get_info.argtypes = [hsa_region_t, ctypes.c_uint32, ctypes.c_void_p]
hsa.hsa_region_get_info.restype = ctypes.c_int
hsa.hsa_memory_allocate.argtypes = [hsa_region_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
hsa.hsa_memory_allocate.restype = ctypes.c_int

def read_reg(dword_offset):
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(dword_offset * 4)
            data = f.read(4)
        if len(data) == 4:
            return struct.unpack("<I", data)[0]
    except:
        pass
    return None

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=== PM4 Memory Write Test v3 ===")

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

    # Find kernarg region (fine-grained, CPU+GPU coherent)
    kernarg = hsa_region_t(0)
    coarse = hsa_region_t(0)
    def find_regions(r, d):
        seg = ctypes.c_uint32(0)
        hsa.hsa_region_get_info(r, 0, ctypes.byref(seg))
        if seg.value == 0:  # GLOBAL
            flags = ctypes.c_uint32(0)
            hsa.hsa_region_get_info(r, 1, ctypes.byref(flags))
            if flags.value & 0x1:
                kernarg.handle = r.handle
            elif flags.value & 0x4:
                coarse.handle = r.handle
        return 0
    hsa.hsa_agent_iterate_regions(gpu, REGION_CB(find_regions), None)

    # Use kernarg region (fine-grained = CPU+GPU coherent)
    region = kernarg if kernarg.handle else coarse
    log(f"Region: 0x{region.handle:x} ({'kernarg' if region.handle == kernarg.handle else 'coarse'})")

    # Allocate test buffer
    test_buf = ctypes.c_void_p(0)
    status = hsa.hsa_memory_allocate(region, 4096, ctypes.byref(test_buf))
    if status != 0:
        log(f"Alloc failed: {status}")
        os._exit(1)
    buf_addr = test_buf.value
    log(f"Buffer: 0x{buf_addr:x}")

    # Fill with known pattern
    arr = (ctypes.c_uint32 * 1024).from_address(buf_addr)
    for i in range(1024):
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

    # ==============================
    # TEST 1: NOP only (should NOT trigger privilege violation)
    # ==============================
    log("\n=== TEST 1: NOP only ===")
    nop = struct.pack("<II", (3 << 30) | (0x10 << 8) | 0, 0)  # NOP, 1 dword payload
    # Pad to 8 dwords (32 bytes)
    pkt = nop + b'\x00' * (32 - len(nop))

    ctypes.memmove(ring, pkt, len(pkt))
    doorbell = hsa_signal_t(q.doorbell_signal)
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(len(pkt) // 4))
    time.sleep(0.5)
    log("  NOP submitted (check dmesg for errors)")

    # ==============================
    # TEST 2: WRITE_DATA to memory (dst_sel=5)
    # ==============================
    log("\n=== TEST 2: WRITE_DATA to memory ===")
    log(f"  Before: [+0]=0x{arr[0]:08X} [+4]=0x{arr[1]:08X} [+8]=0x{arr[2]:08X}")

    # Build packets
    pkt2 = b''
    # WRITE_DATA: dst_sel=5 (memory-async), wr_confirm=1
    for i, val in enumerate([0xCAFE0001, 0xCAFE0002, 0xCAFE0003, 0xCAFE0004]):
        addr = buf_addr + i * 4
        header = (3 << 30) | (0x37 << 8) | 3  # count=3
        control = (5 << 8) | (1 << 20)  # dst_sel=5, wr_confirm
        pkt2 += struct.pack("<IIIII", header, control, addr & 0xFFFFFFFF, addr >> 32, val)

    # Pad
    while len(pkt2) % 32 != 0:
        pkt2 += struct.pack("<II", (3 << 30) | (0x10 << 8) | 0, 0)

    # Write at offset past first test
    offset = 64  # Skip first test's packets
    ctypes.memmove(ring + offset, pkt2, len(pkt2))
    ndw = (offset + len(pkt2)) // 4
    log(f"  Submitting {len(pkt2)} bytes at ring+{offset}, doorbell={ndw}")
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(ndw))

    time.sleep(1.0)

    log(f"  After:  [+0]=0x{arr[0]:08X} [+4]=0x{arr[1]:08X} [+8]=0x{arr[2]:08X} [+12]=0x{arr[3]:08X}")

    matches = sum(1 for i, v in enumerate([0xCAFE0001, 0xCAFE0002, 0xCAFE0003, 0xCAFE0004])
                  if arr[i] == v)
    if matches > 0:
        log(f"\n*** PM4 MEMORY WRITE WORKS! {matches}/4 values written ***")
    else:
        log("\n  No values changed via memory write")

    # ==============================
    # TEST 3: WRITE_DATA to memory with dst_sel=1 (sync)
    # ==============================
    log("\n=== TEST 3: WRITE_DATA memory-sync (dst_sel=1) ===")
    pkt3 = b''
    for i, val in enumerate([0xBEEF0001, 0xBEEF0002]):
        addr = buf_addr + (i + 4) * 4
        header = (3 << 30) | (0x37 << 8) | 3
        control = (1 << 8) | (1 << 20)  # dst_sel=1 (memory-sync), wr_confirm
        pkt3 += struct.pack("<IIIII", header, control, addr & 0xFFFFFFFF, addr >> 32, val)

    while len(pkt3) % 32 != 0:
        pkt3 += struct.pack("<II", (3 << 30) | (0x10 << 8) | 0, 0)

    offset2 = offset + len(pkt2)
    ctypes.memmove(ring + offset2, pkt3, len(pkt3))
    ndw2 = (offset2 + len(pkt3)) // 4
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(ndw2))
    time.sleep(1.0)

    log(f"  After: [+16]=0x{arr[4]:08X} [+20]=0x{arr[5]:08X}")
    m3 = (arr[4] == 0xBEEF0001) + (arr[5] == 0xBEEF0002)
    if m3 > 0:
        log(f"  *** Sync memory write: {m3}/2 ***")

    log("\nDone. Exiting without queue destroy.")
    os._exit(0)

if __name__ == "__main__":
    main()
