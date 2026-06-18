#!/usr/bin/env python3
"""PM4 WRITE_DATA to MEMORY test via hijacked HSA PM4 queue.

Uses dst_sel=5 (memory-async) to write to GPU-visible memory
instead of dst_sel=0 (register) which triggers privilege violation.
"""
import os, sys, struct, ctypes, ctypes.util, time

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

DEBUGFS_REGS = "/sys/kernel/debug/dri/0/amdgpu_regs"

hsa = ctypes.CDLL('/opt/rocm-7.1.1/lib/libhsa-runtime64.so.1', use_errno=True)
libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

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

AGENT_CB = ctypes.CFUNCTYPE(ctypes.c_int, hsa_agent_t, ctypes.c_void_p)
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

# Memory allocation via HSA
class hsa_region_t(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint64)]

REGION_CB = ctypes.CFUNCTYPE(ctypes.c_int, hsa_region_t, ctypes.c_void_p)
hsa.hsa_agent_iterate_regions.argtypes = [hsa_agent_t, REGION_CB, ctypes.c_void_p]
hsa.hsa_agent_iterate_regions.restype = ctypes.c_int
hsa.hsa_region_get_info.argtypes = [hsa_region_t, ctypes.c_uint32, ctypes.c_void_p]
hsa.hsa_region_get_info.restype = ctypes.c_int
hsa.hsa_memory_allocate.argtypes = [hsa_region_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
hsa.hsa_memory_allocate.restype = ctypes.c_int
hsa.hsa_memory_free.argtypes = [ctypes.c_void_p]
hsa.hsa_memory_free.restype = ctypes.c_int

HSA_REGION_INFO_SEGMENT = 0
HSA_REGION_SEGMENT_GLOBAL = 0
HSA_REGION_INFO_GLOBAL_FLAGS = 1

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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

def main():
    log("=== PM4 WRITE_DATA to MEMORY Test ===")

    status = hsa.hsa_init()
    if status != 0:
        log(f"hsa_init failed: {status}")
        return
    log("HSA initialized")

    gpu = hsa_agent_t(0)
    def find_gpu(a, d):
        dt = ctypes.c_uint32(0)
        hsa.hsa_agent_get_info(a, 17, ctypes.byref(dt))
        if dt.value == 1:
            gpu.handle = a.handle
            return 1
        return 0
    hsa.hsa_iterate_agents(AGENT_CB(find_gpu), None)

    # Find GPU-visible memory region for our test buffer
    gpu_region = hsa_region_t(0)
    def find_region(region, data):
        segment = ctypes.c_uint32(0)
        hsa.hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, ctypes.byref(segment))
        if segment.value == HSA_REGION_SEGMENT_GLOBAL:
            flags = ctypes.c_uint32(0)
            hsa.hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, ctypes.byref(flags))
            # flags bit 0: KERNARG, bit 1: FINE_GRAINED, bit 2: COARSE_GRAINED
            if flags.value & 0x6:  # fine or coarse grained
                gpu_region.handle = region.handle
                return 1
        return 0
    hsa.hsa_agent_iterate_regions(gpu, REGION_CB(find_region), None)
    log(f"GPU region: 0x{gpu_region.handle:x}")

    # Allocate test buffer (4KB)
    test_buf = ctypes.c_void_p(0)
    status = hsa.hsa_memory_allocate(gpu_region, 4096, ctypes.byref(test_buf))
    if status != 0:
        log(f"Memory alloc failed: {status}")
        hsa.hsa_shut_down()
        return
    buf_addr = test_buf.value
    log(f"Test buffer: 0x{buf_addr:x}")

    # Initialize buffer with known pattern
    pattern = struct.pack("<I", 0xDEADBEEF) * 1024
    ctypes.memmove(buf_addr, pattern, 4096)
    # Verify
    readback = struct.unpack("<I", ctypes.string_at(buf_addr, 4))[0]
    log(f"  Initial: 0x{readback:08X}")

    # Create PM4 queue (hook converts to type=0)
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        log(f"Queue create failed: {status}")
        hsa.hsa_shut_down()
        return

    queue = qp.contents
    ring_base = ctypes.cast(queue.base_address, ctypes.c_void_p).value
    log(f"PM4 Queue: id={queue.id} size={queue.size} type={queue.type}")
    log(f"  ring=0x{ring_base:x}")

    # Build PM4 WRITE_DATA to memory
    # dst_sel=5 (memory-async), wr_confirm=1, engine_sel=0 (ME)
    def build_write_mem(gpu_va, value):
        """PM4 WRITE_DATA targeting a GPU virtual address."""
        header = (3 << 30) | (0x37 << 8) | 3  # PKT3, WRITE_DATA, count=3 (3 dwords payload)
        # control: dst_sel=5 (bits 11:8), wr_confirm=1 (bit 20)
        control = (5 << 8) | (1 << 20)
        addr_lo = gpu_va & 0xFFFFFFFF
        addr_hi = (gpu_va >> 32) & 0xFFFFFFFF
        return struct.pack("<IIIII", header, control, addr_lo, addr_hi, value)

    def build_nop(n=1):
        header = (3 << 30) | (0x10 << 8) | ((n-1) & 0x3FFF)
        return struct.pack("<I", header) + b'\x00' * (4 * n)

    # Test: write 4 different values to 4 locations in the test buffer
    test_values = [
        (buf_addr + 0,   0xCAFE0001),
        (buf_addr + 4,   0xCAFE0002),
        (buf_addr + 8,   0xCAFE0003),
        (buf_addr + 12,  0xCAFE0004),
    ]

    pkt_stream = b''
    for addr, val in test_values:
        pkt_stream += build_write_mem(addr, val)
        log(f"  Queued: [0x{addr:x}] <- 0x{val:08X}")

    # Pad to alignment
    while len(pkt_stream) % 64 != 0:
        pkt_stream += build_nop(1)

    log(f"  Packet stream: {len(pkt_stream)} bytes ({len(pkt_stream)//4} dwords)")

    # Write to ring
    ctypes.memmove(ring_base, pkt_stream, len(pkt_stream))

    # Ring doorbell
    doorbell = hsa_signal_t(queue.doorbell_signal)
    ndwords = len(pkt_stream) // 4
    log(f"  Ringing doorbell ({ndwords} dwords)")
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(ndwords))

    time.sleep(1.0)

    # Read back from buffer
    log("\n--- Readback ---")
    for addr, expected in test_values:
        offset = addr - buf_addr
        actual = struct.unpack("<I", ctypes.string_at(addr, 4))[0]
        match = actual == expected
        log(f"  [+0x{offset:x}]: 0x{actual:08X} (expected 0x{expected:08X}) {'MATCH!' if match else 'no change'}")

    # Also read IC registers to see if anything changed
    log("\n--- IC Register Check ---")
    for name, reg in [
        ("MES_IC_BASE_LO", 0x2820), ("MES_IC_BASE_HI", 0x2821),
        ("ME_IC_BASE_LO", 0x5844), ("ME_IC_BASE_HI", 0x5845),
        ("ME_IC_OP_CNTL", 0x5847),
    ]:
        val = read_reg(reg)
        vstr = f"0x{val:08X}" if val is not None else "FAIL"
        log(f"  {name}: {vstr}")

    # Cleanup — don't destroy queue (causes GPU reset with PM4 type)
    # Just shut down HSA
    log("\nSkipping queue destroy to avoid GPU reset")
    hsa.hsa_memory_free(test_buf)
    hsa.hsa_shut_down()
    log("Done.")

if __name__ == "__main__":
    main()
