#!/usr/bin/env python3
"""PM4 WRITE_DATA to MEMORY — with proper doorbell ringing.

Strategy:
1. HSA creates PM4 queue (via hook)
2. Hook saves doorbell info to /tmp/pm4_doorbell_info
3. We mmap the doorbell page from KFD fd
4. Write PM4 packets to ring, update wptr, ring doorbell
"""
import os, sys, struct, ctypes, ctypes.util, time, mmap

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

# Memory
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

def find_fd(target):
    for entry in os.listdir('/proc/self/fd'):
        try:
            link = os.readlink(f'/proc/self/fd/{entry}')
            if target in link:
                return int(entry)
        except:
            pass
    return None

def main():
    log("=== PM4 Memory Write v2 — Proper Doorbell ===")

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

    # Find GPU memory region
    gpu_region = hsa_region_t(0)
    kernarg_region = hsa_region_t(0)
    def find_region(region, data):
        segment = ctypes.c_uint32(0)
        hsa.hsa_region_get_info(region, 0, ctypes.byref(segment))
        if segment.value == 0:  # GLOBAL
            flags = ctypes.c_uint32(0)
            hsa.hsa_region_get_info(region, 1, ctypes.byref(flags))
            if flags.value & 0x4:  # COARSE_GRAINED
                gpu_region.handle = region.handle
            if flags.value & 0x1:  # KERNARG
                kernarg_region.handle = region.handle
        return 0
    hsa.hsa_agent_iterate_regions(gpu, REGION_CB(find_region), None)

    # Allocate test buffer in kernarg (fine-grained, CPU+GPU accessible)
    region = kernarg_region if kernarg_region.handle else gpu_region
    test_buf = ctypes.c_void_p(0)
    status = hsa.hsa_memory_allocate(region, 4096, ctypes.byref(test_buf))
    if status != 0:
        log(f"Memory alloc failed: {status}")
        hsa.hsa_shut_down()
        return
    buf_addr = test_buf.value
    log(f"Test buffer: 0x{buf_addr:x}")

    # Initialize with known pattern
    for i in range(1024):
        struct.pack_into("<I", (ctypes.c_char * 4096).from_address(buf_addr), i*4, 0xDEADBEEF)
    log(f"  Initialized with 0xDEADBEEF")

    # Create PM4 queue (hook converts to type=0)
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        log(f"Queue create failed: {status}")
        hsa.hsa_shut_down()
        return

    queue = qp.contents
    ring_base = ctypes.cast(queue.base_address, ctypes.c_void_p).value
    ring_size = queue.size  # in packets (AQL) but for PM4 this is ring size in dwords?
    log(f"PM4 Queue: id={queue.id} size={queue.size} type={queue.type}")
    log(f"  ring=0x{ring_base:x}")

    # Read doorbell info from hook
    time.sleep(0.1)
    try:
        with open('/tmp/pm4_doorbell_info', 'r') as f:
            parts = f.read().strip().split()
            hook_kfd_fd = int(parts[0])
            db_mmap_off = int(parts[1])
            db_offset = int(parts[2])
        log(f"  Doorbell: kfd_fd={hook_kfd_fd} mmap_off=0x{db_mmap_off:x} offset={db_offset}")
    except Exception as e:
        log(f"  Doorbell info not found: {e}")
        log("  Falling back to /proc/self/maps search")
        db_mmap_off = 0
        db_offset = 8  # Default for second queue

    # Find the doorbell page in our address space via /proc/self/maps
    kfd_fd = find_fd('/dev/kfd')
    drm_fd = find_fd('renderD128')
    log(f"  kfd_fd={kfd_fd} drm_fd={drm_fd}")

    # Scan maps for doorbell page
    doorbell_page = None
    with open('/proc/self/maps', 'r') as f:
        for line in f:
            # Doorbell pages are typically small (4KB) and mapped from the render node
            # with offset matching the doorbell mmap_off
            parts_line = line.strip().split()
            if len(parts_line) >= 6:
                addr_range = parts_line[0]
                perms = parts_line[1]
                offset = parts_line[2]
                dev = parts_line[3]
                inode = parts_line[4]
                name = parts_line[5] if len(parts_line) > 5 else ""
                if 'kfd' in name or (db_mmap_off and offset == f"{db_mmap_off:016x}"[:len(offset)]):
                    log(f"  Found map: {line.strip()}")

    # Alternative: mmap doorbell page ourselves from KFD fd
    # The doorbell page offset for KFD mmap is typically at the high range
    # Let's try to mmap it
    if db_mmap_off != 0 and kfd_fd is not None:
        try:
            db_mm = mmap.mmap(kfd_fd, 4096, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE,
                              offset=db_mmap_off)
            doorbell_page = ctypes.addressof(ctypes.c_char.from_buffer(db_mm, 0))
            log(f"  Doorbell page mmap'd at 0x{doorbell_page:x}")
        except Exception as e:
            log(f"  Doorbell mmap failed: {e}")

    # Build PM4 NOP packet first (safest test)
    def build_nop(n=1):
        header = (3 << 30) | (0x10 << 8) | ((n-1) & 0x3FFF)
        return struct.pack("<I", header) + b'\x00' * (4 * n)

    # Build PM4 WRITE_DATA to memory (dst_sel=5, memory-async)
    def build_write_mem(gpu_va, value):
        header = (3 << 30) | (0x37 << 8) | 3  # count=3 (4 dwords follow header)
        control = (5 << 8) | (1 << 20)  # dst_sel=5 (mem-async), wr_confirm=1
        addr_lo = gpu_va & 0xFFFFFFFF
        addr_hi = (gpu_va >> 32) & 0xFFFFFFFF
        return struct.pack("<IIIII", header, control, addr_lo, addr_hi, value)

    # Build packet stream: write 4 values to test buffer
    test_values = [
        (buf_addr + 0,   0xCAFE0001),
        (buf_addr + 4,   0xCAFE0002),
        (buf_addr + 8,   0xCAFE0003),
        (buf_addr + 12,  0xCAFE0004),
    ]

    pkt_stream = b''
    for addr, val in test_values:
        pkt_stream += build_write_mem(addr, val)

    # Pad with NOPs to 64-byte alignment
    while len(pkt_stream) % 16 != 0:
        pkt_stream += build_nop(1)

    ndwords = len(pkt_stream) // 4
    log(f"\nPacket stream: {len(pkt_stream)} bytes, {ndwords} dwords")

    # Read before values
    log("Before:")
    for addr, _ in test_values:
        val = struct.unpack("<I", ctypes.string_at(addr, 4))[0]
        log(f"  [0x{addr:x}]: 0x{val:08X}")

    # Write packets to ring buffer
    ctypes.memmove(ring_base, pkt_stream, len(pkt_stream))
    log("Packets written to ring")

    # Find wptr location — from the trace, HSA puts wptr at ring_alloc_base + 0x38
    # The ring is allocated as a separate buffer from the wptr buffer
    # Let's find the wptr by scanning nearby memory
    # Actually for PM4 queues, we need to update wptr via the doorbell

    # Ring doorbell
    if doorbell_page:
        # For PM4 queues, doorbell value = wptr in bytes
        wptr_bytes = len(pkt_stream)
        db_addr = doorbell_page + db_offset
        log(f"Writing doorbell at 0x{db_addr:x} + offset {db_offset} = wptr_bytes={wptr_bytes}")

        # Write 64-bit doorbell value
        ctypes.c_uint64.from_address(db_addr).value = wptr_bytes
        log("Doorbell written!")
    else:
        log("WARNING: No doorbell page — trying hsa_signal_store_relaxed as fallback")
        doorbell = hsa_signal_t(queue.doorbell_signal)
        hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(ndwords))

    time.sleep(1.0)

    # Read after values
    log("\nAfter:")
    any_changed = False
    for addr, expected in test_values:
        actual = struct.unpack("<I", ctypes.string_at(addr, 4))[0]
        match = actual == expected
        if match:
            any_changed = True
        log(f"  [0x{addr:x}]: 0x{actual:08X} {'MATCH!' if match else '(unchanged)'}")

    if any_changed:
        log("\n*** PM4 WRITE_DATA TO MEMORY WORKS! ***")
    else:
        log("\nNo changes detected. Checking dmesg for errors...")

    # Check IC registers
    log("\n--- IC Registers ---")
    for name, reg in [
        ("MES_IC_BASE_LO", 0x2820), ("MES_IC_BASE_HI", 0x2821),
        ("ME_IC_BASE_LO", 0x5844), ("ME_IC_BASE_HI", 0x5845),
        ("ME_IC_OP_CNTL", 0x5847),
    ]:
        val = read_reg(reg)
        vstr = f"0x{val:08X}" if val is not None else "FAIL"
        log(f"  {name}: {vstr}")

    # Don't destroy queue or free memory — let process exit handle cleanup
    log("\nExiting (no queue destroy)")
    # hsa.hsa_shut_down() — skip to avoid cleanup issues
    os._exit(0)

if __name__ == "__main__":
    main()
