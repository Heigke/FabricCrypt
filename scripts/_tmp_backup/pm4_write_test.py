#!/usr/bin/env python3
"""PM4 WRITE_DATA test via HSA-created PM4 queue.

Uses LD_PRELOAD hook to convert HSA's AQL queue to PM4.
Then writes PM4 packets to ring buffer and rings doorbell.
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
hsa.hsa_signal_store_relaxed.argtypes = [hsa_signal_t, ctypes.c_int64]

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
    log("=== PM4 WRITE_DATA via Hijacked HSA Queue ===")

    # Init HSA
    status = hsa.hsa_init()
    if status != 0:
        log(f"hsa_init failed: {status}")
        return
    log("HSA initialized")

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

    # Create queue (hook will convert to PM4)
    qp = ctypes.POINTER(hsa_queue_t)()
    status = hsa.hsa_queue_create(gpu, 1024, 1, None, None, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
    if status != 0:
        log(f"Queue create failed: {status}")
        hsa.hsa_shut_down()
        return

    queue = qp.contents
    ring_base = ctypes.cast(queue.base_address, ctypes.c_void_p).value
    doorbell_handle = queue.doorbell_signal
    log(f"Queue created: id={queue.id} size={queue.size} type={queue.type}")
    log(f"  ring_base=0x{ring_base:x}")
    log(f"  doorbell=0x{doorbell_handle:x}")

    # PM4 packet builders
    def build_write_data(reg_offset, value):
        """PM4 WRITE_DATA: write value to register."""
        header = (3 << 30) | (0x37 << 8) | 2  # PKT3, WRITE_DATA, count=2 (2 dwords after header+control)
        control = (0 << 8) | (1 << 20)         # dst_sel=0 (register), wr_confirm=1
        return struct.pack("<IIII", header, control, reg_offset, value)

    def build_nop(n=1):
        """PM4 NOP with n payload dwords."""
        header = (3 << 30) | (0x10 << 8) | ((n-1) & 0x3FFF)
        return struct.pack("<I", header) + b'\x00' * (4 * n)

    # Targets to test
    targets = [
        # Safe scratch registers first
        ("SCRATCH_REG0", 0x2040, 0xFACE0001),
        ("SCRATCH_REG1", 0x2041, 0xFACE0002),
        ("SCRATCH_REG2", 0x2042, 0xFACE0003),
        ("SCRATCH_REG3", 0x2043, 0xFACE0004),
        # IC registers
        ("CP_ME_IC_BASE_LO", 0x5844, None),     # read-only test
        ("CP_ME_IC_BASE_HI", 0x5845, None),     # read-only test
        ("CP_ME_IC_OP_CNTL", 0x5847, None),     # read-only test
        ("CP_MES_IC_BASE_LO", 0x2820, None),    # read-only test
        ("CP_MES_IC_BASE_HI", 0x2821, None),    # read-only test
    ]

    # Read baselines
    log("\n--- Register Baselines ---")
    baselines = {}
    for name, reg, _ in targets:
        val = read_reg(reg)
        vstr = f"0x{val:08X}" if val is not None else "FAIL"
        log(f"  {name} (0x{reg:04X}): {vstr}")
        baselines[name] = val

    # Write PM4 packets to ring buffer
    log("\n--- Submitting PM4 WRITE_DATA ---")

    # Build packet stream: 4 WRITE_DATA + NOP padding
    pkt_stream = b''
    for name, reg, val in targets[:4]:  # Only scratch registers for write test
        pkt_stream += build_write_data(reg, val)
        log(f"  Queued WRITE_DATA: {name} <- 0x{val:08X}")

    # Pad to 16-dword alignment with NOPs
    while len(pkt_stream) % 64 != 0:
        pkt_stream += build_nop(1)

    # Write to ring buffer directly via memmove
    log(f"  Writing {len(pkt_stream)} bytes to ring at 0x{ring_base:x}")
    ctypes.memmove(ring_base, pkt_stream, len(pkt_stream))

    # Ring doorbell
    # For PM4 queues, doorbell value = number of dwords written
    ndwords = len(pkt_stream) // 4
    log(f"  Ringing doorbell with value {ndwords} (0x{ndwords:x})")

    # Method 1: via hsa_signal_store_relaxed
    doorbell = hsa_signal_t(doorbell_handle)
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(ndwords))

    # Also try direct wptr update
    # The queue's wptr should be updated to tell HW how many dwords are pending
    # For HSA queues, this is at queue.base_address - some offset
    # But for PM4, we need byte offset in ring / ring_item_size

    time.sleep(0.5)

    # Read back
    log("\n--- Post-Write Readback ---")
    results = []
    for name, reg, expected in targets:
        after = read_reg(reg)
        astr = f"0x{after:08X}" if after is not None else "FAIL"
        before = baselines.get(name)
        changed = before != after if before is not None and after is not None else None

        if expected is not None:
            matched = after == expected if after is not None else False
            log(f"  {name}: {astr} (expected 0x{expected:08X}) {'MATCH!' if matched else 'MISMATCH'} changed={changed}")
            results.append((name, matched, changed))
        else:
            log(f"  {name}: {astr} changed={changed}")
            results.append((name, None, changed))

    # Try IC register writes via PM4
    log("\n--- PM4 IC Register Write Test ---")
    ic_targets = [
        ("CP_ME_IC_BASE_LO", 0x5844, 0x00040000),  # Test write to IC_BASE_LO
        ("CP_ME_IC_BASE_HI", 0x5845, 0x00000001),   # Test write to IC_BASE_HI
    ]

    pkt2 = b''
    for name, reg, val in ic_targets:
        pkt2 += build_write_data(reg, val)
        log(f"  Queued: {name} <- 0x{val:08X}")

    while len(pkt2) % 64 != 0:
        pkt2 += build_nop(1)

    # Write at offset in ring (after first batch)
    offset = len(pkt_stream)
    ctypes.memmove(ring_base + offset, pkt2, len(pkt2))

    ndwords2 = (offset + len(pkt2)) // 4
    log(f"  Ringing doorbell with {ndwords2}")
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(ndwords2))

    time.sleep(0.5)

    for name, reg, expected in ic_targets:
        after = read_reg(reg)
        before = baselines.get(name)
        astr = f"0x{after:08X}" if after is not None else "FAIL"
        changed = before != after if before is not None and after is not None else None
        matched = after == expected if after is not None else False
        log(f"  {name}: {astr} {'WRITE OK!' if matched else ''} changed={changed}")

    # Cleanup
    hsa.hsa_queue_destroy(qp)
    hsa.hsa_shut_down()
    log("\nDone.")

if __name__ == "__main__":
    main()
