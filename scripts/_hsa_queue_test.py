#!/usr/bin/env python3
"""Test HSA runtime queue creation + PM4 register writes.

Uses libhsa-runtime64 to properly create a compute queue,
then writes PM4 WRITE_DATA packets to test IC register access.
"""
import os, sys, struct, ctypes, ctypes.util, time, json
from pathlib import Path

RESULTS = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
DEBUGFS_REGS = "/sys/kernel/debug/dri/0/amdgpu_regs"

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

# Load HSA runtime
hsa = ctypes.CDLL("/opt/rocm-7.1.1/lib/libhsa-runtime64.so.1", use_errno=True)

# HSA types
HSA_STATUS_SUCCESS = 0
HSA_DEVICE_TYPE_GPU = 1
HSA_QUEUE_TYPE_SINGLE = 1

# HSA agent struct (opaque, 8 bytes)
class hsa_agent_t(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]

class hsa_queue_t(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("base_address", ctypes.POINTER(ctypes.c_void_p)),
        ("doorbell_signal", ctypes.c_uint64),  # hsa_signal_t
        ("size", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("id", ctypes.c_uint64),
    ]

class hsa_signal_t(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]

# Function prototypes
hsa.hsa_init.restype = ctypes.c_int
hsa.hsa_shut_down.restype = ctypes.c_int

AGENT_CB = ctypes.CFUNCTYPE(ctypes.c_int, hsa_agent_t, ctypes.c_void_p)
hsa.hsa_iterate_agents.argtypes = [AGENT_CB, ctypes.c_void_p]
hsa.hsa_iterate_agents.restype = ctypes.c_int

hsa.hsa_agent_get_info.argtypes = [hsa_agent_t, ctypes.c_uint32, ctypes.c_void_p]
hsa.hsa_agent_get_info.restype = ctypes.c_int

hsa.hsa_queue_create.argtypes = [
    hsa_agent_t,    # agent
    ctypes.c_uint32, # size
    ctypes.c_uint32, # type
    ctypes.c_void_p, # callback
    ctypes.c_void_p, # data
    ctypes.c_uint32, # private_segment_size
    ctypes.c_uint32, # group_segment_size
    ctypes.POINTER(ctypes.POINTER(hsa_queue_t))  # queue
]
hsa.hsa_queue_create.restype = ctypes.c_int

hsa.hsa_queue_destroy.argtypes = [ctypes.POINTER(hsa_queue_t)]
hsa.hsa_queue_destroy.restype = ctypes.c_int

hsa.hsa_queue_add_write_index_relaxed.argtypes = [ctypes.POINTER(hsa_queue_t), ctypes.c_uint64]
hsa.hsa_queue_add_write_index_relaxed.restype = ctypes.c_uint64

hsa.hsa_signal_store_relaxed.argtypes = [hsa_signal_t, ctypes.c_int64]

# HSA info attributes
HSA_AGENT_INFO_DEVICE = 17
HSA_AGENT_INFO_NAME = 0
HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14

def read_reg(dword_offset):
    byte_off = dword_offset * 4
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(byte_off)
            data = f.read(4)
        if len(data) == 4:
            return struct.unpack("<I", data)[0]
    except:
        pass
    return None

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def main():
    log("=== HSA Queue + PM4 Register Write Test ===")

    # Init HSA
    status = hsa.hsa_init()
    log(f"hsa_init: {status}")
    if status != HSA_STATUS_SUCCESS:
        log("HSA init failed!")
        return

    # Find GPU agent
    gpu_agent = hsa_agent_t(0)

    def find_gpu(agent, data):
        device_type = ctypes.c_uint32(0)
        hsa.hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type))
        if device_type.value == HSA_DEVICE_TYPE_GPU:
            name = ctypes.create_string_buffer(64)
            hsa.hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name)
            log(f"  Found GPU: {name.value.decode()}")
            gpu_agent.handle = agent.handle
            return 1  # stop iteration
        return 0  # continue

    cb = AGENT_CB(find_gpu)
    hsa.hsa_iterate_agents(cb, None)

    if gpu_agent.handle == 0:
        log("No GPU agent found!")
        hsa.hsa_shut_down()
        return

    # Get max queue size
    max_q = ctypes.c_uint32(0)
    hsa.hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_q))
    log(f"  Max queue size: {max_q.value}")

    # Create queue
    queue_ptr = ctypes.POINTER(hsa_queue_t)()
    q_size = min(1024, max_q.value)

    status = hsa.hsa_queue_create(
        gpu_agent,
        q_size,
        HSA_QUEUE_TYPE_SINGLE,
        None,  # callback
        None,  # data
        0xFFFFFFFF,  # private_segment_size (UINT32_MAX = use default)
        0xFFFFFFFF,  # group_segment_size
        ctypes.byref(queue_ptr)
    )

    if status != HSA_STATUS_SUCCESS:
        log(f"Queue create failed: status={status}")
        # Try with explicit sizes
        status = hsa.hsa_queue_create(
            gpu_agent, q_size, HSA_QUEUE_TYPE_SINGLE,
            None, None, 0, 0, ctypes.byref(queue_ptr)
        )
        if status != HSA_STATUS_SUCCESS:
            log(f"Queue create (v2) failed: status={status}")
            hsa.hsa_shut_down()
            return

    queue = queue_ptr.contents
    log(f"  Queue created! id={queue.id} size={queue.size}")
    ring_base = ctypes.cast(queue.base_address, ctypes.c_void_p).value
    log(f"  Ring base: 0x{ring_base:x}")
    log(f"  Doorbell signal: 0x{queue.doorbell_signal:x}")

    # Now we have a working queue. Build PM4 packets.
    # PM4 WRITE_DATA format:
    #   DWORD 0: PKT3 header (type=3, opcode=0x37, count)
    #   DWORD 1: control (dst_sel=0 for reg, wr_confirm=1, engine=ME)
    #   DWORD 2: register dword offset
    #   DWORD 3+: data

    def build_write_data(reg_offset, value):
        """Build PM4 WRITE_DATA packet targeting a register."""
        header = (3 << 30) | (0x37 << 8) | (2)  # PKT3, opcode=WRITE_DATA, count=3 (3 dwords after header)
        control = (0 << 8) | (1 << 20)  # dst_sel=0 (reg), wr_confirm=1
        return struct.pack("<IIII", header, control, reg_offset, value)

    def build_nop(count=1):
        """Build NOP packet."""
        header = (3 << 30) | (0x10 << 8) | ((count - 1) & 0x3FFF)
        return struct.pack("<I", header) + b'\x00' * (4 * count)

    results = {"queue_ok": True, "queue_id": queue.id, "tests": []}

    # Test 1: Write to scratch register (should work)
    log("\n--- Test 1: PM4 write to SCRATCH_REG0 (0x2040) ---")
    scratch_before = read_reg(0x2040)
    log(f"  Before: 0x{scratch_before:08X}" if scratch_before is not None else "  Before: READ_FAIL")

    # Write PM4 packet to ring buffer
    pkt = build_write_data(0x2040, 0xFACECAFE)
    nop = build_nop(4)  # padding

    # Get write index
    write_idx = hsa.hsa_queue_add_write_index_relaxed(queue_ptr, 1)
    log(f"  Write index: {write_idx}")

    # Calculate ring offset
    ring_offset = (write_idx % queue.size) * 64  # AQL packet size is 64 bytes

    # Write packet to ring
    ring_ptr = ctypes.cast(ring_base + ring_offset, ctypes.POINTER(ctypes.c_char))
    # AQL dispatch packet with PM4 embedded? No — HSA queues use AQL packets, not raw PM4.
    # We need a different approach for PM4.

    log("  NOTE: HSA queues use AQL format, not raw PM4.")
    log("  AQL packets launch kernels/barriers, can't directly write registers.")
    log("  PM4 queues require KFD CREATE_QUEUE with queue_type=COMPUTE_AQL or raw PM4.")

    # Alternative: use SDMA queue for register writes
    # Or: use vendor-specific AQL packet

    # Actually, let me check if we can create a PM4 queue via KFD now that HSA has initialized
    # HSA runtime does all the doorbell/wptr setup that we were missing

    log("\n--- Test 2: Direct ring buffer PM4 injection ---")
    log("  Attempting to write PM4 NOP directly to ring buffer...")

    # Write a NOP packet to the beginning of the ring
    nop_pkt = struct.pack("<II", (3 << 30) | (0x10 << 8) | 0, 0)  # NOP, 1 dword payload
    ctypes.memmove(ring_base, nop_pkt, len(nop_pkt))
    log(f"  NOP written to ring at 0x{ring_base:x}")

    # Now try WRITE_DATA to scratch via ring
    wd_pkt = build_write_data(0x2040, 0xFACECAFE)
    ctypes.memmove(ring_base + 64, wd_pkt, len(wd_pkt))

    # Ring doorbell to signal new work
    doorbell = hsa_signal_t(queue.doorbell_signal)
    hsa.hsa_signal_store_relaxed(doorbell, ctypes.c_int64(2))

    time.sleep(0.5)

    scratch_after = read_reg(0x2040)
    log(f"  After doorbell: 0x{scratch_after:08X}" if scratch_after is not None else "  After: READ_FAIL")

    changed = scratch_before != scratch_after if scratch_before is not None and scratch_after is not None else None
    log(f"  Changed: {changed}")

    results["tests"].append({
        "name": "scratch_write",
        "before": f"0x{scratch_before:08X}" if scratch_before else None,
        "after": f"0x{scratch_after:08X}" if scratch_after else None,
        "changed": changed,
    })

    # Test 3: Read IC registers for baseline
    log("\n--- Test 3: IC register baseline ---")
    ic_regs = {
        'ME_IC_BASE_LO': 0x5844,
        'ME_IC_BASE_HI': 0x5845,
        'ME_IC_OP_CNTL': 0x5847,
        'CPC_IC_BASE_LO': 0x584C,
        'CPC_IC_BASE_HI': 0x584D,
        'CPC_IC_OP_CNTL': 0x297A,
    }

    baseline = {}
    for name, off in ic_regs.items():
        val = read_reg(off)
        vstr = f"0x{val:08X}" if val is not None else "FAIL"
        log(f"  {name} (0x{off:04X}): {vstr}")
        baseline[name] = vstr

    results["ic_baseline"] = baseline

    # Cleanup
    hsa.hsa_queue_destroy(queue_ptr)
    hsa.hsa_shut_down()

    # Save results
    with open(RESULTS / "z2349b_hsa_queue.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"\nResults saved to {RESULTS / 'z2349b_hsa_queue.json'}")
    log("Done.")

if __name__ == "__main__":
    main()
