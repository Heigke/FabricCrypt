#!/usr/bin/env python3
"""Hybrid approach: use HSA runtime for KFD init, then create PM4 queue.

Key fix: wptr/rptr must be GPU-mapped addresses (not host pointers).
HSA does ALLOC+MAP for wptr/rptr buffer, then passes GPU VAs.
"""
import os, sys, struct, ctypes, ctypes.util, mmap, time, json
from pathlib import Path

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

RESULTS = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
DEBUGFS_REGS = "/sys/kernel/debug/dri/0/amdgpu_regs"

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
hsa = ctypes.CDLL("/opt/rocm-7.1.1/lib/libhsa-runtime64.so.1", use_errno=True)

K = ord('K')
def _IOC(d, t, nr, sz): return (d << 30) | (t << 8) | nr | (sz << 16)
def _IOR(t, nr, sz):    return _IOC(2, t, nr, sz)
def _IOW(t, nr, sz):    return _IOC(1, t, nr, sz)
def _IOWR(t, nr, sz):   return _IOC(3, t, nr, sz)

IOC_VER        = _IOR(K, 0x01, 8)
IOC_CREATE_Q   = _IOWR(K, 0x02, 96)
IOC_DESTROY_Q  = _IOWR(K, 0x03, 8)
IOC_APER_OLD   = _IOR(K, 0x06, 400)
IOC_ACQ_VM     = _IOW(K, 0x15, 8)
IOC_ALLOC      = _IOWR(K, 0x16, 40)
IOC_MAP        = _IOWR(K, 0x18, 24)

GTT = (1 << 1)
VRAM = (1 << 0)
WR  = (1 << 31)
EX  = (1 << 30)
UC  = (1 << 25)

# HSA types
class hsa_agent_t(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]
class hsa_queue_t(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32), ("features", ctypes.c_uint32),
        ("base_address", ctypes.POINTER(ctypes.c_void_p)),
        ("doorbell_signal", ctypes.c_uint64),
        ("size", ctypes.c_uint32), ("reserved1", ctypes.c_uint32),
        ("id", ctypes.c_uint64),
    ]

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

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

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
    log("=== Hybrid HSA + KFD PM4 Queue Test v2 ===")
    log("Fix: wptr/rptr as GPU-mapped VA, matching HSA's struct layout")

    # Step 1: HSA init
    status = hsa.hsa_init()
    log(f"hsa_init: {status}")
    if status != 0:
        log("HSA init failed!")
        return

    gpu_agent = hsa_agent_t(0)
    def find_gpu(agent, data):
        device_type = ctypes.c_uint32(0)
        hsa.hsa_agent_get_info(agent, 17, ctypes.byref(device_type))
        if device_type.value == 1:
            gpu_agent.handle = agent.handle
            return 1
        return 0
    cb = AGENT_CB(find_gpu)
    hsa.hsa_iterate_agents(cb, None)

    kfd = find_fd('/dev/kfd')
    drm = find_fd('renderD128')
    log(f"kfd={kfd} drm={drm}")

    # Get apertures
    ab = ctypes.create_string_buffer(400)
    libc.ioctl(kfd, IOC_APER_OLD, ab)
    vals = struct.unpack_from('<QQQQQQII', ab.raw, 0)
    gpu_id = vals[6]
    gpuvm_b = vals[4]
    scratch_b = vals[2]
    log(f"gpu_id=0x{gpu_id:x}")

    # Alloc helpers
    va_cur = [gpuvm_b + 0x80000000]

    def alloc_gpu(sz, flags, nm):
        va = va_cur[0]
        b = ctypes.create_string_buffer(40)
        struct.pack_into('<QQQQII', b, 0, va, sz, 0, 0, gpu_id, flags)
        r = libc.ioctl(kfd, IOC_ALLOC, b)
        if r != 0:
            e = ctypes.get_errno()
            log(f"  {nm} ALLOC FAIL: {e} ({os.strerror(e)})")
            return None
        h = struct.unpack_from('<QQQQII', b.raw)
        va_cur[0] += (sz + 0xFFF) & ~0xFFF
        log(f"  {nm}: handle=0x{h[2]:x} mmap_off=0x{h[3]:x} va=0x{va:x}")
        return (h[2], va, h[3])

    def mapgpu(handle, nm):
        gids = ctypes.create_string_buffer(8)
        struct.pack_into('<I', gids, 0, gpu_id)
        gp = ctypes.addressof(gids)
        b = ctypes.create_string_buffer(24)
        struct.pack_into('<QQIi', b, 0, handle, gp, 1, 0)
        r = libc.ioctl(kfd, IOC_MAP, b)
        if r != 0:
            e = ctypes.get_errno()
            log(f"  MAP {nm}: FAIL {e} ({os.strerror(e)})")
            return False
        log(f"  MAP {nm}: OK")
        return True

    # Allocate all queue buffers
    ring = alloc_gpu(4096, GTT | WR | EX | UC, 'ring')     # Match HSA's ring_size=4096
    eop  = alloc_gpu(4096, GTT | WR | UC, 'eop')
    # HSA uses ctx_sz=13942784 (0xD4C000) — this is ctx_save_restore + ctl_stack combined
    # ctx_save_restore_size includes ctl_stack_size
    ctx  = alloc_gpu(13942784, GTT | WR | UC, 'ctx')
    # wptr/rptr buffer — GPU-mapped! One page, wptr at +0x38, rptr at +0x80 (matching HSA offsets)
    wrbuf = alloc_gpu(4096, GTT | WR | UC, 'wptr_rptr')

    if not all([ring, eop, ctx, wrbuf]):
        log("ALLOC failed")
        hsa.hsa_shut_down()
        return

    for h, nm in [(ring, 'ring'), (eop, 'eop'), (ctx, 'ctx'), (wrbuf, 'wptr_rptr')]:
        mapgpu(h[0], nm)

    # CPU mmap of ring
    ring_mm = mmap.mmap(drm, 4096, mmap.MAP_SHARED,
                         mmap.PROT_READ | mmap.PROT_WRITE, offset=ring[2])
    ring_mm.write(b'\x00' * 4096)
    ring_mm.seek(0)

    # CPU mmap of wptr/rptr page — zero it, then get GPU VAs for offsets
    wr_mm = mmap.mmap(drm, 4096, mmap.MAP_SHARED,
                       mmap.PROT_READ | mmap.PROT_WRITE, offset=wrbuf[2])
    wr_mm.write(b'\x00' * 4096)

    # HSA puts wptr at +0x38 and rptr at +0x80 from buffer base
    wptr_gpu_va = wrbuf[1] + 0x38
    rptr_gpu_va = wrbuf[1] + 0x80

    log(f"  wptr_gpu_va=0x{wptr_gpu_va:x}")
    log(f"  rptr_gpu_va=0x{rptr_gpu_va:x}")

    # Zero the wptr/rptr locations
    struct.pack_into('<Q', wr_mm, 0x38, 0)
    struct.pack_into('<Q', wr_mm, 0x80, 0)

    # CREATE_QUEUE
    log("\n--- CREATE_QUEUE ---")
    results = {"tests": []}

    # Match HSA exactly: queue_type=2 (AQL) first, then try type=0 (PM4)
    for qtype, qtname in [(2, 'AQL'), (0, 'PM4')]:
        for ring_sz in [4096, 16384, 65536]:
            for ctl in [16384, 4096, 0]:
                qb = ctypes.create_string_buffer(96)
                struct.pack_into('<QQQQIIIIIiQQQIIII', qb, 0,
                    ring[1],        # ring_base (GPU VA)
                    wptr_gpu_va,    # write_pointer (GPU VA!)
                    rptr_gpu_va,    # read_pointer (GPU VA!)
                    0,              # doorbell (output)
                    ring_sz,        # ring_size (must match allocation)
                    gpu_id,
                    qtype,
                    100,            # percentage
                    7,              # priority (normal)
                    0,              # queue_id (output)
                    eop[1],         # eop addr
                    4096,           # eop size
                    ctx[1],         # ctx addr
                    13942784,       # ctx_save_restore_size (matches HSA's 0xD4C000)
                    ctl,            # ctl_stack_size
                    0,              # sdma_engine_id
                    0,              # pad
                )
                r = libc.ioctl(kfd, IOC_CREATE_Q, qb)
                if r == 0:
                    vals = struct.unpack_from('<QQQQIIIIIiQQQIIII', qb.raw)
                    qid = vals[9]
                    doorbell_off = vals[3]
                    log(f"*** QUEUE OK! type={qtname} ring_sz={ring_sz} ctl={ctl} ***")
                    log(f"    queue_id={qid}")
                    log(f"    doorbell_offset=0x{doorbell_off:x}")
                    log(f"    ring_base=0x{vals[0]:x}")

                    results["queue_ok"] = True
                    results["queue_type"] = qtname
                    results["queue_id"] = qid
                    results["doorbell_offset"] = f"0x{doorbell_off:x}"

                    if qtype == 0:
                        log("\n=== PM4 queue created! Testing WRITE_DATA ===")
                        test_pm4(ring_mm, wr_mm, doorbell_off, drm, kfd, results)
                    else:
                        log(f"\n  AQL queue created (type={qtype}). For PM4, need type=0.")
                        log("  Will try PM4 next...")
                        # Don't exit — try PM4 too
                        results["aql_ok"] = True
                        # Destroy this queue and try PM4
                        dq = ctypes.create_string_buffer(8)
                        struct.pack_into('<Ii', dq, 0, qid, 0)
                        libc.ioctl(kfd, _IOWR(K, 0x03, 8), dq)
                        continue

                    save_results(results)
                    hsa.hsa_shut_down()
                    return
                else:
                    e = ctypes.get_errno()
                    results["tests"].append({
                        "type": qtname, "ring_sz": ring_sz, "ctl": ctl,
                        "errno": e, "error": os.strerror(e)
                    })
                    if ring_sz == 4096 and ctl == 16384:  # only log first combo per type
                        log(f"  type={qtname} ring={ring_sz} ctl={ctl}: errno={e} ({os.strerror(e)})")

    log("All queue variants failed")
    results["queue_ok"] = False
    save_results(results)
    hsa.hsa_shut_down()


def test_pm4(ring_mm, wr_mm, doorbell_off, drm, kfd, results):
    """Test PM4 WRITE_DATA to IC registers."""
    # IC registers to test
    targets = [
        ("CP_ME_IC_BASE_LO", 0x5844),
        ("CP_ME_IC_BASE_HI", 0x5845),
        ("CP_ME_IC_OP_CNTL", 0x5847),
        ("CP_MES_IC_BASE_LO", 0x2820),
        ("SCRATCH_REG0", 0x2040),
    ]

    results["pm4_tests"] = []

    for name, reg_off in targets:
        before = read_reg(reg_off)
        test_val = 0xFACE0000 | (reg_off & 0xFFFF)

        # Build PM4 WRITE_DATA
        header = (3 << 30) | (0x37 << 8) | 2  # PKT3, WRITE_DATA, count=2
        control = (0 << 8) | (1 << 20)         # dst_sel=reg, wr_confirm=1
        pkt = struct.pack("<IIII", header, control, reg_off, test_val)

        # Write to ring
        ring_mm.seek(0)
        ring_mm.write(pkt)

        # Update wptr
        struct.pack_into('<Q', wr_mm, 0x38, 1)

        # TODO: ring doorbell via mmap
        time.sleep(0.3)
        after = read_reg(reg_off)

        bstr = f"0x{before:08X}" if before is not None else "FAIL"
        astr = f"0x{after:08X}" if after is not None else "FAIL"
        changed = before != after if before is not None and after is not None else None
        log(f"  {name} (0x{reg_off:04X}): {bstr} → {astr} {'CHANGED!' if changed else ''}")

        results["pm4_tests"].append({
            "name": name, "register": f"0x{reg_off:04X}",
            "before": bstr, "test_value": f"0x{test_val:08X}",
            "after": astr, "changed": changed,
        })


def save_results(results):
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(RESULTS / "z2350_hybrid_queue.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {RESULTS / 'z2350_hybrid_queue.json'}")


if __name__ == "__main__":
    main()
