#!/usr/bin/env python3
"""z2349: PM4 WRITE_DATA to IC registers via KFD compute queue.

z2345 KFD alloc failed because ioctl struct size was wrong (32 vs 40 bytes).
z2348b showed debugfs can't write ME IC_BASE_LO/HI but CAN write IC_OP_CNTL.

This script:
  1. Fixes KFD ioctl struct sizes
  2. Creates a compute queue
  3. Submits PM4 WRITE_DATA targeting ME IC_BASE_LO/HI
  4. If writable, attempts full ME IC redirect (BASE + INVALIDATE + PRIME)
  5. Logs every step, checkpoints JSON after each step

Run: sudo PYTHONUNBUFFERED=1 python3 scripts/z2349_pm4_ic_write.py
"""
import os, sys, json, time, struct, ctypes, ctypes.util, mmap, signal, hashlib
from pathlib import Path
from datetime import datetime

BASE = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
LOG_FILE = RESULTS / "z2349_log.txt"
JSON_OUT = RESULTS / "z2349_pm4_ic_write.json"
THERMAL = "/sys/class/thermal/thermal_zone0/temp"
DEBUGFS_REGS = "/sys/kernel/debug/dri/0/amdgpu_regs"
VRAM_FILE = "/sys/kernel/debug/dri/0/amdgpu_vram"

log_fh = open(LOG_FILE, "w")

state = {
    "started": datetime.now().isoformat(),
    "pid": os.getpid(),
    "steps": {},
    "anomalies": [],
    "health_checks": 0,
    "writes_attempted": 0,
    "writes_confirmed": 0,
    "writes_rejected": 0,
}

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()

def save():
    state["last_save"] = datetime.now().isoformat()
    with open(JSON_OUT, "w") as f:
        json.dump(state, f, indent=2, default=str)

def get_temp():
    try: return int(open(THERMAL).read().strip()) / 1000.0
    except: return 0.0

def health(tag):
    t = get_temp()
    state["health_checks"] += 1
    grbm = read_reg(0x2000)
    gstr = f"0x{grbm:08X}" if grbm is not None else "TIMEOUT"
    log(f"  HEALTH {tag}: GRBM={gstr} T={t:.1f}C")
    if t > 85:
        log(f"  THERMAL ABORT at {t:.1f}C")
        save()
        sys.exit(1)
    return t

def read_reg(dword_offset, timeout=5):
    """Read register via debugfs. dword_offset = register index."""
    byte_off = dword_offset * 4
    old_handler = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(timeout)
    try:
        with open(DEBUGFS_REGS, "rb") as f:
            f.seek(byte_off)
            data = f.read(4)
        signal.alarm(0)
        if len(data) == 4:
            return struct.unpack("<I", data)[0]
        return None
    except:
        signal.alarm(0)
        return None
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_DFL)

def debugfs_write_reg(dword_offset, value):
    """Write register via debugfs."""
    byte_off = dword_offset * 4
    try:
        with open(DEBUGFS_REGS, "r+b") as f:
            f.seek(byte_off)
            f.write(struct.pack("<I", value))
            f.flush()
        return True
    except:
        return False

# ===== KFD ioctl definitions with CORRECT struct sizes =====
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

AMDKFD_IOCTL_BASE = ord('K')

def _IOC(d, t, nr, sz):
    return (d << 30) | (t << 8) | nr | (sz << 16)
def _IOR(t, nr, sz):  return _IOC(2, t, nr, sz)
def _IOW(t, nr, sz):  return _IOC(1, t, nr, sz)
def _IOWR(t, nr, sz): return _IOC(3, t, nr, sz)

# Correct struct sizes from kernel kfd_ioctl.h
# GET_VERSION: {u32 major, u32 minor} = 8 bytes
AMDKFD_IOC_GET_VERSION = _IOR(AMDKFD_IOCTL_BASE, 0x01, 8)

# GET_PROCESS_APERTURES: 7 * {u64,u64,u64,u64,u64,u64,u32,u32} + {u32,u32}
# Each aperture = 6*8 + 2*4 = 56 bytes. 7*56 + 8 = 400 bytes
AMDKFD_IOC_GET_PROCESS_APERTURES = _IOR(AMDKFD_IOCTL_BASE, 0x06, 400)

# ACQUIRE_VM: {u32 drm_fd, u32 gpu_id} = 8 bytes
AMDKFD_IOC_ACQUIRE_VM = _IOW(AMDKFD_IOCTL_BASE, 0x15, 8)

# ALLOC_MEMORY_OF_GPU: {u64 va, u64 size, u64 handle, u64 mmap_off, u32 gpu_id, u32 flags} = 40 bytes
# *** nr=0x16 per kernel kfd_ioctl.h (was 0x18 in z2349v1, and 32-byte struct in z2345) ***
AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = _IOWR(AMDKFD_IOCTL_BASE, 0x16, 40)

# FREE_MEMORY_OF_GPU: {u64 handle} = 8 bytes (nr=0x17)
AMDKFD_IOC_FREE_MEMORY_OF_GPU = _IOW(AMDKFD_IOCTL_BASE, 0x17, 8)

# MAP_MEMORY_TO_GPU: {u64 handle, u64 device_ids_array_ptr, u32 n_devices, u32 n_success} = 24 bytes (nr=0x18)
AMDKFD_IOC_MAP_MEMORY_TO_GPU = _IOWR(AMDKFD_IOCTL_BASE, 0x18, 24)

# CREATE_QUEUE: complex struct, 88 bytes
AMDKFD_IOC_CREATE_QUEUE = _IOWR(AMDKFD_IOCTL_BASE, 0x02, 88)

# DESTROY_QUEUE: {u32 queue_id, u32 pad} = 8 bytes
AMDKFD_IOC_DESTROY_QUEUE = _IOWR(AMDKFD_IOCTL_BASE, 0x03, 8)

KFD_IOC_QUEUE_TYPE_COMPUTE = 0
KFD_IOC_ALLOC_MEM_FLAGS_GTT       = (1 << 1)
KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE  = (1 << 31)
KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = (1 << 30)
KFD_IOC_ALLOC_MEM_FLAGS_COHERENT  = (1 << 26)
KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED  = (1 << 25)
KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = (1 << 23)

# PM4 opcodes (GFX11)
def PKT3(opcode, count):
    return (3 << 30) | ((opcode & 0xFF) << 8) | ((count - 1) & 0x3FFF)

PM4_NOP         = 0x10
PM4_WRITE_DATA  = 0x37
PM4_COPY_DATA   = 0x40
PM4_RELEASE_MEM = 0x49

# WRITE_DATA dst selectors
WRITE_DATA_DST_REG = 0
WRITE_DATA_DST_MEM = 5

# IC register addresses (dword offsets)
IC_REGS = {
    'ME_IC_BASE_LO':   0x5844,
    'ME_IC_BASE_HI':   0x5845,
    'ME_IC_BASE_CNTL': 0x5846,
    'ME_IC_OP_CNTL':   0x5847,
    'CPC_IC_BASE_LO':  0x584C,
    'CPC_IC_BASE_HI':  0x584D,
    'CPC_IC_BASE_CNTL':0x584E,
    'CPC_IC_OP_CNTL':  0x297A,
    'PFP_IC_BASE_LO':  0x5840,
    'PFP_IC_BASE_HI':  0x5841,
    'PFP_IC_BASE_CNTL':0x5842,
    'PFP_IC_OP_CNTL':  0x5843,
    'MES_IC_BASE_LO':  0x5850,
    'MES_IC_BASE_HI':  0x5851,
    'MES_IC_BASE_CNTL':0x5852,
    'MES_IC_OP_CNTL':  0x2820,
}


def step0():
    """Sanity check."""
    log("=" * 70)
    log("STEP 0: Sanity")
    log("=" * 70)
    t = health("step0")
    state["steps"]["step0"] = {"status": "OK", "temp": t}
    save()
    time.sleep(1)


def step1_kfd_setup():
    """Open KFD, get GPU info, allocate memory, create compute queue."""
    log("=" * 70)
    log("STEP 1: KFD Queue Setup (fixed struct sizes)")
    log("=" * 70)

    result = {"substeps": []}

    # 1a: Open /dev/kfd
    kfd_fd = os.open("/dev/kfd", os.O_RDWR)
    log(f"  /dev/kfd opened: fd={kfd_fd}")
    result["substeps"].append(f"kfd_fd={kfd_fd}")

    # 1b: KFD version
    ver_buf = ctypes.create_string_buffer(8)
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_GET_VERSION, ver_buf)
    if ret == 0:
        major, minor = struct.unpack("<II", ver_buf.raw)
        log(f"  KFD version: {major}.{minor}")
        result["kfd_version"] = f"{major}.{minor}"
    else:
        errno = ctypes.get_errno()
        log(f"  GET_VERSION failed: errno={errno}")
        result["kfd_version"] = f"FAIL errno={errno}"

    # 1c: Process apertures -> gpu_id
    ap_size = 400
    ap_buf = ctypes.create_string_buffer(ap_size)
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_GET_PROCESS_APERTURES, ap_buf)
    gpu_id = None
    if ret == 0:
        num_nodes = struct.unpack_from("<I", ap_buf.raw, 7 * 56)[0]
        log(f"  GPU nodes: {num_nodes}")
        for i in range(min(num_nodes, 7)):
            off = i * 56
            vals = struct.unpack_from("<QQQQQQII", ap_buf.raw, off)
            lds_b, lds_l, scratch_b, scratch_l, gpuvm_b, gpuvm_l, gid = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]
            log(f"  GPU {i}: gpu_id=0x{gid:04x} gpuvm=0x{gpuvm_b:016x}-0x{gpuvm_l:016x}")
            if gid != 0:
                gpu_id = gid
    else:
        errno = ctypes.get_errno()
        log(f"  GET_PROCESS_APERTURES failed: errno={errno}")

    if gpu_id is None:
        log("  FAIL: No GPU found")
        state["steps"]["step1"] = {"status": "FAIL", "reason": "no GPU"}
        save()
        return None

    result["gpu_id"] = f"0x{gpu_id:04x}"
    log(f"  Selected gpu_id=0x{gpu_id:04x}")

    # 1d: Open DRM render node + acquire VM
    drm_fd = os.open("/dev/dri/renderD128", os.O_RDWR)
    log(f"  renderD128: fd={drm_fd}")

    acq_buf = ctypes.create_string_buffer(8)
    struct.pack_into("<II", acq_buf, 0, drm_fd, gpu_id)
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_ACQUIRE_VM, acq_buf)
    log(f"  ACQUIRE_VM: {'OK' if ret == 0 else f'FAIL errno={ctypes.get_errno()}'}")
    result["acquire_vm"] = ret == 0

    health("step1_vm")

    # 1e: Allocate GPU memory
    # Need a VA allocator within GPUVM range (0x10000 - 0x7fffffffffff)
    next_va = [0x7FFF00000000]  # start high in GPUVM range to avoid conflicts

    def kfd_alloc(kfd_fd, gpu_id, size, flags, name):
        va_hint = next_va[0]
        # Align to page
        aligned_size = (size + 0xFFF) & ~0xFFF
        buf = struct.pack("<QQQQII", va_hint, aligned_size, 0, 0, gpu_id, flags & 0xFFFFFFFF)
        arr = ctypes.create_string_buffer(40)
        ctypes.memmove(arr, buf, 40)
        log(f"  ALLOC {name}: size={aligned_size}, flags=0x{flags:08x}, va_hint=0x{va_hint:x}")
        ret = libc.ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, arr)
        if ret != 0:
            errno = ctypes.get_errno()
            log(f"  ALLOC {name} FAILED: errno={errno} ({os.strerror(errno)})")
            return None
        va, sz, handle, mmap_off, gid, fl = struct.unpack_from("<QQQQII", arr.raw)
        log(f"  ALLOC {name} OK: handle=0x{handle:x} va=0x{va:016x} mmap=0x{mmap_off:x}")
        next_va[0] += aligned_size  # advance for next alloc
        return handle, va_hint, mmap_off

    ring_flags = (KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                  KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                  KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                  KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    ring_size = 4096 * 4  # 16KB

    ring = kfd_alloc(kfd_fd, gpu_id, ring_size, ring_flags, "ring")
    if ring is None:
        # Try without EXECUTABLE
        log("  Retrying ring without EXECUTABLE flag...")
        ring_flags2 = (KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                       KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                       KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
        ring = kfd_alloc(kfd_fd, gpu_id, ring_size, ring_flags2, "ring_v2")

    if ring is None:
        # Try with COHERENT
        log("  Retrying ring with COHERENT flag...")
        ring_flags3 = (KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                       KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                       KFD_IOC_ALLOC_MEM_FLAGS_COHERENT)
        ring = kfd_alloc(kfd_fd, gpu_id, ring_size, ring_flags3, "ring_v3")

    if ring is None:
        log("  All ring alloc attempts FAILED")
        # Try to dump the actual ioctl number for debugging
        log(f"  ALLOC ioctl number: 0x{AMDKFD_IOC_ALLOC_MEMORY_OF_GPU:08x}")
        log(f"  Expected for 40-byte struct: dir=3 type=0x4B nr=0x18 size=40")
        computed = _IOWR(AMDKFD_IOCTL_BASE, 0x18, 40)
        log(f"  Computed: 0x{computed:08x}")

        # The primary ioctl is now 0x16 (correct). Try alternate sizes as fallback.
        log("  Primary ioctl 0x16 already tried. Skipping duplicate.")

    if ring is None:
        # Last resort: try different struct sizes (maybe kernel added fields)
        for test_size in [48, 56, 64]:
            log(f"  Trying struct size {test_size} with ioctl nr=0x16...")
            test_ioctl = _IOWR(AMDKFD_IOCTL_BASE, 0x16, test_size)
            buf = struct.pack("<QQQQII", 0, ring_size, 0, 0, gpu_id, ring_flags & 0xFFFFFFFF)
            buf += b'\x00' * (test_size - len(buf))
            arr = ctypes.create_string_buffer(test_size)
            ctypes.memmove(arr, buf, test_size)
            ret = libc.ioctl(kfd_fd, test_ioctl, arr)
            if ret == 0:
                va, sz, handle, mmap_off, gid, fl = struct.unpack_from("<QQQQII", arr.raw)
                log(f"  SIZE {test_size} WORKED! handle=0x{handle:x} va=0x{va:016x}")
                ring = (handle, va, mmap_off)
                break
            else:
                errno = ctypes.get_errno()
                log(f"  Size {test_size}: errno={errno}")

    if ring is None:
        state["steps"]["step1"] = {"status": "FAIL", "reason": "ring alloc failed all variants"}
        save()
        return None

    ring_handle, ring_va, ring_mmap = ring
    result["ring"] = {"handle": hex(ring_handle), "va": hex(ring_va)}

    health("step1_alloc")

    # EOP buffer
    eop_flags = (KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                 KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                 KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    eop_size = 4096
    eop = kfd_alloc(kfd_fd, gpu_id, eop_size, eop_flags, "eop")
    eop_handle, eop_va, eop_mmap = eop if eop else (0, 0, 0)

    # CTX save/restore
    ctx_size = 4096 * 16
    ctx = kfd_alloc(kfd_fd, gpu_id, ctx_size, eop_flags, "ctx")
    ctx_handle, ctx_va, ctx_mmap = ctx if ctx else (0, 0, 0)

    # Result buffer (for COPY_DATA readback)
    rb_size = 4096
    rb = kfd_alloc(kfd_fd, gpu_id, rb_size, eop_flags, "result_buf")
    rb_handle, rb_va, rb_mmap = rb if rb else (0, 0, 0)

    # Map all buffers to GPU
    def kfd_map(kfd_fd, handle, gpu_id, name):
        if handle == 0:
            return False
        gpu_ids = struct.pack("<I", gpu_id)
        gpu_ids_buf = ctypes.create_string_buffer(gpu_ids)
        gpu_ids_ptr = ctypes.addressof(gpu_ids_buf)
        buf = struct.pack("<QQIi", handle, gpu_ids_ptr, 1, 0)
        arr = ctypes.create_string_buffer(24)
        ctypes.memmove(arr, buf, 24)
        ret = libc.ioctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, arr)
        ok = ret == 0
        if not ok:
            errno = ctypes.get_errno()
            log(f"  MAP {name}: FAIL errno={errno}")
        else:
            log(f"  MAP {name}: OK")
        return ok

    kfd_map(kfd_fd, ring_handle, gpu_id, "ring")
    kfd_map(kfd_fd, eop_handle, gpu_id, "eop")
    kfd_map(kfd_fd, ctx_handle, gpu_id, "ctx")
    kfd_map(kfd_fd, rb_handle, gpu_id, "result_buf")

    # mmap ring to CPU
    ring_cpu = None
    try:
        ring_cpu = mmap.mmap(kfd_fd, ring_size, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE,
                             offset=ring_mmap)
        log(f"  Ring mmap: OK")
    except Exception as e:
        log(f"  Ring mmap FAILED: {e}")
        # Try via /dev/kfd offset
        try:
            ring_cpu = mmap.mmap(kfd_fd, ring_size, flags=mmap.MAP_SHARED,
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                 offset=ring_mmap & 0xFFFFFFFFFFFF)
            log(f"  Ring mmap (masked offset): OK")
        except Exception as e2:
            log(f"  Ring mmap (masked): {e2}")

    # mmap result buf to CPU
    rb_cpu = None
    if rb_mmap:
        try:
            rb_cpu = mmap.mmap(kfd_fd, rb_size, mmap.MAP_SHARED,
                               mmap.PROT_READ | mmap.PROT_WRITE,
                               offset=rb_mmap)
            log(f"  Result buf mmap: OK")
        except Exception as e:
            log(f"  Result buf mmap FAILED: {e}")

    health("step1_map")

    # Create write/read pointer
    wptr_buf = ctypes.create_string_buffer(8)
    rptr_buf = ctypes.create_string_buffer(8)
    wptr_addr = ctypes.addressof(wptr_buf)
    rptr_addr = ctypes.addressof(rptr_buf)

    # Try mmap ring via drm fd
    if ring_cpu is None:
        try:
            ring_cpu = mmap.mmap(drm_fd, ring_size, mmap.MAP_SHARED,
                                 mmap.PROT_READ | mmap.PROT_WRITE,
                                 offset=ring_mmap)
            log(f"  Ring mmap (drm fd): OK")
        except Exception as e:
            log(f"  Ring mmap (drm fd): {e}")

    # Try mmap ring as anonymous + USERPTR
    if ring_cpu is None:
        log("  Trying userptr approach for ring...")
        ring_cpu = mmap.mmap(-1, ring_size, mmap.MAP_SHARED | 0x20,  # MAP_ANONYMOUS
                              mmap.PROT_READ | mmap.PROT_WRITE)
        ring_cpu_addr = ctypes.addressof(ctypes.c_char.from_buffer(ring_cpu))
        log(f"  Anonymous mmap for ring: addr=0x{ring_cpu_addr:x}")

    # Create compute queue
    log("  Creating compute queue...")
    q_arr = ctypes.create_string_buffer(88)
    struct.pack_into("<QQQQIIIIIiQQQII", q_arr, 0,
        ring_va,       # ring_base_address
        wptr_addr,     # write_pointer_address
        rptr_addr,     # read_pointer_address
        0,             # doorbell_offset (output)
        ring_size,     # ring_size
        gpu_id,        # gpu_id
        KFD_IOC_QUEUE_TYPE_COMPUTE,
        100,           # queue_percentage
        7,             # queue_priority
        0,             # queue_id (output)
        eop_va,        # eop_buffer_address
        eop_size,      # eop_buffer_size
        ctx_va,        # ctx_save_restore_address
        ctx_size,      # ctx_save_restore_size
        4096,          # ctl_stack_size
    )
    log(f"  Queue params: ring_va=0x{ring_va:x} eop_va=0x{eop_va:x} ctx_va=0x{ctx_va:x}")
    log(f"  Queue params: wptr=0x{wptr_addr:x} rptr=0x{rptr_addr:x}")
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, q_arr)

    queue_id = None
    doorbell_offset = None
    if ret == 0:
        vals = struct.unpack_from("<QQQQIIIIIiQQQII", q_arr.raw)
        queue_id = vals[9]
        doorbell_offset = vals[3]
        log(f"  Queue CREATED! id={queue_id} doorbell=0x{doorbell_offset:016x}")
        result["queue_id"] = queue_id
        result["doorbell"] = hex(doorbell_offset)
    else:
        errno = ctypes.get_errno()
        log(f"  CREATE_QUEUE FAILED: errno={errno} ({os.strerror(errno)})")
        result["queue_fail"] = f"errno={errno}"

    result["status"] = "QUEUE_OK" if queue_id is not None else "QUEUE_FAIL"
    state["steps"]["step1"] = result
    save()

    if queue_id is None and ring_cpu is None:
        return None

    return {
        "kfd_fd": kfd_fd,
        "drm_fd": drm_fd,
        "gpu_id": gpu_id,
        "queue_id": queue_id,
        "ring_cpu": ring_cpu,
        "ring_va": ring_va,
        "ring_size": ring_size,
        "rb_cpu": rb_cpu,
        "rb_va": rb_va,
        "wptr_buf": wptr_buf,
        "rptr_buf": rptr_buf,
        "wptr_addr": wptr_addr,
        "doorbell_offset": doorbell_offset,
    }


def submit_pm4(ctx, dwords):
    """Write PM4 packet to ring buffer and advance write pointer."""
    ring_cpu = ctx["ring_cpu"]
    wptr_buf = ctx["wptr_buf"]

    if ring_cpu is None:
        log("  ERROR: ring_cpu is None")
        return False

    # Get current wptr
    wptr = struct.unpack("<Q", wptr_buf.raw)[0]
    ring_off = (wptr % ctx["ring_size"])

    # Write PM4 dwords
    for i, dw in enumerate(dwords):
        off = (ring_off + i * 4) % ctx["ring_size"]
        ring_cpu.seek(off)
        ring_cpu.write(struct.pack("<I", dw))

    ring_cpu.flush()

    # Advance write pointer
    new_wptr = wptr + len(dwords) * 4
    struct.pack_into("<Q", wptr_buf, 0, new_wptr)

    # Ring doorbell (write to doorbell page)
    # The kernel maps the doorbell at the doorbell offset
    # For now, just updating wptr should trigger processing
    time.sleep(0.1)
    return True


def build_write_data_reg(reg_addr, value):
    """Build PM4 WRITE_DATA packet to write a register.
    reg_addr is the dword offset (register index).
    """
    # WRITE_DATA: header + control + dst_addr_lo + dst_addr_hi + data
    # Control: DST_SEL=0 (register), WR_CONFIRM=1
    control = (WRITE_DATA_DST_REG << 8) | (1 << 20)  # WR_CONFIRM
    return [
        PKT3(PM4_WRITE_DATA, 4),  # 4 DWORDs follow header
        control,
        reg_addr,        # register address (dword offset)
        0,               # dst_addr_hi (0 for regs)
        value,           # data to write
    ]


def build_copy_data_reg_to_mem(reg_addr, dst_gpu_va):
    """Build PM4 COPY_DATA packet: read register -> write to GPU memory."""
    # COPY_DATA: src_sel=0 (reg), dst_sel=5 (mem_mapped), count=0 (1 dword)
    control = (0 << 0) | (5 << 8) | (1 << 20)  # src=reg, dst=mem, wr_confirm
    return [
        PKT3(PM4_COPY_DATA, 5),  # 5 DWORDs follow
        control,
        reg_addr,                         # src_reg (dword offset)
        0,                                # src_reg_hi
        dst_gpu_va & 0xFFFFFFFF,          # dst_addr_lo
        (dst_gpu_va >> 32) & 0xFFFFFFFF,  # dst_addr_hi
    ]


def step2_pm4_scratch_test(ctx):
    """Test PM4 WRITE_DATA on scratch registers to verify the queue works."""
    log("=" * 70)
    log("STEP 2: PM4 scratch register write test")
    log("=" * 70)

    if ctx is None or ctx.get("queue_id") is None:
        log("  SKIP: No compute queue available")
        state["steps"]["step2"] = {"status": "SKIP", "reason": "no queue"}
        save()
        return False

    SCRATCH_REG0 = 0x2040

    # Read original via debugfs
    orig = read_reg(SCRATCH_REG0)
    log(f"  SCRATCH_REG0 before: {f'0x{orig:08X}' if orig is not None else 'TIMEOUT'}")

    # Submit WRITE_DATA to SCRATCH_REG0
    test_val = 0xCAFE1234
    pkt = build_write_data_reg(SCRATCH_REG0, test_val)
    log(f"  Submitting WRITE_DATA(0x{SCRATCH_REG0:04X}, 0x{test_val:08X})")
    log(f"  PM4 packet: {' '.join(f'{dw:08X}' for dw in pkt)}")

    ok = submit_pm4(ctx, pkt)
    log(f"  Submit: {'OK' if ok else 'FAIL'}")

    time.sleep(0.2)

    # Read back
    rb = read_reg(SCRATCH_REG0)
    accepted = rb == test_val
    log(f"  Readback: {f'0x{rb:08X}' if rb is not None else 'TIMEOUT'}")
    log(f"  PM4 write {'ACCEPTED' if accepted else 'REJECTED'}")

    state["writes_attempted"] += 1
    if accepted:
        state["writes_confirmed"] += 1
    else:
        state["writes_rejected"] += 1

    # Restore
    if accepted and orig is not None:
        pkt2 = build_write_data_reg(SCRATCH_REG0, orig)
        submit_pm4(ctx, pkt2)
        time.sleep(0.1)
        rb2 = read_reg(SCRATCH_REG0)
        log(f"  Restored: 0x{rb2:08X}" if rb2 is not None else "  Restore: TIMEOUT")

    result = {
        "original": f"0x{orig:08X}" if orig is not None else None,
        "test_value": f"0x{test_val:08X}",
        "readback": f"0x{rb:08X}" if rb is not None else None,
        "pm4_write_accepted": accepted,
        "status": "PM4_WORKS" if accepted else "PM4_REJECTED",
    }
    state["steps"]["step2"] = result
    health("step2")
    save()
    time.sleep(1)
    return accepted


def step3_pm4_ic_write_sweep(ctx):
    """Use PM4 WRITE_DATA to test all 16 IC registers."""
    log("=" * 70)
    log("STEP 3: PM4 WRITE_DATA to IC registers")
    log("=" * 70)

    if ctx is None or ctx.get("queue_id") is None:
        log("  SKIP: No compute queue")
        state["steps"]["step3"] = {"status": "SKIP"}
        save()
        return {}

    results = []
    writable = []
    non_writable = []

    test_values = {
        'BASE_LO': 0x0000CAFE,
        'BASE_HI': 0x00000001,
        'BASE_CNTL': 0x00000000,  # try zeroing
        'OP_CNTL': 0x00000001,    # INVALIDATE bit
    }

    for name, addr in sorted(IC_REGS.items()):
        health(f"step3_{name}")

        # Pick test value
        tv = 0x0000CAFE
        for suffix, val in test_values.items():
            if name.endswith(suffix):
                tv = val
                break

        # Read original
        orig = read_reg(addr)
        log(f"  {name} (0x{addr:04X}): orig={f'0x{orig:08X}' if orig is not None else 'TIMEOUT'}")

        if orig is None:
            results.append({"register": name, "status": "UNREADABLE"})
            continue

        # Submit PM4 WRITE_DATA
        pkt = build_write_data_reg(addr, tv)
        log(f"    PM4 WRITE_DATA(0x{addr:04X}, 0x{tv:08X})")
        submit_pm4(ctx, pkt)
        time.sleep(0.15)

        state["writes_attempted"] += 1

        # Read back
        rb = read_reg(addr)
        accepted = rb == tv
        log(f"    Readback: 0x{rb:08X} {'✓ ACCEPTED' if accepted else '✗ REJECTED'}")

        if accepted:
            state["writes_confirmed"] += 1
            writable.append(name)
        else:
            state["writes_rejected"] += 1
            non_writable.append(name)

        # Restore
        pkt_restore = build_write_data_reg(addr, orig)
        submit_pm4(ctx, pkt_restore)
        time.sleep(0.1)
        rb_restore = read_reg(addr)
        log(f"    Restore: 0x{rb_restore:08X}" if rb_restore is not None else "    Restore: TIMEOUT")

        results.append({
            "register": name,
            "address": f"0x{addr:04X}",
            "original": f"0x{orig:08X}",
            "test_value": f"0x{tv:08X}",
            "readback": f"0x{rb:08X}" if rb is not None else None,
            "pm4_accepted": accepted,
            "restored": f"0x{rb_restore:08X}" if rb_restore is not None else None,
        })

    # Summary
    log(f"\n  PM4 WRITABLE IC registers: {writable}")
    log(f"  PM4 NON-WRITABLE IC registers: {non_writable}")

    # Check per-engine capability
    engines = {}
    for eng in ['PFP', 'ME', 'CPC', 'MES']:
        has_base = any(n.startswith(eng) and ('BASE_LO' in n or 'BASE_HI' in n) for n in writable)
        has_op = any(n.startswith(eng) and 'OP_CNTL' in n for n in writable)
        has_cntl = any(n.startswith(eng) and 'BASE_CNTL' in n for n in writable)
        engines[eng] = {"base": has_base, "op_cntl": has_op, "cntl": has_cntl}
        if has_base and has_op:
            log(f"  *** {eng}: FULL REDIRECT POSSIBLE (BASE + OP_CNTL writable via PM4) ***")

    state["steps"]["step3"] = {
        "results": results,
        "writable": writable,
        "non_writable": non_writable,
        "engines": engines,
    }
    health("step3_done")
    save()
    time.sleep(1)
    return engines


def step4_me_redirect(ctx, engines):
    """If ME has BASE + OP_CNTL via PM4, attempt full redirect sequence."""
    log("=" * 70)
    log("STEP 4: ME IC Redirect Attempt")
    log("=" * 70)

    me = engines.get("ME", {})
    has_base = me.get("base", False)
    has_op = me.get("op_cntl", False)

    log(f"  ME IC_BASE writable (PM4): {has_base}")
    log(f"  ME IC_OP_CNTL writable (PM4): {has_op}")

    # Also check CPC as alternative
    cpc = engines.get("CPC", {})
    cpc_base = cpc.get("base", False)
    cpc_op = cpc.get("op_cntl", False)
    log(f"  CPC IC_BASE writable (PM4): {cpc_base}")
    log(f"  CPC IC_OP_CNTL writable (PM4): {cpc_op}")

    # Find any engine with both
    target_engine = None
    for eng in ['ME', 'CPC', 'PFP', 'MES']:
        e = engines.get(eng, {})
        if e.get("base") and e.get("op_cntl"):
            target_engine = eng
            break

    if target_engine is None:
        log("  No engine has both IC_BASE + IC_OP_CNTL writable via PM4")

        # Can we combine debugfs (OP_CNTL for ME) + PM4 (BASE)?
        log("  Checking hybrid approach: PM4 for BASE + debugfs for OP_CNTL...")
        # ME has debugfs OP_CNTL. If PM4 can write ME BASE, that's enough.
        if has_base:
            log("  HYBRID possible: PM4 writes ME IC_BASE, debugfs writes ME IC_OP_CNTL")
            target_engine = "ME"
            hybrid = True
        elif cpc_base:
            log("  CPC has PM4 BASE. Trying debugfs OP_CNTL on CPC...")
            # Test CPC OP_CNTL via debugfs (was rejected in z2348b, but let's confirm)
            debugfs_write_reg(IC_REGS['CPC_IC_OP_CNTL'], 0x00000001)
            time.sleep(0.05)
            rb = read_reg(IC_REGS['CPC_IC_OP_CNTL'])
            if rb == 1:
                log("  CPC OP_CNTL debugfs write ACCEPTED")
                target_engine = "CPC"
                hybrid = True
            else:
                log(f"  CPC OP_CNTL debugfs: 0x{rb:08X} (rejected)")
                hybrid = False
        else:
            hybrid = False
    else:
        hybrid = False
        log(f"  Target engine: {target_engine} (full PM4 path)")

    if target_engine is None:
        log("  REDIRECT NOT POSSIBLE — no write path to both BASE and OP_CNTL on any engine")
        state["steps"]["step4"] = {"status": "NO_PATH", "reason": "no engine with both BASE+OP_CNTL"}
        save()
        return

    # Write NOP-sled payload to VRAM at 64MB offset
    PAYLOAD_OFFSET = 64 * 1024 * 1024  # 64MB
    PAYLOAD_SIZE = 4096
    NOP_WORD = 0xBF800000  # s_nop 0 (GFX11 scalar NOP)
    ENDPGM_WORD = 0xBF810000  # s_endpgm

    log(f"\n  Writing NOP-sled payload to VRAM at offset 0x{PAYLOAD_OFFSET:X}...")
    payload = struct.pack("<I", NOP_WORD) * (PAYLOAD_SIZE // 4 - 1)
    payload += struct.pack("<I", ENDPGM_WORD)

    try:
        with open(VRAM_FILE, "r+b") as f:
            f.seek(PAYLOAD_OFFSET)
            f.write(payload)
            f.flush()
        log(f"  Payload written ({PAYLOAD_SIZE} bytes)")

        # Verify
        with open(VRAM_FILE, "rb") as f:
            f.seek(PAYLOAD_OFFSET)
            readback = f.read(PAYLOAD_SIZE)
        match = readback == payload
        log(f"  Payload verify: {'MATCH' if match else 'MISMATCH'}")
    except Exception as e:
        log(f"  Payload write FAILED: {e}")
        state["steps"]["step4"] = {"status": "VRAM_FAIL", "error": str(e)}
        save()
        return

    health("step4_payload")

    # Compute IC_BASE value: gpu_addr >> 8
    # VRAM starts at GPU VA offset. For IC_BASE, the address is (gpu_addr >> 8)
    # VRAM physical offset 64MB -> register value = 0x40000000 >> 8 = 0x00400000
    ic_base_lo = PAYLOAD_OFFSET >> 8  # = 0x00040000
    ic_base_hi = 0x00000000

    eng_prefix = target_engine + "_IC"
    base_lo_reg = IC_REGS.get(f"{target_engine}_IC_BASE_LO")
    base_hi_reg = IC_REGS.get(f"{target_engine}_IC_BASE_HI")
    op_cntl_reg = IC_REGS.get(f"{target_engine}_IC_OP_CNTL")

    log(f"\n  Redirect sequence for {target_engine}:")
    log(f"    IC_BASE_LO <- 0x{ic_base_lo:08X} (payload at VRAM 64MB)")
    log(f"    IC_BASE_HI <- 0x{ic_base_hi:08X}")
    log(f"    IC_OP_CNTL <- 0x01 (INVALIDATE)")
    log(f"    IC_OP_CNTL <- 0x04 (PRIME)")

    # Save originals
    orig_lo = read_reg(base_lo_reg)
    orig_hi = read_reg(base_hi_reg)
    orig_op = read_reg(op_cntl_reg)
    log(f"    Originals: LO=0x{orig_lo:08X} HI=0x{orig_hi:08X} OP=0x{orig_op:08X}")

    # Step 4a: Write IC_BASE_LO
    log(f"\n  4a: Write IC_BASE_LO = 0x{ic_base_lo:08X}")
    pkt = build_write_data_reg(base_lo_reg, ic_base_lo)
    submit_pm4(ctx, pkt)
    time.sleep(0.1)
    rb_lo = read_reg(base_lo_reg)
    lo_ok = rb_lo == ic_base_lo
    log(f"    Readback: 0x{rb_lo:08X} {'✓' if lo_ok else '✗'}")

    # Step 4b: Write IC_BASE_HI
    log(f"  4b: Write IC_BASE_HI = 0x{ic_base_hi:08X}")
    pkt = build_write_data_reg(base_hi_reg, ic_base_hi)
    submit_pm4(ctx, pkt)
    time.sleep(0.1)
    rb_hi = read_reg(base_hi_reg)
    hi_ok = rb_hi == ic_base_hi
    log(f"    Readback: 0x{rb_hi:08X} {'✓' if hi_ok else '✗'}")

    health("step4_base_written")

    # Step 4c: INVALIDATE cache
    log(f"  4c: INVALIDATE (IC_OP_CNTL bit 0)")
    if hybrid and target_engine == "ME":
        # Use debugfs for OP_CNTL
        debugfs_write_reg(op_cntl_reg, 0x00000001)
    else:
        pkt = build_write_data_reg(op_cntl_reg, 0x00000001)
        submit_pm4(ctx, pkt)
    time.sleep(0.1)
    rb_inv = read_reg(op_cntl_reg)
    log(f"    IC_OP_CNTL after INVALIDATE: 0x{rb_inv:08X}")

    # Check INV_COMPLETE (bit 1)
    inv_complete = (rb_inv & 0x02) != 0 if rb_inv is not None else False
    log(f"    INV_COMPLETE: {inv_complete}")

    # Step 4d: PRIME cache (bit 2)
    log(f"  4d: PRIME (IC_OP_CNTL bit 2)")
    if hybrid and target_engine == "ME":
        debugfs_write_reg(op_cntl_reg, 0x00000004)
    else:
        pkt = build_write_data_reg(op_cntl_reg, 0x00000004)
        submit_pm4(ctx, pkt)
    time.sleep(0.2)
    rb_prime = read_reg(op_cntl_reg)
    log(f"    IC_OP_CNTL after PRIME: 0x{rb_prime:08X}")

    # Check ICACHE_PRIMED (bit 3)
    primed = (rb_prime & 0x08) != 0 if rb_prime is not None else False
    log(f"    ICACHE_PRIMED: {primed}")

    health("step4_redirect")

    # Step 4e: Restore originals
    log(f"\n  4e: Restoring originals...")
    if hybrid and target_engine == "ME":
        debugfs_write_reg(op_cntl_reg, orig_op)
    else:
        submit_pm4(ctx, build_write_data_reg(op_cntl_reg, orig_op))
    time.sleep(0.05)

    submit_pm4(ctx, build_write_data_reg(base_lo_reg, orig_lo))
    submit_pm4(ctx, build_write_data_reg(base_hi_reg, orig_hi))
    time.sleep(0.1)

    final_lo = read_reg(base_lo_reg)
    final_hi = read_reg(base_hi_reg)
    final_op = read_reg(op_cntl_reg)
    log(f"    Final: LO=0x{final_lo:08X} HI=0x{final_hi:08X} OP=0x{final_op:08X}")

    health("step4_restored")

    state["steps"]["step4"] = {
        "target_engine": target_engine,
        "hybrid": hybrid if target_engine else False,
        "base_lo_accepted": lo_ok,
        "base_hi_accepted": hi_ok,
        "invalidate_readback": f"0x{rb_inv:08X}" if rb_inv is not None else None,
        "inv_complete": inv_complete,
        "prime_readback": f"0x{rb_prime:08X}" if rb_prime is not None else None,
        "primed": primed,
        "redirect_success": lo_ok and hi_ok and primed,
        "status": "REDIRECT_OK" if (lo_ok and hi_ok and primed) else "PARTIAL",
    }
    save()
    time.sleep(1)


def step5_analysis():
    """Final analysis and verdict."""
    log("=" * 70)
    log("STEP 5: Final Analysis")
    log("=" * 70)

    findings = []

    s1 = state["steps"].get("step1", {})
    s2 = state["steps"].get("step2", {})
    s3 = state["steps"].get("step3", {})
    s4 = state["steps"].get("step4", {})

    findings.append(f"KFD queue: {s1.get('status', 'N/A')}")
    findings.append(f"PM4 scratch write: {s2.get('status', 'N/A')}")

    if "writable" in s3:
        findings.append(f"PM4 IC writable: {s3['writable']}")
        findings.append(f"PM4 IC non-writable: {s3['non_writable']}")
    else:
        findings.append(f"PM4 IC test: {s3.get('status', 'N/A')}")

    if s4.get("redirect_success"):
        verdict = "FIRMWARE REDIRECT ACHIEVED"
    elif s4.get("base_lo_accepted") and s4.get("base_hi_accepted"):
        verdict = "IC_BASE writable but PRIME failed — cache may need different trigger"
    elif s4.get("status") == "NO_PATH":
        verdict = "NO REDIRECT PATH — neither PM4 nor debugfs gives both BASE+OP_CNTL on any engine"
    else:
        verdict = f"PARTIAL — {s4.get('status', 'N/A')}"

    findings.append(f"VERDICT: {verdict}")
    for f_line in findings:
        log(f"  {f_line}")

    state["steps"]["step5"] = {
        "findings": findings,
        "verdict": verdict,
        "stats": {
            "writes_attempted": state["writes_attempted"],
            "writes_confirmed": state["writes_confirmed"],
            "writes_rejected": state["writes_rejected"],
            "health_checks": state["health_checks"],
        }
    }
    state["completed"] = datetime.now().isoformat()
    state["temperature_C"] = get_temp()
    save()


def main():
    log(f"z2349: PM4 IC Write Test")
    log(f"PID: {os.getpid()}")
    log(f"Temperature: {get_temp():.1f}C")
    log("")

    step0()
    ctx = step1_kfd_setup()

    if ctx and ctx.get("queue_id") is not None:
        pm4_works = step2_pm4_scratch_test(ctx)
        if pm4_works:
            engines = step3_pm4_ic_write_sweep(ctx)
            step4_me_redirect(ctx, engines)
        else:
            log("PM4 scratch test failed — queue may not be functional")
            # Still try IC writes in case the scratch test was misleading
            engines = step3_pm4_ic_write_sweep(ctx)
            if any(e.get("base") and e.get("op_cntl") for e in engines.values()):
                step4_me_redirect(ctx, engines)
    else:
        log("\nKFD queue creation FAILED — trying debugfs-only hybrid approach")
        log("(ME has debugfs OP_CNTL, CPC has debugfs BASE — testing PM4 alternatives)")

        # Even without a queue, document what we know
        state["steps"]["step2"] = {"status": "SKIP", "reason": "no queue"}
        state["steps"]["step3"] = {"status": "SKIP", "reason": "no queue"}

        # Try the hybrid approach with debugfs only
        log("\nStep 4 (debugfs hybrid): ME OP_CNTL + CPC BASE")
        log("  From z2348b: ME has OP_CNTL, CPC has BASE_LO+BASE_HI")
        log("  These are DIFFERENT engines — redirect requires SAME engine")
        log("  Exploring: Can we use CPC BASE to load code, then trigger CPC somehow?")

        # Test: write CPC BASE to our VRAM, then try to trigger CPC cache refresh
        # via workload submission or other mechanism
        state["steps"]["step4"] = {
            "status": "DEBUGFS_ONLY",
            "note": "ME has OP_CNTL, CPC has BASE — different engines, redirect needs same engine",
            "next_steps": [
                "Fix KFD alloc (ioctl number might differ on this kernel)",
                "Try libdrm_amdgpu for PM4 submission",
                "Try amdgpu_cs_submit via DRM render node",
                "Check if compute dispatch triggers CPC IC re-prime",
            ]
        }

    step5_analysis()
    log(f"\nALL DONE. Results: {JSON_OUT}")
    log(f"Final temp: {get_temp():.1f}C")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}")
        import traceback
        log(traceback.format_exc())
        save()
    finally:
        log_fh.close()
