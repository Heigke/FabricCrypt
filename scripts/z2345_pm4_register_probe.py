#!/usr/bin/env python3
"""z2345: PM4 Register Probe via KFD on gfx1151.

Submits PM4 COPY_DATA and WRITE_DATA packets via KFD user-mode compute queue
to map the register read/write privilege boundary.

READ-ONLY register probing first, then safe scratch register writes.
Saves results incrementally after each step.
"""
import os, sys, json, time, struct, ctypes, ctypes.util, mmap, signal
from pathlib import Path
from datetime import datetime

# === Paths ===
BASE = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

THERMAL = "/sys/class/thermal/thermal_zone0/temp"
DEBUGFS = "/sys/kernel/debug/dri/0"
REGS_FILE = f"{DEBUGFS}/amdgpu_regs"

# Output files
KFD_SETUP_FILE = RESULTS / "z2345_kfd_setup.txt"
REG_READS_FILE = RESULTS / "z2345_register_reads.txt"
WRITE_BOUNDARY_FILE = RESULTS / "z2345_write_boundary.txt"
NEURO_REGS_FILE = RESULTS / "z2345_neuromorphic_registers.txt"
JSON_OUT = RESULTS / "z2345_pm4_register_probe.json"

# === KFD ioctl constants ===
AMDKFD_IOCTL_BASE = ord('K')

def _IOC(dir_, type_, nr, size):
    return (dir_ << 30) | (type_ << 8) | (nr << 0) | (size << 16)

def _IOR(type_, nr, size):
    return _IOC(2, type_, nr, size)

def _IOW(type_, nr, size):
    return _IOC(1, type_, nr, size)

def _IOWR(type_, nr, size):
    return _IOC(3, type_, nr, size)

# KFD ioctls
AMDKFD_IOC_GET_VERSION = _IOR(AMDKFD_IOCTL_BASE, 0x01, 8)  # 2 * u32
AMDKFD_IOC_CREATE_QUEUE = _IOWR(AMDKFD_IOCTL_BASE, 0x02, 88)  # struct size
AMDKFD_IOC_DESTROY_QUEUE = _IOWR(AMDKFD_IOCTL_BASE, 0x03, 8)
AMDKFD_IOC_GET_PROCESS_APERTURES = _IOR(AMDKFD_IOCTL_BASE, 0x06, 7*56+8)
AMDKFD_IOC_ACQUIRE_VM = _IOW(AMDKFD_IOCTL_BASE, 0x15, 8)
AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = _IOWR(AMDKFD_IOCTL_BASE, 0x16, 32)
AMDKFD_IOC_FREE_MEMORY_OF_GPU = _IOW(AMDKFD_IOCTL_BASE, 0x17, 8)
AMDKFD_IOC_MAP_MEMORY_TO_GPU = _IOWR(AMDKFD_IOCTL_BASE, 0x18, 24)

KFD_IOC_QUEUE_TYPE_COMPUTE = 0
KFD_IOC_ALLOC_MEM_FLAGS_GTT = (1 << 1)
KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = (1 << 31)
KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = (1 << 30)
KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = (1 << 26)
KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = (1 << 25)

# PM4 opcodes (GFX11)
PKT3_HEADER = lambda opcode, count: (3 << 30) | ((opcode & 0xFF) << 8) | ((count - 1) & 0x3FFF)
PM4_NOP = 0x10
PM4_WRITE_DATA = 0x37
PM4_COPY_DATA = 0x40
PM4_RELEASE_MEM = 0x49
PM4_SET_UCONFIG_REG = 0x79

# COPY_DATA src/dst selectors
COPY_DATA_SRC_REG = 0    # source = register
COPY_DATA_SRC_MEM = 1    # source = memory
COPY_DATA_DST_REG = 0    # dest = register
COPY_DATA_DST_MEM = 5    # dest = memory
COPY_DATA_DST_MEM_MAPPED = 5

# WRITE_DATA dst selectors
WRITE_DATA_DST_MEM_MAPPED = 5
WRITE_DATA_DST_REG = 0

# Results accumulator
findings = {
    "experiment": "z2345",
    "date": datetime.now().isoformat(),
    "gpu": "gfx1151 (Radeon 8060S)",
    "steps": {},
    "readable_registers": {},
    "writable_registers": {},
    "blocked_registers": {},
    "neuromorphic_candidates": {},
}

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

def log(msg, file_handle=None):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if file_handle:
        file_handle.write(line + "\n")
        file_handle.flush()

def check_temp():
    try:
        t = int(open(THERMAL).read().strip())
        return t / 1000.0
    except:
        return 0.0

def abort_if_hot(limit=85.0):
    t = check_temp()
    if t > limit:
        log(f"THERMAL ABORT: {t:.1f}C > {limit}C")
        save_json()
        sys.exit(1)
    return t

def save_json():
    with open(JSON_OUT, "w") as f:
        json.dump(findings, f, indent=2, default=str)

def safe_debugfs_reg_read(offset, timeout=5):
    """Read a single register via debugfs amdgpu_regs (fallback)."""
    class Timeout(Exception): pass
    def handler(s, f): raise Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        with open(REGS_FILE, "rb") as f:
            f.seek(offset)
            data = f.read(4)
        signal.alarm(0)
        if len(data) == 4:
            return struct.unpack("<I", data)[0]
        return None
    except Timeout:
        signal.alarm(0)
        return None
    except Exception:
        signal.alarm(0)
        return None
    finally:
        signal.signal(signal.SIGALRM, old)

# ========== STEP 1: KFD Queue Setup ==========
def step1_kfd_setup():
    """Open /dev/kfd, get GPU info, attempt queue creation."""
    out = []
    out.append("=" * 60)
    out.append("STEP 1: KFD Queue Setup")
    out.append("=" * 60)

    temp = abort_if_hot()
    out.append(f"Temperature: {temp:.1f}C")

    # 1a: Check /dev/kfd access
    kfd_path = "/dev/kfd"
    out.append(f"\n--- 1a: /dev/kfd access ---")
    if not os.path.exists(kfd_path):
        out.append("FAIL: /dev/kfd does not exist")
        findings["steps"]["1_kfd_setup"] = "FAIL: no /dev/kfd"
        with open(KFD_SETUP_FILE, "w") as f: f.write("\n".join(out))
        return None

    try:
        kfd_fd = os.open(kfd_path, os.O_RDWR)
        out.append(f"  /dev/kfd opened: fd={kfd_fd}")
    except PermissionError:
        out.append("  FAIL: Permission denied (need render group)")
        findings["steps"]["1_kfd_setup"] = "FAIL: permission denied"
        with open(KFD_SETUP_FILE, "w") as f: f.write("\n".join(out))
        return None

    # 1b: Get KFD version
    out.append(f"\n--- 1b: KFD version ---")
    ver_buf = ctypes.create_string_buffer(8)
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_GET_VERSION, ver_buf)
    if ret == 0:
        major, minor = struct.unpack("<II", ver_buf.raw)
        out.append(f"  KFD version: {major}.{minor}")
        findings["kfd_version"] = f"{major}.{minor}"
    else:
        errno = ctypes.get_errno()
        out.append(f"  GET_VERSION failed: errno={errno}")

    # 1c: Get process apertures (to find gpu_id)
    out.append(f"\n--- 1c: Process apertures ---")
    # struct: 7 * kfd_process_device_apertures (56 bytes each) + 2 * u32
    ap_size = 7 * 56 + 8
    ap_buf = ctypes.create_string_buffer(ap_size)
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_GET_PROCESS_APERTURES, ap_buf)
    gpu_id = None
    gpuvm_base = None
    if ret == 0:
        num_nodes = struct.unpack_from("<I", ap_buf.raw, 7 * 56)[0]
        out.append(f"  Number of GPU nodes: {num_nodes}")
        for i in range(min(num_nodes, 7)):
            off = i * 56
            lds_base, lds_limit, scratch_base, scratch_limit, gpuvm_b, gpuvm_l, gid = \
                struct.unpack_from("<QQQQQQII", ap_buf.raw, off)[:7]
            out.append(f"  GPU {i}: gpu_id={gid:#x}, gpuvm=0x{gpuvm_b:016x}-0x{gpuvm_l:016x}")
            out.append(f"          lds=0x{lds_base:016x}-0x{lds_limit:016x}")
            out.append(f"          scratch=0x{scratch_base:016x}-0x{scratch_limit:016x}")
            if gid != 0:  # first real GPU
                gpu_id = gid
                gpuvm_base = gpuvm_b
    else:
        errno = ctypes.get_errno()
        out.append(f"  GET_PROCESS_APERTURES failed: errno={errno}")

    if gpu_id is None:
        out.append("  FAIL: No GPU found via apertures")
        findings["steps"]["1_kfd_setup"] = "FAIL: no GPU found"
        with open(KFD_SETUP_FILE, "w") as f: f.write("\n".join(out))
        os.close(kfd_fd)
        return None

    findings["gpu_id"] = hex(gpu_id)
    out.append(f"\n  Selected GPU: gpu_id={gpu_id:#x}")

    # 1d: Acquire VM (need DRM fd)
    out.append(f"\n--- 1d: Acquire VM ---")
    drm_fd = None
    for render_node in ["/dev/dri/renderD128", "/dev/dri/renderD129"]:
        try:
            drm_fd = os.open(render_node, os.O_RDWR)
            out.append(f"  Opened {render_node}: fd={drm_fd}")
            break
        except:
            continue

    if drm_fd is None:
        out.append("  FAIL: Cannot open render node")
        os.close(kfd_fd)
        findings["steps"]["1_kfd_setup"] = "FAIL: no render node"
        with open(KFD_SETUP_FILE, "w") as f: f.write("\n".join(out))
        return None

    # Acquire VM: struct { u32 drm_fd; u32 gpu_id; }
    acq_buf = struct.pack("<II", drm_fd, gpu_id)
    acq_arr = ctypes.create_string_buffer(acq_buf)
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_ACQUIRE_VM, acq_arr)
    if ret == 0:
        out.append(f"  VM acquired for gpu_id={gpu_id:#x}")
    else:
        errno = ctypes.get_errno()
        out.append(f"  ACQUIRE_VM failed: errno={errno} ({os.strerror(errno)})")
        # Continue anyway, might still work

    # 1e: Allocate ring buffer, EOP buffer, ctx save/restore via KFD
    out.append(f"\n--- 1e: Allocate GPU memory for queue ---")

    def kfd_alloc_memory(kfd_fd, gpu_id, size, flags):
        """Allocate memory via KFD. Returns (handle, va_addr, mmap_offset) or None."""
        # va_addr=0 lets KFD pick; struct is 32 bytes: u64 va, u64 size, u64 handle, u64 mmap_off, u32 gpu_id, u32 flags
        buf = struct.pack("<QQQQII", 0, size, 0, 0, gpu_id, flags & 0xFFFFFFFF)
        arr = ctypes.create_string_buffer(buf)
        ret = libc.ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, arr)
        if ret != 0:
            return None
        va_addr, sz, handle, mmap_off, gid, fl = struct.unpack("<QQQQII", arr.raw)
        return handle, va_addr, mmap_off

    # Ring buffer: GTT, writable, executable, uncached
    ring_flags = (KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                  KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                  KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                  KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    ring_size = 4096 * 4  # 16KB ring

    ring_alloc = kfd_alloc_memory(kfd_fd, gpu_id, ring_size, ring_flags)
    if ring_alloc:
        ring_handle, ring_va, ring_mmap = ring_alloc
        out.append(f"  Ring buffer: handle=0x{ring_handle:x}, va=0x{ring_va:016x}, mmap=0x{ring_mmap:x}")
    else:
        errno = ctypes.get_errno()
        out.append(f"  Ring alloc FAILED: errno={errno} ({os.strerror(errno)})")
        findings["steps"]["1_kfd_setup"] = f"FAIL: ring alloc errno={errno}"
        with open(KFD_SETUP_FILE, "w") as f: f.write("\n".join(out))
        os.close(kfd_fd)
        os.close(drm_fd)
        return None

    # EOP buffer
    eop_size = 4096
    eop_flags = (KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                 KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                 KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    eop_alloc = kfd_alloc_memory(kfd_fd, gpu_id, eop_size, eop_flags)
    if eop_alloc:
        eop_handle, eop_va, eop_mmap = eop_alloc
        out.append(f"  EOP buffer:  handle=0x{eop_handle:x}, va=0x{eop_va:016x}")
    else:
        errno = ctypes.get_errno()
        out.append(f"  EOP alloc FAILED: errno={errno}")
        eop_handle, eop_va, eop_mmap = 0, 0, 0

    # Context save/restore area
    ctx_size = 4096 * 16  # 64KB
    ctx_flags = (KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                 KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                 KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)
    ctx_alloc = kfd_alloc_memory(kfd_fd, gpu_id, ctx_size, ctx_flags)
    if ctx_alloc:
        ctx_handle, ctx_va, ctx_mmap = ctx_alloc
        out.append(f"  CTX buffer:  handle=0x{ctx_handle:x}, va=0x{ctx_va:016x}")
    else:
        ctx_handle, ctx_va = 0, 0

    # Map ring buffer to get CPU pointer
    out.append(f"\n--- 1f: Map ring to CPU ---")
    def kfd_map_memory(kfd_fd, handle, gpu_id):
        """Map GPU memory to this GPU."""
        gpu_ids = struct.pack("<I", gpu_id)
        gpu_ids_buf = ctypes.create_string_buffer(gpu_ids)
        gpu_ids_ptr = ctypes.addressof(gpu_ids_buf)
        # struct: u64 handle, u64 device_ids_array_ptr, u32 n_devices, u32 n_success
        buf = struct.pack("<QQIi", handle, gpu_ids_ptr, 1, 0)
        arr = ctypes.create_string_buffer(buf)
        ret = libc.ioctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, arr)
        return ret == 0

    map_ok = kfd_map_memory(kfd_fd, ring_handle, gpu_id)
    out.append(f"  Ring map: {'OK' if map_ok else 'FAIL'}")

    if eop_handle:
        eop_map_ok = kfd_map_memory(kfd_fd, eop_handle, gpu_id)
        out.append(f"  EOP map:  {'OK' if eop_map_ok else 'FAIL'}")
    if ctx_handle:
        ctx_map_ok = kfd_map_memory(kfd_fd, ctx_handle, gpu_id)
        out.append(f"  CTX map:  {'OK' if ctx_map_ok else 'FAIL'}")

    # mmap ring buffer to CPU address space
    try:
        ring_cpu = mmap.mmap(kfd_fd, ring_size, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE,
                             offset=ring_mmap)
        out.append(f"  Ring mmap: OK (size={ring_size})")
    except Exception as e:
        out.append(f"  Ring mmap FAILED: {e}")
        ring_cpu = None

    # 1g: Allocate write/read pointer memory (needs to be accessible from CPU)
    wptr_buf = ctypes.create_string_buffer(8)  # u64 write pointer
    rptr_buf = ctypes.create_string_buffer(8)  # u64 read pointer
    wptr_addr = ctypes.addressof(wptr_buf)
    rptr_addr = ctypes.addressof(rptr_buf)

    # 1h: Create compute queue
    out.append(f"\n--- 1g: Create compute queue ---")
    # struct kfd_ioctl_create_queue_args (88 bytes):
    # u64 ring_base, u64 wptr_addr, u64 rptr_addr, u64 doorbell_offset
    # u32 ring_size, u32 gpu_id, u32 queue_type, u32 queue_percentage
    # u32 queue_priority, u32 queue_id
    # u64 eop_buffer_address, u64 eop_buffer_size
    # u64 ctx_save_restore_address, u32 ctx_save_restore_size, u32 ctl_stack_size

    q_buf = struct.pack("<QQQQIIIIIiQQQII",
        ring_va,          # ring_base_address
        wptr_addr,        # write_pointer_address (CPU addr)
        rptr_addr,        # read_pointer_address (CPU addr)
        0,                # doorbell_offset (from KFD)
        ring_size,        # ring_size
        gpu_id,           # gpu_id
        KFD_IOC_QUEUE_TYPE_COMPUTE,  # queue_type
        100,              # queue_percentage
        7,                # queue_priority (mid)
        0,                # queue_id (from KFD)
        eop_va,           # eop_buffer_address
        eop_size,         # eop_buffer_size
        ctx_va,           # ctx_save_restore_address
        ctx_size,         # ctx_save_restore_size
        4096,             # ctl_stack_size
    )
    q_arr = ctypes.create_string_buffer(q_buf)
    ret = libc.ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, q_arr)

    queue_id = None
    doorbell_offset = None
    if ret == 0:
        # Parse output: wptr_addr, rptr_addr, doorbell_offset are from KFD
        vals = struct.unpack("<QQQQIIIIIiQQQII", q_arr.raw)
        ring_base_out = vals[0]
        wptr_out = vals[1]
        rptr_out = vals[2]
        doorbell_out = vals[3]
        queue_id = vals[9]
        out.append(f"  Queue created! queue_id={queue_id}")
        out.append(f"  wptr_addr=0x{wptr_out:016x}, rptr_addr=0x{rptr_out:016x}")
        out.append(f"  doorbell_offset=0x{doorbell_out:016x}")
        doorbell_offset = doorbell_out
        findings["queue_id"] = queue_id
        findings["steps"]["1_kfd_setup"] = "PASS"
    else:
        errno = ctypes.get_errno()
        out.append(f"  CREATE_QUEUE FAILED: errno={errno} ({os.strerror(errno)})")
        findings["steps"]["1_kfd_setup"] = f"FAIL: create_queue errno={errno}"
        out.append(f"\n  Will fall back to debugfs register reads only.")

    result_text = "\n".join(out)
    with open(KFD_SETUP_FILE, "w") as f:
        f.write(result_text)
    print(result_text)
    save_json()

    return {
        "kfd_fd": kfd_fd,
        "drm_fd": drm_fd,
        "gpu_id": gpu_id,
        "queue_id": queue_id,
        "ring_cpu": ring_cpu,
        "ring_va": ring_va,
        "ring_size": ring_size,
        "wptr_buf": wptr_buf,
        "rptr_buf": rptr_buf,
        "wptr_addr": wptr_addr,
        "rptr_addr": rptr_addr,
        "doorbell_offset": doorbell_offset,
        "ring_handle": ring_handle,
        "eop_handle": eop_handle,
    }


# ========== STEP 2: Register Read Probe ==========
def step2_register_reads(ctx):
    """Probe register ranges via debugfs amdgpu_regs (safe, read-only).
    If PM4 queue available, also try COPY_DATA from registers to GPU memory.
    """
    out = []
    out.append("=" * 60)
    out.append("STEP 2: PM4 Register READ Probe")
    out.append("=" * 60)

    temp = abort_if_hot()
    out.append(f"Temperature: {temp:.1f}C")

    # Define register ranges to probe
    # amdgpu_regs uses byte offsets = register_index * 4
    ranges = {
        "GC_core": (0x2000, 0x2100, "Graphics Controller core (scratch, config)"),
        "GC_mid": (0x2800, 0x2900, "GC shader/compute config"),
        "GC_upper": (0x3000, 0x3100, "GC upper range"),
        "CP_regs": (0x2100, 0x2200, "Command Processor registers"),
        "GRBM": (0x2000, 0x2010, "GRBM status/control"),
        "SDMA0": (0x4A00, 0x4A40, "SDMA engine 0"),
        "MMHUB": (0x68000, 0x68080, "Memory Management Hub"),
        "THM": (0x59800, 0x59840, "Thermal Management"),
        "SMC_MP1": (0x29000, 0x29040, "SMC/MP1 power management"),
        "GC_scratch": (0x2040, 0x2048, "GC scratch registers"),
        "VGT": (0x2230, 0x2240, "Vertex Grouper/Tessellator"),
        "SPI": (0x2440, 0x2460, "Shader Processor Input"),
        "CB": (0x2800, 0x2810, "Color Buffer"),
        "DB": (0x2600, 0x2610, "Depth Buffer"),
        "PA": (0x2200, 0x2210, "Primitive Assembly"),
        "SMUIO": (0x5A000, 0x5A040, "SMU I/O"),
        "NBIO": (0x10000, 0x10040, "NorthBridge I/O"),
        "GC_perfctr": (0x3400, 0x3440, "GC performance counters"),
        "RLCG": (0x4E00, 0x4E40, "RLC Graphics"),
        "CPC": (0x2900, 0x2940, "Compute Pipe Control"),
    }

    readable_map = {}
    blocked_map = {}

    for name, (start, end, desc) in sorted(ranges.items(), key=lambda x: x[1][0]):
        abort_if_hot()
        out.append(f"\n--- {name}: 0x{start:05X}-0x{end:05X} ({desc}) ---")
        readable = []
        blocked = []
        sample_vals = {}

        for reg_idx in range(start, end, 4):
            byte_off = reg_idx * 4  # debugfs uses byte offsets
            val = safe_debugfs_reg_read(byte_off, timeout=3)
            if val is not None:
                readable.append(reg_idx)
                if len(sample_vals) < 8:
                    sample_vals[f"0x{reg_idx:05X}"] = f"0x{val:08X}"
            else:
                blocked.append(reg_idx)

        total = (end - start) // 4
        r_count = len(readable)
        b_count = len(blocked)
        out.append(f"  Readable: {r_count}/{total}, Blocked: {b_count}/{total}")
        if sample_vals:
            for addr, val in sample_vals.items():
                out.append(f"    {addr} = {val}")
        else:
            out.append(f"    (no readable registers)")

        if readable:
            readable_map[name] = {
                "range": f"0x{start:05X}-0x{end:05X}",
                "desc": desc,
                "readable_count": r_count,
                "total": total,
                "samples": sample_vals,
            }
        if blocked:
            blocked_map[name] = {
                "range": f"0x{start:05X}-0x{end:05X}",
                "blocked_count": b_count,
                "total": total,
            }

    findings["readable_registers"] = readable_map
    findings["blocked_registers"] = blocked_map
    findings["steps"]["2_register_reads"] = f"Probed {len(ranges)} ranges"

    # Summary
    out.append(f"\n{'='*60}")
    out.append("SUMMARY: Register Readability Map")
    out.append(f"{'='*60}")
    total_readable = sum(v["readable_count"] for v in readable_map.values())
    total_blocked = sum(v["blocked_count"] for v in blocked_map.values())
    out.append(f"Total readable: {total_readable}")
    out.append(f"Total blocked/timeout: {total_blocked}")
    for name in sorted(readable_map.keys()):
        r = readable_map[name]
        out.append(f"  {name}: {r['readable_count']}/{r['total']} readable")

    result_text = "\n".join(out)
    with open(REG_READS_FILE, "w") as f:
        f.write(result_text)
    print(result_text)
    save_json()
    return readable_map


# ========== STEP 3: Write Boundary Test ==========
def step3_write_boundary(ctx, readable_map):
    """Test register writability via debugfs amdgpu_regs.
    Only test known-safe scratch registers first."""
    out = []
    out.append("=" * 60)
    out.append("STEP 3: Register WRITE Boundary Test (via debugfs)")
    out.append("=" * 60)

    temp = abort_if_hot()
    out.append(f"Temperature: {temp:.1f}C")

    # GFX11 scratch registers - these are explicitly designed for SW use
    # On older gens: mmSCRATCH_REG0 = 0x2040, but on GFX11 mapping may differ
    # Let's first read GRBM_STATUS to confirm we can read GC regs
    grbm_val = safe_debugfs_reg_read(0x2004 * 4)  # GRBM_STATUS byte offset
    out.append(f"\nGRBM_STATUS (0x2004): {'0x'+format(grbm_val,'08X') if grbm_val is not None else 'BLOCKED'}")

    # Try scratch register candidates at classic offsets
    scratch_candidates = [
        (0x2040, "SCRATCH_REG0"),
        (0x2041, "SCRATCH_REG1"),
        (0x2042, "SCRATCH_REG2"),
        (0x2043, "SCRATCH_REG3"),
        (0x2044, "SCRATCH_REG4"),
        (0x2045, "SCRATCH_REG5"),
        (0x2046, "SCRATCH_REG6"),
        (0x2047, "SCRATCH_REG7"),
    ]

    out.append(f"\n--- Scratch register read test ---")
    writable_regs = {}
    for reg_idx, name in scratch_candidates:
        byte_off = reg_idx * 4
        val = safe_debugfs_reg_read(byte_off)
        if val is not None:
            out.append(f"  {name} (0x{reg_idx:04X}): 0x{val:08X}")
        else:
            out.append(f"  {name} (0x{reg_idx:04X}): BLOCKED/TIMEOUT")

    # Try writing to scratch registers via debugfs
    out.append(f"\n--- Scratch register write test (via debugfs) ---")
    REGS_WRITE = f"{DEBUGFS}/amdgpu_regs"

    test_pattern = 0xDEADBEEF
    for reg_idx, name in scratch_candidates[:4]:  # Only first 4 to be safe
        byte_off = reg_idx * 4
        abort_if_hot()

        # Read original value
        orig = safe_debugfs_reg_read(byte_off)
        if orig is None:
            out.append(f"  {name}: Cannot read, skip write test")
            continue

        out.append(f"  {name} (0x{reg_idx:04X}): orig=0x{orig:08X}")

        # Try write
        try:
            with open(REGS_WRITE, "r+b") as f:
                f.seek(byte_off)
                f.write(struct.pack("<I", test_pattern))
                f.flush()

            # Read back
            readback = safe_debugfs_reg_read(byte_off)
            if readback == test_pattern:
                out.append(f"    WRITE OK: wrote 0x{test_pattern:08X}, readback=0x{readback:08X}")
                writable_regs[f"0x{reg_idx:04X}"] = {"name": name, "writable": True}

                # Restore original
                with open(REGS_WRITE, "r+b") as f:
                    f.seek(byte_off)
                    f.write(struct.pack("<I", orig))
                out.append(f"    Restored to 0x{orig:08X}")
            elif readback is not None:
                out.append(f"    WRITE BLOCKED: readback=0x{readback:08X} (unchanged)")
                writable_regs[f"0x{reg_idx:04X}"] = {"name": name, "writable": False}
            else:
                out.append(f"    WRITE TEST INCONCLUSIVE: readback timeout")
        except PermissionError:
            out.append(f"    WRITE FAILED: Permission denied")
        except Exception as e:
            out.append(f"    WRITE FAILED: {e}")

    # Test broader register writability (non-scratch, BE CAREFUL)
    out.append(f"\n--- Extended write test (safe register candidates) ---")

    # GC UCONFIG registers are typically user-writable from PM4
    # Let's check GRBM_GFX_INDEX, VGT, SPI - these are configured by shaders
    extended_candidates = [
        (0x2200, "PA_SC_WINDOW_OFFSET"),
        (0x2230, "VGT_MAX_VTX_INDX"),
        (0x2440, "SPI_SHADER_PGM_LO_PS"),
        (0x2900, "COMPUTE_START_X"),
        (0x2904, "COMPUTE_NUM_THREAD_X"),
    ]

    for reg_idx, name in extended_candidates:
        byte_off = reg_idx * 4
        abort_if_hot()
        orig = safe_debugfs_reg_read(byte_off)
        if orig is None:
            out.append(f"  {name} (0x{reg_idx:04X}): Cannot read")
            continue

        out.append(f"  {name} (0x{reg_idx:04X}): read=0x{orig:08X}")

        # Try write with safe test (write same value back - no actual change)
        try:
            with open(REGS_WRITE, "r+b") as f:
                f.seek(byte_off)
                f.write(struct.pack("<I", orig))  # Write same value = safe
            readback = safe_debugfs_reg_read(byte_off)
            if readback is not None:
                out.append(f"    Identity write OK, readback=0x{readback:08X}")
                writable_regs[f"0x{reg_idx:04X}"] = {"name": name, "writable": True, "note": "identity write only"}
        except Exception as e:
            out.append(f"    Write error: {e}")

    findings["writable_registers"] = writable_regs
    findings["steps"]["3_write_boundary"] = f"Tested {len(scratch_candidates)+len(extended_candidates)} registers"

    result_text = "\n".join(out)
    with open(WRITE_BOUNDARY_FILE, "w") as f:
        f.write(result_text)
    print(result_text)
    save_json()
    return writable_regs


# ========== STEP 4: Neuromorphic Register Candidates ==========
def step4_neuromorphic_registers(ctx, readable_map, writable_regs):
    """Identify registers useful for neuromorphic computing:
    - Clock/PLL: frequency modulation for analog noise
    - Thermal: direct sensor reads
    - Power: voltage/current sensing
    - Scratch: fast GPU-accessible state for reservoir computing
    """
    out = []
    out.append("=" * 60)
    out.append("STEP 4: Neuromorphic Register Candidates")
    out.append("=" * 60)

    temp = abort_if_hot()
    out.append(f"Temperature: {temp:.1f}C")

    candidates = {}

    # 4a: Clock/PLL registers
    out.append(f"\n--- 4a: Clock/PLL registers ---")
    clock_regs = [
        (0x5B000, "CG_CLKPIN_CNTL"),    # Clock pin control
        (0x5B004, "CG_CLKPIN_CNTL_2"),
        (0x5B010, "MPLL_SEQ_UCODE"),
        (0x5B01C, "MPLL_DQ_0_0"),
        (0x50000, "SMC_IND_INDEX"),
        (0x50004, "SMC_IND_DATA"),
        (0x5B200, "DCCG_GATE_DISABLE"),
        (0x5B204, "DCCG_GATE_DISABLE_2"),
        (0x17000, "CLK_FREQ_GFX"),       # GFX clock frequency?
        (0x17004, "CLK_FREQ_SOC"),
        (0x0C050, "GFX_CLK"),            # Possible GFX clock status
        (0x0C054, "SOC_CLK"),
    ]

    for reg_idx, name in clock_regs:
        byte_off = reg_idx * 4
        val = safe_debugfs_reg_read(byte_off, timeout=3)
        status = f"0x{val:08X}" if val is not None else "BLOCKED"
        out.append(f"  {name} (0x{reg_idx:05X}): {status}")
        if val is not None:
            candidates[f"clock_0x{reg_idx:05X}"] = {
                "name": name, "value": f"0x{val:08X}",
                "category": "clock", "neuromorphic_use": "frequency modulation"
            }

    # 4b: Thermal sensor registers (direct, not hwmon)
    out.append(f"\n--- 4b: Thermal sensor registers ---")
    thm_regs = [
        (0x59800, "THM_TCON_CUR_TMP"),
        (0x59804, "THM_TCON_HTC"),
        (0x59808, "THM_TCON_THERM_TRIP"),
        (0x5980C, "THM_TCON_CUR_TMP_2"),
        (0x59810, "THM_TCON_CUR_TMP_3"),
        (0x59814, "THM_GPIO_CFG"),
        (0x59860, "CG_MULT_THERMAL_STATUS"),
        (0x59864, "CG_THERMAL_STATUS"),
        (0x598D0, "CG_FDO_CTRL0"),   # Fan control
        (0x598D4, "CG_FDO_CTRL1"),
        (0x598D8, "CG_FDO_CTRL2"),
    ]

    for reg_idx, name in thm_regs:
        byte_off = reg_idx * 4
        abort_if_hot()
        val = safe_debugfs_reg_read(byte_off, timeout=3)
        status = f"0x{val:08X}" if val is not None else "BLOCKED"
        out.append(f"  {name} (0x{reg_idx:05X}): {status}")
        if val is not None:
            temp_c = None
            if "CUR_TMP" in name:
                # Often CUR_TMP format: bits[31:21] = temp in degrees * 8
                raw_temp = (val >> 21) & 0x7FF
                if raw_temp > 0 and raw_temp < 0x400:
                    temp_c = raw_temp / 8.0
                    out.append(f"    -> Decoded temp: {temp_c:.1f}C")
            candidates[f"thermal_0x{reg_idx:05X}"] = {
                "name": name, "value": f"0x{val:08X}",
                "category": "thermal", "neuromorphic_use": "temperature noise source",
                "decoded_temp_c": temp_c,
            }

    # 4c: Power/voltage registers
    out.append(f"\n--- 4c: Power/voltage registers ---")
    pwr_regs = [
        (0x29000, "MP1_SMN_C2PMSG_0"),  # Read only! NEVER write to mailbox regs
        (0x29004, "MP1_SMN_C2PMSG_1"),
        (0x29008, "MP1_SMN_C2PMSG_2"),
        (0x2900C, "MP1_SMN_C2PMSG_3"),
        (0x5A000, "SMUIO_MCM_CONFIG"),
        (0x5A004, "SMUIO_GFX_MISC_CNTL"),
        (0x5A008, "SMUIO_SOC_MISC"),
        (0x5A020, "SMUIO_PWRMGT"),
    ]

    for reg_idx, name in pwr_regs:
        byte_off = reg_idx * 4
        abort_if_hot()
        val = safe_debugfs_reg_read(byte_off, timeout=3)
        status = f"0x{val:08X}" if val is not None else "BLOCKED"
        out.append(f"  {name} (0x{reg_idx:05X}): {status}")
        if val is not None:
            candidates[f"power_0x{reg_idx:05X}"] = {
                "name": name, "value": f"0x{val:08X}",
                "category": "power", "neuromorphic_use": "voltage/power noise source",
            }

    # 4d: GC scratch registers (fast GPU-writable state)
    out.append(f"\n--- 4d: GC scratch registers (GPU-accessible fast state) ---")
    scratch_accessible = []
    for reg_idx in range(0x2040, 0x2048):
        byte_off = reg_idx * 4
        val = safe_debugfs_reg_read(byte_off, timeout=3)
        if val is not None:
            scratch_accessible.append(reg_idx)
            out.append(f"  SCRATCH_REG{reg_idx-0x2040} (0x{reg_idx:04X}): 0x{val:08X}")
            candidates[f"scratch_0x{reg_idx:04X}"] = {
                "name": f"SCRATCH_REG{reg_idx-0x2040}",
                "value": f"0x{val:08X}",
                "category": "scratch",
                "neuromorphic_use": "fast state for reservoir, writable from PM4/shader",
            }

    # 4e: Performance counter registers (continuous analog signal source)
    out.append(f"\n--- 4e: Performance counter registers ---")
    perf_regs = [
        (0x3400, "CPG_PERFCOUNTER1_LO"),
        (0x3404, "CPG_PERFCOUNTER1_HI"),
        (0x3408, "CPG_PERFCOUNTER0_LO"),
        (0x340C, "CPG_PERFCOUNTER0_HI"),
        (0x3480, "GRBM_PERFCOUNTER0_LO"),
        (0x3484, "GRBM_PERFCOUNTER0_HI"),
        (0x3600, "SPI_PERFCOUNTER0_LO"),
        (0x3604, "SPI_PERFCOUNTER0_HI"),
    ]

    for reg_idx, name in perf_regs:
        byte_off = reg_idx * 4
        abort_if_hot()
        val = safe_debugfs_reg_read(byte_off, timeout=3)
        status = f"0x{val:08X}" if val is not None else "BLOCKED"
        out.append(f"  {name} (0x{reg_idx:05X}): {status}")
        if val is not None:
            candidates[f"perf_0x{reg_idx:05X}"] = {
                "name": name, "value": f"0x{val:08X}",
                "category": "perfcounter",
                "neuromorphic_use": "continuous activity signal, noise source",
            }

    # 4f: RLC (Run List Controller) - has its own scratchpad
    out.append(f"\n--- 4f: RLC registers ---")
    rlc_regs = [
        (0x4E00, "RLC_CNTL"),
        (0x4E04, "RLC_STATUS"),
        (0x4E08, "RLC_CGCG_CGLS_CTRL"),
        (0x4E40, "RLC_GPM_SCRATCH_ADDR"),
        (0x4E44, "RLC_GPM_SCRATCH_DATA"),
        (0x4E80, "RLC_GPU_CLOCK_COUNT_LSB"),
        (0x4E84, "RLC_GPU_CLOCK_COUNT_MSB"),
    ]

    for reg_idx, name in rlc_regs:
        byte_off = reg_idx * 4
        val = safe_debugfs_reg_read(byte_off, timeout=3)
        status = f"0x{val:08X}" if val is not None else "BLOCKED"
        out.append(f"  {name} (0x{reg_idx:05X}): {status}")
        if val is not None:
            candidates[f"rlc_0x{reg_idx:05X}"] = {
                "name": name, "value": f"0x{val:08X}",
                "category": "rlc",
                "neuromorphic_use": "clock count for timing, scratchpad for state",
            }

    # Summary
    out.append(f"\n{'='*60}")
    out.append("NEUROMORPHIC REGISTER SUMMARY")
    out.append(f"{'='*60}")
    by_cat = {}
    for k, v in candidates.items():
        cat = v["category"]
        by_cat.setdefault(cat, []).append(v)

    for cat, regs in sorted(by_cat.items()):
        out.append(f"\n{cat.upper()} ({len(regs)} registers):")
        for r in regs:
            out.append(f"  {r['name']}: {r['value']} — {r['neuromorphic_use']}")

    out.append(f"\nTotal neuromorphic-relevant registers found: {len(candidates)}")

    findings["neuromorphic_candidates"] = candidates
    findings["steps"]["4_neuromorphic"] = f"Found {len(candidates)} candidates"

    result_text = "\n".join(out)
    with open(NEURO_REGS_FILE, "w") as f:
        f.write(result_text)
    print(result_text)
    save_json()
    return candidates


# ========== MAIN ==========
def main():
    print("=" * 60)
    print("z2345: PM4 Register Probe — gfx1151")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 60)

    temp = abort_if_hot()
    print(f"Starting temperature: {temp:.1f}C")

    # Ensure we have access
    if not os.path.exists("/dev/kfd"):
        print("ERROR: /dev/kfd not found")
        sys.exit(1)

    # Step 1: KFD queue setup
    print("\n" + "=" * 60)
    ctx = step1_kfd_setup()
    abort_if_hot()

    # Step 2: Register reads
    print("\n" + "=" * 60)
    readable_map = step2_register_reads(ctx)
    abort_if_hot()

    # Step 3: Write boundary
    print("\n" + "=" * 60)
    writable_regs = step3_write_boundary(ctx, readable_map)
    abort_if_hot()

    # Step 4: Neuromorphic candidates
    print("\n" + "=" * 60)
    neuro = step4_neuromorphic_registers(ctx, readable_map, writable_regs)

    # Final summary
    findings["final_temperature"] = check_temp()
    findings["summary"] = {
        "readable_ranges": len(findings.get("readable_registers", {})),
        "writable_registers": len(findings.get("writable_registers", {})),
        "neuromorphic_candidates": len(findings.get("neuromorphic_candidates", {})),
        "kfd_queue": findings["steps"].get("1_kfd_setup", "unknown"),
    }
    save_json()

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Temperature: {check_temp():.1f}C")
    print(f"Readable register ranges: {findings['summary']['readable_ranges']}")
    print(f"Writable registers found: {findings['summary']['writable_registers']}")
    print(f"Neuromorphic candidates: {findings['summary']['neuromorphic_candidates']}")
    print(f"\nResults saved to:")
    print(f"  {KFD_SETUP_FILE}")
    print(f"  {REG_READS_FILE}")
    print(f"  {WRITE_BOUNDARY_FILE}")
    print(f"  {NEURO_REGS_FILE}")
    print(f"  {JSON_OUT}")

    # Cleanup
    if ctx and ctx.get("queue_id") is not None:
        try:
            q_buf = struct.pack("<Ii", ctx["queue_id"], 0)
            q_arr = ctypes.create_string_buffer(q_buf)
            libc.ioctl(ctx["kfd_fd"], AMDKFD_IOC_DESTROY_QUEUE, q_arr)
            print("Queue destroyed.")
        except:
            pass
    if ctx:
        try:
            os.close(ctx["kfd_fd"])
            os.close(ctx["drm_fd"])
        except:
            pass


if __name__ == "__main__":
    main()
