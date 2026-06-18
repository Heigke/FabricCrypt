#!/usr/bin/env python3
"""KFD queue creation with full HSA-equivalent prerequisite sequence.

Matches the ioctl sequence observed via strace of libhsa-runtime64:
  GET_VERSION → GET_PROCESS_APERTURES_NEW → ACQUIRE_VM → SET_MEMORY_POLICY →
  ALLOC+MAP (initial) → RUNTIME_ENABLE → SET_XNACK_MODE → CREATE_EVENT ×N →
  SET_SCRATCH_BACKING_VA → SET_TRAP_HANDLER → ALLOC+MAP (ring,eop,ctx) →
  CREATE_QUEUE
"""
import os, struct, ctypes, ctypes.util, mmap, errno

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
K = ord('K')

# ── ioctl direction helpers ──────────────────────────────────────────
def _IOC(d, t, nr, sz): return (d << 30) | (t << 8) | nr | (sz << 16)
def _IO(t, nr):         return _IOC(0, t, nr, 0)
def _IOR(t, nr, sz):    return _IOC(2, t, nr, sz)
def _IOW(t, nr, sz):    return _IOC(1, t, nr, sz)
def _IOWR(t, nr, sz):   return _IOC(3, t, nr, sz)

# ── KFD ioctls (kernel 6.14) ────────────────────────────────────────
IOC_VER                = _IOR(K, 0x01, 8)
IOC_CREATE_Q           = _IOWR(K, 0x02, 96)
IOC_DESTROY_Q          = _IOWR(K, 0x03, 8)
IOC_SET_MEM_POLICY     = _IOW(K, 0x04, 32)   # 4×u64 + 4×u32 → but actually 2×u64 + 4×u32 = 32
IOC_GET_CLOCK          = _IOR(K, 0x05, 16)
IOC_APER_OLD           = _IOR(K, 0x06, 400)
IOC_CREATE_EVENT       = _IOWR(K, 0x08, 32)  # 1×u64 + 6×u32 = 32
IOC_DESTROY_EVENT      = _IOW(K, 0x09, 8)    # 2×u32 = 8 but padded
IOC_SET_SCRATCH_VA     = _IOWR(K, 0x11, 16)  # u64 + u32 + pad = 16
IOC_SET_TRAP_HANDLER   = _IOW(K, 0x13, 24)   # 2×u64 + u32 + pad = 24
IOC_APER_NEW           = _IOWR(K, 0x14, 16)  # actually variable, but header struct is 16 fixed
IOC_ACQ_VM             = _IOW(K, 0x15, 8)
IOC_ALLOC              = _IOWR(K, 0x16, 40)
IOC_FREE               = _IOW(K, 0x17, 8)
IOC_MAP                = _IOWR(K, 0x18, 24)
IOC_UNMAP              = _IOWR(K, 0x19, 24)
IOC_SVM                = _IOWR(K, 0x20, 24)  # base size 24 + variable attrs
IOC_SET_XNACK          = _IOWR(K, 0x21, 4)
IOC_RUNTIME_ENABLE     = _IOWR(K, 0x25, 16)  # u64 + u32 + u32 = 16

# ── Memory flags ─────────────────────────────────────────────────────
GTT = (1 << 1)
WR  = (1 << 31)
EX  = (1 << 30)
UC  = (1 << 25)

def ioctl_call(fd, req, buf, name=""):
    r = libc.ioctl(fd, req, buf)
    if r != 0:
        e = ctypes.get_errno()
        print(f'  {name}: FAIL errno={e} ({os.strerror(e)})')
        return False
    print(f'  {name}: OK')
    return True

# ── Open devices ─────────────────────────────────────────────────────
kfd = os.open('/dev/kfd', os.O_RDWR)
drm = os.open('/dev/dri/renderD128', os.O_RDWR)
print(f'kfd={kfd} drm={drm}')

# ── Step 1: GET_VERSION ──────────────────────────────────────────────
print('\n=== Step 1: GET_VERSION ===')
vb = ctypes.create_string_buffer(8)
ioctl_call(kfd, IOC_VER, vb, 'GET_VERSION')
maj, mi = struct.unpack('<II', vb.raw)
print(f'  KFD version: {maj}.{mi}')

# ── Step 2: GET_PROCESS_APERTURES_NEW ────────────────────────────────
print('\n=== Step 2: GET_PROCESS_APERTURES ===')
# Try new apertures first (variable-length)
# struct: u64 kfd_process_device_apertures *ptr, u32 num_of_nodes, u32 pad
# First call with num=0 to get count
aper_buf = ctypes.create_string_buffer(16)
struct.pack_into('<QII', aper_buf, 0, 0, 0, 0)
r = libc.ioctl(kfd, IOC_APER_NEW, aper_buf)
if r == 0:
    _, num_nodes, _ = struct.unpack_from('<QII', aper_buf.raw)
    print(f'  Apertures NEW: {num_nodes} node(s)')
else:
    num_nodes = 0
    print(f'  Apertures NEW failed, falling back to old')

# Use old apertures for actual data (simpler)
ab = ctypes.create_string_buffer(400)
ioctl_call(kfd, IOC_APER_OLD, ab, 'GET_PROCESS_APERTURES')
vals = struct.unpack_from('<QQQQQQII', ab.raw, 0)
lds_b = vals[0]
lds_l = vals[1]
scratch_b = vals[2]
scratch_l = vals[3]
gpuvm_b = vals[4]
gpuvm_l = vals[5]
gpu_id = vals[6]
print(f'  gpu_id=0x{gpu_id:x}')
print(f'  LDS: 0x{lds_b:x}-0x{lds_l:x}')
print(f'  Scratch: 0x{scratch_b:x}-0x{scratch_l:x}')
print(f'  GPUVM: 0x{gpuvm_b:x}-0x{gpuvm_l:x}')

# ── Step 3: ACQUIRE_VM ──────────────────────────────────────────────
print('\n=== Step 3: ACQUIRE_VM ===')
ab2 = ctypes.create_string_buffer(8)
struct.pack_into('<II', ab2, 0, drm, gpu_id)
ioctl_call(kfd, IOC_ACQ_VM, ab2, 'ACQUIRE_VM')

# ── Step 4: SET_MEMORY_POLICY ────────────────────────────────────────
print('\n=== Step 4: SET_MEMORY_POLICY ===')
# struct: u64 alt_aperture_base, u64 alt_aperture_size, u32 gpu_id, u32 default_policy, u32 alt_policy, u32 pad
mp = ctypes.create_string_buffer(32)
struct.pack_into('<QQIIII', mp, 0,
    gpuvm_b,  # alternate_aperture_base
    0,        # alternate_aperture_size (0 = no alternate)
    gpu_id,   # gpu_id
    0,        # default_policy = COHERENT
    1,        # alternate_policy = NONCOHERENT
    0,        # pad
)
ioctl_call(kfd, IOC_SET_MEM_POLICY, mp, 'SET_MEMORY_POLICY')

# ── Step 5: Initial ALLOC + MAP (for runtime housekeeping) ──────────
print('\n=== Step 5: Initial allocation ===')
va_cur = gpuvm_b + 0x40000000  # +1GB from base

def alloc(sz, flags, nm):
    global va_cur
    va = va_cur
    b = ctypes.create_string_buffer(40)
    struct.pack_into('<QQQQII', b, 0, va, sz, 0, 0, gpu_id, flags)
    r = libc.ioctl(kfd, IOC_ALLOC, b)
    if r != 0:
        e = ctypes.get_errno()
        print(f'  {nm} ALLOC FAIL: {e} ({os.strerror(e)})')
        return None
    h = struct.unpack_from('<QQQQII', b.raw)
    va_cur += (sz + 0xFFF) & ~0xFFF
    print(f'  {nm}: handle=0x{h[2]:x} mmap_off=0x{h[3]:x} va=0x{va:x}')
    return (h[2], va, h[3])

def mapgpu(handle, nm):
    gids = ctypes.create_string_buffer(8)
    struct.pack_into('<I', gids, 0, gpu_id)
    gp = ctypes.addressof(gids)
    b = ctypes.create_string_buffer(24)
    struct.pack_into('<QQIi', b, 0, handle, gp, 1, 0)
    r = libc.ioctl(kfd, IOC_MAP, b)
    ok = r == 0
    if not ok:
        e = ctypes.get_errno()
        print(f'  MAP {nm}: FAIL {e} ({os.strerror(e)})')
    else:
        print(f'  MAP {nm}: OK')
    return ok

# Initial scratch/signal page
init_page = alloc(4096, GTT | WR | UC, 'init_page')
if init_page:
    mapgpu(init_page[0], 'init_page')

# ── Step 6: RUNTIME_ENABLE ──────────────────────────────────────────
print('\n=== Step 6: RUNTIME_ENABLE ===')
# struct: u64 r_debug, u32 mode_mask, u32 capabilities_mask
re = ctypes.create_string_buffer(16)
struct.pack_into('<QII', re, 0,
    0,    # r_debug (NULL = no debug)
    0,    # mode_mask (0 = just enable runtime, no debug modes)
    0,    # capabilities_mask (output)
)
ioctl_call(kfd, IOC_RUNTIME_ENABLE, re, 'RUNTIME_ENABLE')
_, mode_out, cap_out = struct.unpack_from('<QII', re.raw)
print(f'  mode=0x{mode_out:x} capabilities=0x{cap_out:x}')

# ── Step 7: SET_XNACK_MODE ──────────────────────────────────────────
print('\n=== Step 7: SET_XNACK_MODE ===')
xn = ctypes.create_string_buffer(4)
struct.pack_into('<i', xn, 0, 0)  # 0 = disabled
ioctl_call(kfd, IOC_SET_XNACK, xn, 'SET_XNACK_MODE')

# ── Step 8: CREATE_EVENT (signal events for queue completion) ────────
print('\n=== Step 8: CREATE_EVENT ===')
# struct: u64 event_page_offset, u32 trigger_data, u32 event_type, u32 auto_reset, u32 node_id, u32 event_id, u32 event_slot_index
# Create a few signal events like HSA does
event_ids = []
for i in range(3):
    ev = ctypes.create_string_buffer(32)
    struct.pack_into('<QIIIIII', ev, 0,
        0,    # event_page_offset (output)
        0,    # event_trigger_data (output)
        0,    # event_type = SIGNAL
        1,    # auto_reset = true
        0,    # node_id
        0,    # event_id (output)
        0,    # event_slot_index (output)
    )
    if ioctl_call(kfd, IOC_CREATE_EVENT, ev, f'CREATE_EVENT[{i}]'):
        vals = struct.unpack_from('<QIIIIII', ev.raw)
        eid = vals[5]
        eslot = vals[6]
        epoff = vals[0]
        print(f'    event_id={eid} slot={eslot} page_off=0x{epoff:x}')
        event_ids.append(eid)

# ── Step 9: Allocate scratch backing ─────────────────────────────────
print('\n=== Step 9: SET_SCRATCH_BACKING_VA ===')
scratch_alloc = alloc(262144, GTT | WR | UC, 'scratch_backing')  # 256KB
if scratch_alloc:
    mapgpu(scratch_alloc[0], 'scratch_backing')
    sb = ctypes.create_string_buffer(16)
    struct.pack_into('<QII', sb, 0, scratch_alloc[1], gpu_id, 0)
    ioctl_call(kfd, IOC_SET_SCRATCH_VA, sb, 'SET_SCRATCH_BACKING_VA')

# ── Step 10: SET_TRAP_HANDLER ────────────────────────────────────────
print('\n=== Step 10: SET_TRAP_HANDLER ===')
# Allocate trap handler area (TBA + TMA)
trap_alloc = alloc(8192, GTT | WR | EX | UC, 'trap_handler')
if trap_alloc:
    mapgpu(trap_alloc[0], 'trap_handler')
    # CPU mmap to zero it out
    try:
        trap_mm = mmap.mmap(drm, 8192, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=trap_alloc[2])
        trap_mm.write(b'\x00' * 8192)
        trap_mm.close()
    except:
        pass

    th = ctypes.create_string_buffer(24)
    tba_addr = trap_alloc[1]        # trap base address (GPU VA)
    tma_addr = trap_alloc[1] + 4096  # trap memory area (second page)
    struct.pack_into('<QQII', th, 0, tba_addr, tma_addr, gpu_id, 0)
    ioctl_call(kfd, IOC_SET_TRAP_HANDLER, th, 'SET_TRAP_HANDLER')

# ── Step 11: Allocate ring, eop, ctx for queue ──────────────────────
print('\n=== Step 11: Allocate queue buffers ===')
ring = alloc(16384, GTT | WR | EX | UC, 'ring')
eop  = alloc(4096,  GTT | WR | UC,      'eop')
ctx  = alloc(65536, GTT | WR | UC,      'ctx')

if not ring or not eop or not ctx:
    print("Queue buffer ALLOC failed, aborting")
    exit(1)

mapgpu(ring[0], 'ring')
mapgpu(eop[0], 'eop')
mapgpu(ctx[0], 'ctx')

# CPU mmap of ring
ring_mm = mmap.mmap(drm, 16384, mmap.MAP_SHARED,
                     mmap.PROT_READ | mmap.PROT_WRITE, offset=ring[2])
ring_mm.write(b'\x00' * 16384)
ring_mm.seek(0)
print('  Ring CPU mmap: OK')

# wptr/rptr on a shared anon page
wr_page = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE)
wr_ptr = ctypes.addressof(ctypes.c_char.from_buffer(wr_page, 0))
rd_ptr = wr_ptr + 64
struct.pack_into('<Q', wr_page, 0, 0)
struct.pack_into('<Q', wr_page, 64, 0)
print(f'  wptr=0x{wr_ptr:x} rptr=0x{rd_ptr:x}')

# ── Step 12: CREATE_QUEUE ───────────────────────────────────────────
print('\n=== Step 12: CREATE_QUEUE ===')

# CREATE_QUEUE struct (96 bytes, kernel 6.14):
# u64 ring_base_address
# u64 write_pointer_address
# u64 read_pointer_address
# u64 doorbell_offset          (output)
# u32 ring_size
# u32 gpu_id
# u32 queue_type               (0=PM4/COMPUTE, 2=AQL, 3=SDMA)
# u32 queue_percentage
# u32 queue_priority            (from enum: 1=min..15=max, 7=normal)
# i32 queue_id                 (output)
# u64 eop_buffer_address
# u64 eop_buffer_size
# u64 ctx_save_restore_address
# u32 ctx_save_restore_size
# u32 ctl_stack_size
# u32 sdma_engine_id
# u32 pad

# Try PM4 (type=0) first — this is what we need for WRITE_DATA packets
# Then AQL (type=2) as fallback
for qtype, qtname in [(0, 'COMPUTE/PM4'), (2, 'AQL')]:
    for prio in [7, 15, 1]:
        for ctl in [4096, 0, 8192]:
            qb = ctypes.create_string_buffer(96)
            struct.pack_into('<QQQQIIIIIiQQQIIII', qb, 0,
                ring[1],    # ring_base (GPU VA)
                wr_ptr,     # write_pointer_address
                rd_ptr,     # read_pointer_address
                0,          # doorbell (output)
                16384,      # ring_size
                gpu_id,     # gpu_id
                qtype,      # queue_type
                100,        # percentage
                prio,       # priority
                0,          # queue_id (output)
                eop[1],     # eop addr
                4096,       # eop size
                ctx[1],     # ctx addr
                65536,      # ctx size
                ctl,        # ctl_stack_size
                0,          # sdma_engine_id
                0,          # pad
            )
            r = libc.ioctl(kfd, IOC_CREATE_Q, qb)
            if r == 0:
                vals = struct.unpack_from('<QQQQIIIIIiQQQIIII', qb.raw)
                qid = vals[9]
                doorbell = vals[3]
                print(f'*** QUEUE OK! type={qtname} prio={prio} ctl={ctl} ***')
                print(f'    queue_id={qid} doorbell_offset=0x{doorbell:x}')
                print(f'    wptr_addr=0x{vals[1]:x} rptr_addr=0x{vals[2]:x}')
                print(f'    ring_base=0x{vals[0]:x}')

                # Success! Now test PM4 WRITE_DATA
                if qtype == 0:
                    print('\n=== PM4 WRITE_DATA Test ===')
                    # Write a NOP packet first
                    nop = struct.pack('<II', (3 << 30) | (0x10 << 8) | 0, 0)
                    ring_mm.seek(0)
                    ring_mm.write(nop)
                    print('  NOP written to ring')

                    # TODO: doorbell ring to execute

                # Cleanup
                dq = ctypes.create_string_buffer(8)
                struct.pack_into('<Ii', dq, 0, qid, 0)
                # Don't destroy yet, just report success
                exit(0)
            else:
                e = ctypes.get_errno()
                print(f'  type={qtname} prio={prio} ctl={ctl}: errno={e} ({os.strerror(e)})')

print('\nAll queue variants failed')
print('\nChecking dmesg for hints...')
os.system('dmesg | tail -10')
