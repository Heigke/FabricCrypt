#!/usr/bin/env python3
"""Minimal KFD CREATE_QUEUE test piggybacking on HSA init.

Key fix: wptr/rptr must be CPU-side mmap addresses (not GPU VAs).
The GPU accesses them via IOMMU/GTT, but the kernel registers them
as CPU userspace addresses.
"""
import os, struct, ctypes, ctypes.util, mmap

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
hsa = ctypes.CDLL('/opt/rocm-7.1.1/lib/libhsa-runtime64.so.1', use_errno=True)

class hsa_agent_t(ctypes.Structure):
    _fields_ = [('handle', ctypes.c_uint64)]

AGENT_CB = ctypes.CFUNCTYPE(ctypes.c_int, hsa_agent_t, ctypes.c_void_p)
hsa.hsa_init.restype = ctypes.c_int
hsa.hsa_iterate_agents.argtypes = [AGENT_CB, ctypes.c_void_p]
hsa.hsa_iterate_agents.restype = ctypes.c_int
hsa.hsa_agent_get_info.argtypes = [hsa_agent_t, ctypes.c_uint32, ctypes.c_void_p]
hsa.hsa_agent_get_info.restype = ctypes.c_int

hsa.hsa_init()
gpu = hsa_agent_t(0)

def find_gpu(a, d):
    dt = ctypes.c_uint32(0)
    hsa.hsa_agent_get_info(a, 17, ctypes.byref(dt))
    if dt.value == 1:
        gpu.handle = a.handle
        return 1
    return 0

hsa.hsa_iterate_agents(AGENT_CB(find_gpu), None)

K = ord('K')
def _IOC(d, t, nr, sz): return (d << 30) | (t << 8) | nr | (sz << 16)
IOC_APER = _IOC(2, K, 0x06, 400)
IOC_ALLOC = _IOC(3, K, 0x16, 40)
IOC_MAP = _IOC(3, K, 0x18, 24)
IOC_CQ = _IOC(3, K, 0x02, 96)

kfd = drm = None
for e in os.listdir('/proc/self/fd'):
    try:
        lnk = os.readlink(f'/proc/self/fd/{e}')
        if lnk == '/dev/kfd':
            kfd = int(e)
        if 'renderD128' in lnk:
            drm = int(e)
    except:
        pass

print(f'kfd={kfd} drm={drm}')

ab = ctypes.create_string_buffer(400)
libc.ioctl(kfd, IOC_APER, ab)
vals = struct.unpack_from('<QQQQQQII', ab.raw)
gpu_id = vals[6]
gpuvm_b = vals[4]
print(f'gpu_id=0x{gpu_id:x}')

va = gpuvm_b + 0x80000000
GTT = (1 << 1)
WR = (1 << 31)
EX = (1 << 30)
UC = (1 << 25)

def alloc(sz, flags, nm=""):
    global va
    b = ctypes.create_string_buffer(40)
    struct.pack_into('<QQQQII', b, 0, va, sz, 0, 0, gpu_id, flags)
    r = libc.ioctl(kfd, IOC_ALLOC, b)
    h = struct.unpack_from('<QQQQII', b.raw)
    v = va
    va += (sz + 0xFFF) & ~0xFFF
    if r == 0:
        print(f'  {nm}: handle=0x{h[2]:x} gpu_va=0x{v:x} mmap_off=0x{h[3]:x}')
        return (h[2], v, h[3])
    else:
        print(f'  {nm} ALLOC FAIL: {ctypes.get_errno()}')
        return None

def mapg(handle, nm=""):
    gids = ctypes.create_string_buffer(8)
    struct.pack_into('<I', gids, 0, gpu_id)
    b = ctypes.create_string_buffer(24)
    struct.pack_into('<QQIi', b, 0, handle, ctypes.addressof(gids), 1, 0)
    r = libc.ioctl(kfd, IOC_MAP, b)
    if r == 0:
        return True
    print(f'  MAP {nm} FAIL: {ctypes.get_errno()}')
    return False

# Allocate buffers
ring = alloc(65536, GTT | WR | EX | UC, 'ring')
eop = alloc(4096, GTT | WR | UC, 'eop')
ctx = alloc(13942784, GTT | WR | UC, 'ctx')
wrb = alloc(4096, GTT | WR | UC, 'wrb')

for h, nm in [(ring, 'ring'), (eop, 'eop'), (ctx, 'ctx'), (wrb, 'wrb')]:
    if h:
        mapg(h[0], nm)

# CPU mmap of buffers via drm fd
ring_mm = mmap.mmap(drm, 65536, mmap.MAP_SHARED,
                     mmap.PROT_READ | mmap.PROT_WRITE, offset=ring[2])
ring_mm.write(b'\x00' * 65536)

# CRITICAL: CPU mmap of wptr/rptr buffer — get CPU addresses for wptr/rptr
wrb_mm = mmap.mmap(drm, 4096, mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE, offset=wrb[2])
wrb_mm.write(b'\x00' * 4096)

# Get CPU addresses of the mmap'd buffer (these are what KFD needs)
wrb_cpu_base = ctypes.addressof(ctypes.c_char.from_buffer(wrb_mm, 0))
# Match HSA's offsets: wptr at +0x38, rptr at +0x80
wptr_cpu = wrb_cpu_base + 0x38
rptr_cpu = wrb_cpu_base + 0x80

# Also get CPU address of ring buffer
ring_cpu = ctypes.addressof(ctypes.c_char.from_buffer(ring_mm, 0))

print(f'\nGPU VAs: ring=0x{ring[1]:x} eop=0x{eop[1]:x} ctx=0x{ctx[1]:x}')
print(f'CPU ptrs: ring=0x{ring_cpu:x} wptr=0x{wptr_cpu:x} rptr=0x{rptr_cpu:x}')

# Zero wptr/rptr
struct.pack_into('<Q', wrb_mm, 0x38, 0)
struct.pack_into('<Q', wrb_mm, 0x80, 0)

# Test with GPU VA for ring_base but CPU addresses for wptr/rptr
# (matching what HSA does)
print('\n--- Test A: GPU VA ring, CPU addr wptr/rptr ---')
for qtype, qtname in [(2, 'AQL'), (0, 'PM4')]:
    for ring_sz in [4096, 65536]:
        for ctl in [16384, 4096]:
            qb = ctypes.create_string_buffer(96)
            struct.pack_into('<QQQQIIIIIiQQQIIII', qb, 0,
                ring[1],       # ring_base (GPU VA)
                wptr_cpu,      # write_pointer (CPU address!)
                rptr_cpu,      # read_pointer (CPU address!)
                0,             # doorbell (output)
                ring_sz,       # ring_size
                gpu_id,
                qtype,         # queue_type
                100,           # percentage
                7,             # priority
                0,             # queue_id (output)
                eop[1],        # eop addr (GPU VA)
                4096,          # eop size
                ctx[1],        # ctx addr (GPU VA)
                13942784,      # ctx_save_restore_size
                ctl,           # ctl_stack_size
                0, 0           # sdma, pad
            )
            r = libc.ioctl(kfd, IOC_CQ, qb)
            if r == 0:
                v = struct.unpack_from('<QQQQIIIIIiQQQIIII', qb.raw)
                print(f'*** {qtname} ring={ring_sz} ctl={ctl}: OK! qid={v[9]} doorbell=0x{v[3]:x} ***')
            else:
                e = ctypes.get_errno()
                print(f'  {qtname} ring={ring_sz} ctl={ctl}: errno={e} ({os.strerror(e)})')

# Test B: CPU address for ALL pointers (ring, wptr, rptr)
print('\n--- Test B: CPU addr for ring AND wptr/rptr ---')
for qtype, qtname in [(2, 'AQL'), (0, 'PM4')]:
    qb = ctypes.create_string_buffer(96)
    struct.pack_into('<QQQQIIIIIiQQQIIII', qb, 0,
        ring_cpu,      # ring_base (CPU address!)
        wptr_cpu,      # write_pointer (CPU address!)
        rptr_cpu,      # read_pointer (CPU address!)
        0,
        4096,          # ring_size
        gpu_id,
        qtype,
        100, 7, 0,
        eop[1],        # eop (GPU VA)
        4096,
        ctx[1],        # ctx (GPU VA)
        13942784,
        16384,
        0, 0
    )
    r = libc.ioctl(kfd, IOC_CQ, qb)
    if r == 0:
        v = struct.unpack_from('<QQQQIIIIIiQQQIIII', qb.raw)
        print(f'*** {qtname}: OK! qid={v[9]} doorbell=0x{v[3]:x} ***')
    else:
        e = ctypes.get_errno()
        print(f'  {qtname}: errno={e} ({os.strerror(e)})')

hsa.hsa_shut_down()
