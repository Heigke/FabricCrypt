#!/usr/bin/env python3
"""Minimal HSA queue create — just init + find GPU + create queue."""
import os, ctypes
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
hsa = ctypes.CDLL('/opt/rocm-7.1.1/lib/libhsa-runtime64.so.1')

class A(ctypes.Structure):
    _fields_ = [('h', ctypes.c_uint64)]

class Q(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint32), ('features', ctypes.c_uint32),
        ('base', ctypes.POINTER(ctypes.c_void_p)),
        ('doorbell', ctypes.c_uint64),
        ('size', ctypes.c_uint32), ('r1', ctypes.c_uint32),
        ('id', ctypes.c_uint64),
    ]

CB = ctypes.CFUNCTYPE(ctypes.c_int, A, ctypes.c_void_p)
hsa.hsa_init.restype = ctypes.c_int
hsa.hsa_iterate_agents.argtypes = [CB, ctypes.c_void_p]
hsa.hsa_iterate_agents.restype = ctypes.c_int
hsa.hsa_agent_get_info.argtypes = [A, ctypes.c_uint32, ctypes.c_void_p]
hsa.hsa_agent_get_info.restype = ctypes.c_int
hsa.hsa_queue_create.argtypes = [A, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(ctypes.POINTER(Q))]
hsa.hsa_queue_create.restype = ctypes.c_int
hsa.hsa_queue_destroy.argtypes = [ctypes.POINTER(Q)]
hsa.hsa_queue_destroy.restype = ctypes.c_int

print("init:", hsa.hsa_init())
gpu = A(0)
def find(a, d):
    dt = ctypes.c_uint32(0)
    hsa.hsa_agent_get_info(a, 17, ctypes.byref(dt))
    if dt.value == 1:
        gpu.h = a.h
        return 1
    return 0

hsa.hsa_iterate_agents(CB(find), None)
print(f"gpu: 0x{gpu.h:x}")

qp = ctypes.POINTER(Q)()
s = hsa.hsa_queue_create(gpu, 1024, 1, None, None, 0xFFFFFFFF, 0xFFFFFFFF, ctypes.byref(qp))
print(f"queue_create: {s}")
if s == 0:
    print(f"  id={qp.contents.id} size={qp.contents.size}")
    hsa.hsa_queue_destroy(qp)
hsa.hsa_shut_down()
