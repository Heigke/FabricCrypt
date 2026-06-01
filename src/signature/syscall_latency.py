"""Block 4: syscall p99.9 latency tail.

Nanosleep(1us) repeated N times, 10 percentile features extracted.
"""
import ctypes
import time
import numpy as np

from .thermal import thermal_guard


class _Timespec(ctypes.Structure):
    _fields_ = [("s", ctypes.c_long), ("ns", ctypes.c_long)]


def block_nanosleep(n: int = 20_000, sleep_ns: int = 1000) -> np.ndarray:
    thermal_guard()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    ts = _Timespec(0, sleep_ns)
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        t0 = perf()
        libc.nanosleep(ctypes.byref(ts), None)
        out[i] = perf() - t0
    ps = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99, 99.999, 99.9999]
    return np.asarray([np.percentile(out, p) for p in ps], dtype=np.float64)
