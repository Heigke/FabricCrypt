"""Task D: Sub-µs syscall latency tail.

100k nanosleep(0), sched_yield, getpid calls each. Histogram p50/p90/p99/p99.9/p99.99.
Thermal guard every 5s of wall time.
"""
import time
import os
import sys
import ctypes
import ctypes.util
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import thermal_guard, save_json, hostname

# clock_gettime via ctypes for low overhead
CLOCK_MONOTONIC_RAW = 4

class Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
libc.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(Timespec)]
libc.nanosleep.argtypes = [ctypes.POINTER(Timespec), ctypes.POINTER(Timespec)]
libc.sched_yield.argtypes = []
libc.getpid.argtypes = []

def now_ns():
    ts = Timespec()
    libc.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def bench_nanosleep0(n):
    zero = Timespec(0, 0)
    rem = Timespec(0, 0)
    lat = np.empty(n, dtype=np.int64)
    last_guard = time.time()
    for i in range(n):
        t0 = now_ns()
        libc.nanosleep(ctypes.byref(zero), ctypes.byref(rem))
        t1 = now_ns()
        lat[i] = t1 - t0
        if (i & 0x3FFF) == 0 and time.time() - last_guard > 5:
            thermal_guard()
            last_guard = time.time()
    return lat


def bench_sched_yield(n):
    lat = np.empty(n, dtype=np.int64)
    last_guard = time.time()
    for i in range(n):
        t0 = now_ns()
        libc.sched_yield()
        t1 = now_ns()
        lat[i] = t1 - t0
        if (i & 0x3FFF) == 0 and time.time() - last_guard > 5:
            thermal_guard()
            last_guard = time.time()
    return lat


def bench_getpid(n):
    lat = np.empty(n, dtype=np.int64)
    last_guard = time.time()
    for i in range(n):
        t0 = now_ns()
        libc.getpid()
        t1 = now_ns()
        lat[i] = t1 - t0
        if (i & 0x3FFF) == 0 and time.time() - last_guard > 5:
            thermal_guard()
            last_guard = time.time()
    return lat


def summarize(lat):
    return {
        'n': int(lat.size),
        'mean_ns': float(lat.mean()),
        'std_ns': float(lat.std()),
        'p50': float(np.percentile(lat, 50)),
        'p90': float(np.percentile(lat, 90)),
        'p99': float(np.percentile(lat, 99)),
        'p99_9': float(np.percentile(lat, 99.9)),
        'p99_99': float(np.percentile(lat, 99.99)),
        'min': int(lat.min()),
        'max': int(lat.max()),
    }


def main():
    N = 100_000
    host = hostname()
    print(f"[D] host={host} starting syscall latency benchmark N={N}")

    thermal_guard()
    # warmup
    for _ in range(1000):
        libc.getpid()

    t0 = time.time()
    lat_ns = bench_nanosleep0(N)
    print(f"[D] nanosleep done in {time.time()-t0:.1f}s")

    thermal_guard()
    t0 = time.time()
    lat_yi = bench_sched_yield(N)
    print(f"[D] sched_yield done in {time.time()-t0:.1f}s")

    thermal_guard()
    t0 = time.time()
    lat_pi = bench_getpid(N)
    print(f"[D] getpid done in {time.time()-t0:.1f}s")

    # store raw for KS test (downsampled if huge)
    out = {
        'host': host,
        'N': N,
        'nanosleep0': summarize(lat_ns),
        'sched_yield': summarize(lat_yi),
        'getpid': summarize(lat_pi),
        # raw histograms for KS test (binned to 1ns up to 50µs)
        'raw_samples_ns_nanosleep0': lat_ns[::10].tolist(),  # 10k samples
        'raw_samples_ns_sched_yield': lat_yi[::10].tolist(),
        'raw_samples_ns_getpid': lat_pi[::10].tolist(),
    }
    out_path = f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment12/task_D_syscall_{host}.json"
    save_json(out_path, out)
    print(f"[D] done")


if __name__ == '__main__':
    main()
