"""Block 5: NVMe (or filesystem) read-latency p99.9 tail.

Defaults to reading /usr/bin/python3 (always present, cached after warmup,
so what we measure is the queue/scheduler/syscall tail rather than the
raw NVMe medium). For true block-device behaviour use /dev/nvme0n1 with
O_DIRECT (requires root).
"""
import os
import time
import numpy as np

from .thermal import thermal_guard


def block_nvme(n: int = 10_000, path: str = "/usr/bin/python3") -> np.ndarray:
    thermal_guard()
    fd = os.open(path, os.O_RDONLY)
    try:
        out = np.empty(n, dtype=np.int64)
        perf = time.perf_counter_ns
        for i in range(n):
            os.lseek(fd, 0, 0)
            t0 = perf()
            os.read(fd, 4096)
            out[i] = perf() - t0
    finally:
        os.close(fd)
    ps = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99, 99.999, 99.9999]
    return np.asarray([np.percentile(out, p) for p in ps], dtype=np.float64)
