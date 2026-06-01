"""Block 3: DRAM-refresh-aligned latency histogram.

Random-walk reads over a 128MB buffer, latency log-binned to 200 bins
spanning 10ns..10us. Output is a probability mass function (sums to 1).
"""
import os
import random
import time
import numpy as np

from .thermal import thermal_guard

DRAM_NBINS = 200
DRAM_BIN_EDGES = np.logspace(1.0, 4.0, DRAM_NBINS + 1)


def block_dram(n_samples: int = 100_000, size_mb: int = 128) -> np.ndarray:
    thermal_guard()
    SIZE = size_mb * 1024 * 1024
    arr = bytearray(SIZE)
    rng = random.Random(0xBEEF + os.getpid())
    # warm-map pages
    for i in range(0, SIZE, 4096):
        arr[i] = 1
    positions = [rng.randrange(0, SIZE - 64) & ~63 for _ in range(n_samples)]
    samples = np.empty(n_samples, dtype=np.int64)
    perf = time.perf_counter_ns
    for i, p in enumerate(positions):
        t0 = perf()
        _ = arr[p]
        samples[i] = perf() - t0
    hist, _ = np.histogram(samples, bins=DRAM_BIN_EDGES)
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        hist = hist.astype(np.float64)
    return hist
