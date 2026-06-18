"""Shared utilities for Phase 15 embodiment experiments.

Strict thermal guard + live-chip-state readers (cheap path).
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/embodiment15'
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14'))

THERMAL = '/sys/class/thermal/thermal_zone0/temp'
RAPL_PKG = '/sys/class/powercap/intel-rapl:0/energy_uj'


def temp_c():
    try:
        return int(open(THERMAL).read()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c=82, pause_c=70, cool_c=55, verbose=False, wait_max_s=240):
    """Strict but pragmatic: pause/cool above pause_c; only ABORT if still hot after wait."""
    t = temp_c()
    if t >= pause_c:
        if verbose:
            print(f"[THERMAL PAUSE] {t:.1f}C, cooling to {cool_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s and temp_c() >= abort_c:
                raise SystemExit(f"[THERMAL ABORT] {temp_c():.1f}C after {wait_max_s}s cool wait")
            time.sleep(5)
        if verbose:
            print(f"[THERMAL RESUMED] {temp_c():.1f}C", flush=True)


def rapl_uj():
    try:
        return int(open(RAPL_PKG).read())
    except Exception:
        return 0


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Return (mean, lo, hi) for the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def diff_ci(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Mean of (a - b) with bootstrap CI; a,b same length OR independent samples."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == len(b):
        diffs = a - b
        idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
        means = diffs[idx].mean(axis=1)
    else:
        ia = rng.integers(0, len(a), size=(n_boot, len(a)))
        ib = rng.integers(0, len(b), size=(n_boot, len(b)))
        means = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(means.mean()), float(lo), float(hi)


def save_json(name, payload):
    p = os.path.join(RESULTS, name)
    with open(p, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[saved] {p}")


def dram_latency_burst(n=8, stride_kb=4096):
    """Quick cache-thrashing burst — proxy for DRAM access timing.

    Returns n samples of ns/iter timings. Cheap (~50us) so we can call per-token.
    Uses a working set ~stride_kb*256 = 1 MB to ensure misses.
    """
    if not hasattr(dram_latency_burst, '_buf'):
        dram_latency_burst._buf = np.random.default_rng(0).integers(
            0, 1 << 30, size=stride_kb * 256, dtype=np.int64)
    buf = dram_latency_burst._buf
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    L = len(buf)
    stride = max(1, (stride_kb * 1024) // 8)  # cacheline-busting stride
    for i in range(n):
        t0 = perf()
        s = 0
        # 16 random-ish strided accesses
        idx = (i * 7919) % L
        for k in range(16):
            s ^= int(buf[(idx + k * stride) % L])
        out[i] = perf() - t0
    return out
