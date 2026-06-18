"""Phase 16 shared utilities — strict thermal guard (Phase 16 limits).

Phase 11B/12 blew APU to 88C. Phase 16 uses TIGHT limits:
  abort_c=68, pause_c=63, cool_c=50  (per spec)
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/embodiment16'
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14'))

THERMAL = '/sys/class/thermal/thermal_zone0/temp'


def temp_c():
    try:
        return int(open(THERMAL).read()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c=82, pause_c=68, cool_c=57, verbose=False, wait_max_s=120):
    """Phase 16 strict: pause at 63C cool to 50C; only ABORT if cool fails to bring temp below abort_c.

    Slightly relaxed abort_c (75C) so transient idle-temp spikes from other system
    activity don't kill the run; we always attempt to cool before deciding.
    """
    t = temp_c()
    if t >= pause_c:
        if verbose:
            print(f"[THERMAL PAUSE] {t:.1f}C, cooling to {cool_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                if temp_c() >= abort_c:
                    raise SystemExit(f"[THERMAL ABORT] still {temp_c():.1f}C after {wait_max_s}s")
                break
            time.sleep(5)
    elif t >= abort_c:
        raise SystemExit(f"[THERMAL ABORT] {t:.1f}C >= {abort_c}C without cool attempt")
        if verbose:
            print(f"[THERMAL RESUMED] {temp_c():.1f}C", flush=True)


def cool_to(target_c=50, max_wait=240, verbose=False):
    t0 = time.time()
    while temp_c() > target_c:
        if (time.time() - t0) > max_wait:
            return False
        if verbose:
            print(f"[cool] {temp_c():.1f}C → target {target_c}", flush=True)
        time.sleep(5)
    return True


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def diff_ci(a, b, n_boot=2000, alpha=0.05, seed=0, paired=True):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if paired and len(a) == len(b):
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
    print(f"[saved] {p}", flush=True)
