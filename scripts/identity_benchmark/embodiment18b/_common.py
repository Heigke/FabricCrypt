"""Phase 18B shared utilities — REAL GPT-2 small chip injection.

THERMAL: EXTRA STRICT (abort=65 pause=60 cool=50). GPT-2 hit 88C in 9s
in Phase 14; we use stricter guard and per-batch checks.
"""
from __future__ import annotations
import os, sys, time, json, hashlib, socket
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
RESULTS = os.path.join(REPO, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment18b')
os.makedirs(RESULTS, exist_ok=True)

# Re-use Phase 14B live signature
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14b'))
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment14'))

THERMAL = '/sys/class/thermal/thermal_zone0/temp'


def temp_c():
    try:
        return int(open(THERMAL).read()) / 1000.0
    except Exception:
        return 0.0


def thermal_guard(abort_c=65, pause_c=60, cool_c=50, wait_max_s=300, verbose=False):
    """Phase 18B STRICTER guard. Returns dict with action taken."""
    t = temp_c()
    rec = {'t_start': t, 'action': 'ok'}
    if t >= abort_c:
        rec['action'] = 'abort_cool'
        if verbose:
            print(f"[THERMAL ABORT-COOL] {t:.1f}C >= {abort_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                rec['t_end'] = temp_c()
                rec['cooled'] = False
                raise SystemExit(f"[THERMAL ABORT] still {temp_c():.1f}C after {wait_max_s}s")
            time.sleep(10)
        rec['t_end'] = temp_c()
        rec['cooled'] = True
        rec['wait_s'] = time.time() - t0
        return rec
    if t >= pause_c:
        rec['action'] = 'pause_cool'
        if verbose:
            print(f"[THERMAL PAUSE] {t:.1f}C >= {pause_c}, cooling to {cool_c}", flush=True)
        t0 = time.time()
        while temp_c() > cool_c:
            if (time.time() - t0) > wait_max_s:
                break
            time.sleep(10)
            if verbose:
                print(f"[cool] {temp_c():.1f}C", flush=True)
        rec['t_end'] = temp_c()
        rec['wait_s'] = time.time() - t0
    return rec


def wait_cool(target_c=50, timeout_s=300, verbose=True):
    t0 = time.time()
    while temp_c() > target_c:
        if (time.time() - t0) > timeout_s:
            if verbose:
                print(f"[wait_cool] timeout {temp_c():.1f}C > {target_c}", flush=True)
            return False
        time.sleep(5)
    if verbose:
        print(f"[wait_cool] OK {temp_c():.1f}C <= {target_c}", flush=True)
    return True


def save_json(name, obj):
    path = os.path.join(RESULTS, name) if not os.path.isabs(name) else name
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[save] {path}", flush=True)
    return path


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        m = float(arr.mean()) if n else 0.0
        return m, m, m
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def sig_to_seed(sig_vec):
    b = np.asarray(sig_vec, dtype=np.float64).tobytes()
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], 'little')


def hostname():
    return socket.gethostname()
