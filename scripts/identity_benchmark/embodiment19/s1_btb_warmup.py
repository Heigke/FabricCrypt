#!/usr/bin/env python3
"""S1: Branch predictor warmup latency — features per-rep.

Returns a feature vector (10 dims) capturing the BTB-saturation curve:
  - warmup peak (first 10%), steady-state median, ratio peak/steady,
  - p50/p90/p99 of steady-state, std of warmup, std of steady,
  - cycles to 1.5x steady, residual jitter.
"""
import os, sys, time, struct, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common19 import thermal_guard, compile_c, hostname, save_json, wait_cool

DIM = 10

def _bin():
    src = os.path.join(HERE, 's1_btb_warmup.c')
    out = os.path.join(HERE, 's1_btb_warmup')
    if not os.path.exists(out) or os.path.getmtime(src) > os.path.getmtime(out):
        compile_c(src, out)
    return out

def measure(n_branches=4096, n_iter=4000):
    thermal_guard()
    binp = _bin()
    p = subprocess.run([binp, str(n_branches), str(n_iter)],
                       capture_output=True, check=True)
    arr = np.frombuffer(p.stdout, dtype=np.uint64).astype(np.float64)
    if arr.size < n_iter:
        raise RuntimeError(f"S1 short read {arr.size}/{n_iter}")
    return arr

def featurize(arr):
    n = len(arr)
    warm_n = max(50, n // 10)
    warm = arr[:warm_n]
    steady = arr[n // 2:]
    p50 = np.percentile(steady, 50)
    p90 = np.percentile(steady, 90)
    p99 = np.percentile(steady, 99)
    peak = np.percentile(warm, 95)
    ratio = peak / max(p50, 1.0)
    # cycles-to-1.5x-steady: index where running median drops below 1.5*p50
    win = 32
    rm = np.convolve(arr, np.ones(win)/win, mode='valid')
    thresh = 1.5 * p50
    idx = np.argmax(rm <= thresh) if (rm <= thresh).any() else len(rm)-1
    feats = np.array([
        peak, p50, ratio, p90, p99,
        np.std(warm), np.std(steady), float(idx),
        np.median(np.diff(steady)), np.percentile(steady, 99.9),
    ], dtype=np.float64)
    return feats

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's1_btb_warmup',
            't_start': time.time(), 'rep_seconds': []}
    print(f"[s1] host={host} reps={reps}", flush=True)
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            arr = measure()
            vecs[r] = featurize(arr)
        except SystemExit as e:
            print(f"[s1] abort rep {r}: {e}", flush=True); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s1] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s feats[:3]={vecs[r,:3]}", flush=True)
    out = os.path.join(out_dir, f'{host}_s1.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s1_meta.json'), meta)
    print(f"[s1] saved {out}", flush=True)
    return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
