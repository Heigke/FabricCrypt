#!/usr/bin/env python3
"""S3: Per-CCX cache line wakeup latency. Strix-Halo 16-core has 2 CCDs.
Picks pairs across CCXs (0-7 vs 8-15) and same-CCX pairs as contrast.
Feature: 5 stats x 4 pairs = 20 dims.
"""
import os, sys, time, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common19 import thermal_guard, compile_c, hostname, save_json, wait_cool

PAIRS = [(0, 8), (1, 9), (2, 10), (0, 4)]  # 3 cross-CCD + 1 same-CCD
DIM = len(PAIRS) * 5

def _bin():
    src = os.path.join(HERE, 's3_ccx_wakeup.c')
    out = os.path.join(HERE, 's3_ccx_wakeup')
    if not os.path.exists(out) or os.path.getmtime(src) > os.path.getmtime(out):
        compile_c(src, out)
    return out

def measure_pair(a, b, n=4000):
    thermal_guard()
    p = subprocess.run([_bin(), str(a), str(b), str(n)],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.uint64).astype(np.float64)

def stats5(arr):
    return [np.percentile(arr,50), np.percentile(arr,90), np.percentile(arr,99),
            np.std(arr), np.mean(arr)]

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's3_ccx_wakeup',
            'pairs': PAIRS, 't_start': time.time(), 'rep_seconds': []}
    print(f"[s3] host={host} reps={reps}", flush=True)
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            feats = []
            for (a, b) in PAIRS:
                arr = measure_pair(a, b)
                feats.extend(stats5(arr))
            vecs[r] = feats
        except SystemExit as e:
            print(f"[s3] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s3] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s p50pair0={vecs[r,0]:.0f}", flush=True)
    out = os.path.join(out_dir, f'{host}_s3.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s3_meta.json'), meta)
    print(f"[s3] saved {out}"); return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
