#!/usr/bin/env python3
"""S2: TLB miss latency distribution — 20-dim histogram + 4 stats."""
import os, sys, time, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common19 import thermal_guard, compile_c, hostname, save_json, wait_cool

DIM = 24  # 20 hist bins + p50/p90/p99/std
BIN_EDGES = np.linspace(50, 800, 21)  # cycles

def _bin():
    src = os.path.join(HERE, 's2_tlb_miss.c')
    out = os.path.join(HERE, 's2_tlb_miss')
    if not os.path.exists(out) or os.path.getmtime(src) > os.path.getmtime(out):
        compile_c(src, out)
    return out

def measure(n_pages=16384, n_samples=4000):
    thermal_guard()
    p = subprocess.run([_bin(), str(n_pages), str(n_samples)],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.uint64).astype(np.float64)

def featurize(arr):
    h, _ = np.histogram(arr, bins=BIN_EDGES)
    h = h.astype(np.float64) / max(h.sum(), 1.0)
    stats = np.array([np.percentile(arr, 50), np.percentile(arr, 90),
                      np.percentile(arr, 99), np.std(arr)])
    return np.concatenate([h, stats])

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's2_tlb_miss',
            't_start': time.time(), 'rep_seconds': []}
    print(f"[s2] host={host} reps={reps}", flush=True)
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            arr = measure()
            vecs[r] = featurize(arr)
        except SystemExit as e:
            print(f"[s2] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s2] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s p50={vecs[r,20]:.0f}", flush=True)
    out = os.path.join(out_dir, f'{host}_s2.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s2_meta.json'), meta)
    print(f"[s2] saved {out}", flush=True); return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
