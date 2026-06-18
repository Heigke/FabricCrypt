#!/usr/bin/env python3
"""S7: RAPL energy quantization residual.
Read energy_uj at maximum sustainable rate; compute per-tick increments.
The ADC LSB and integration-window jitter give per-die signature.
"""
import os, sys, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common19 import thermal_guard, hostname, save_json, wait_cool

RAPL_PATHS = sorted(glob.glob('/sys/class/powercap/intel-rapl*/energy_uj'))
DIM = 16  # see featurize

def _read_uj(p):
    try: return int(open(p).read().strip())
    except Exception: return -1

def measure(n=200000):
    """200k reads ~ a few s on modern HW."""
    thermal_guard()
    if not RAPL_PATHS:
        return np.zeros(DIM), {'n_paths': 0}
    p = RAPL_PATHS[0]
    fd = os.open(p, os.O_RDONLY)
    try:
        buf = np.empty(n, dtype=np.int64)
        for i in range(n):
            os.lseek(fd, 0, 0)
            buf[i] = int(os.read(fd, 32).strip() or 0)
    finally:
        os.close(fd)
    return buf

def featurize(buf):
    diffs = np.diff(buf.astype(np.int64))
    diffs = diffs[diffs >= 0]  # drop wraparounds
    nz = diffs[diffs > 0]
    # The minimum non-zero diff is the LSB; zero rate reflects how often
    # consecutive reads fall within the same quantum (per-die ADC precision)
    lsb = int(nz.min()) if nz.size else 0
    zero_rate = float((diffs == 0).mean())
    # Quantum residuals: for each diff, (diff mod lsb) — should be ~0 if ADC is clean
    if lsb > 0:
        residuals = diffs % lsb
        res_mean = float(residuals.mean())
        res_std  = float(residuals.std())
    else:
        res_mean = res_std = 0.0
    # Run-length distribution of zero-diffs (per-die integration window)
    is_zero = (diffs == 0).astype(np.int8)
    runs = []
    cur = 0
    for v in is_zero:
        if v: cur += 1
        else:
            if cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    runs = np.asarray(runs) if runs else np.asarray([0])
    feats = np.array([
        lsb, zero_rate, res_mean, res_std,
        float(nz.mean()) if nz.size else 0.0,
        float(nz.std())  if nz.size else 0.0,
        float(np.percentile(nz, 50)) if nz.size else 0.0,
        float(np.percentile(nz, 99)) if nz.size else 0.0,
        float(runs.mean()), float(runs.std()),
        float(runs.max()), float(np.percentile(runs, 90)),
        float(diffs.size),
        float((diffs > 2*lsb).mean()) if lsb > 0 else 0.0,
        float(nz.size),
        float(np.percentile(diffs, 90)),
    ], dtype=np.float64)
    return feats

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's7_rapl_precision',
            'rapl_paths': RAPL_PATHS, 't_start': time.time(), 'rep_seconds': []}
    print(f"[s7] host={host} reps={reps}", flush=True)
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            buf = measure()
            vecs[r] = featurize(buf)
        except SystemExit as e:
            print(f"[s7] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s7] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s lsb={vecs[r,0]:.0f}", flush=True)
    out = os.path.join(out_dir, f'{host}_s7.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s7_meta.json'), meta)
    print(f"[s7] saved {out}"); return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
