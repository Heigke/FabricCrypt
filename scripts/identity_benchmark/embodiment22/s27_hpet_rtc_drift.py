#!/usr/bin/env python3
"""S27: HPET/RTC vs MONOTONIC clock drift.

Compare CLOCK_REALTIME vs CLOCK_MONOTONIC over a short window. The
two clocks come from different hardware paths (RTC/HPET vs TSC), and
the ratio drift is dominated by the local crystal PLL trim & temperature.

Per-board crystals have ±20 ppm spec but each board's specific offset
within that envelope is unique and stable on idle.

Light: 60-second sampling at 50 Hz = 0.5% CPU. Single-threaded.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common22 import (thermal_guard, hostname, save_json, get_apu_temp_c)

DIM = 12


def collect(duration_s=30.0, hz=20):
    n = int(duration_s * hz)
    dt = 1.0 / hz
    r0 = time.clock_gettime(time.CLOCK_REALTIME)
    m0 = time.clock_gettime(time.CLOCK_MONOTONIC)
    drifts = np.empty(n)
    for i in range(n):
        target_m = m0 + (i + 1) * dt
        slack = target_m - time.clock_gettime(time.CLOCK_MONOTONIC)
        if slack > 0: time.sleep(slack)
        r = time.clock_gettime(time.CLOCK_REALTIME)
        m = time.clock_gettime(time.CLOCK_MONOTONIC)
        drifts[i] = (r - r0) - (m - m0)  # seconds
    # ppm-equivalent
    ppm = drifts[-1] / max(duration_s, 1e-9) * 1e6
    return {'drifts': drifts, 'final_drift_s': float(drifts[-1]),
            'ppm': float(ppm), 'duration_s': float(duration_s)}


def featurize(c):
    d = c['drifts']
    feats = [float(d.mean()), float(d.std()),
             float(d.max() - d.min()),
             float(c['final_drift_s']),
             float(c['ppm']),
             float(c['duration_s'])]
    # slope (linear fit)
    n = d.size
    if n > 2:
        x = np.arange(n, dtype=np.float64)
        slope, intercept = np.polyfit(x, d, 1)
        feats += [float(slope), float(intercept)]
        resid = d - (slope * x + intercept)
        feats += [float(resid.std()), float(np.abs(resid).max())]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    # quantile features
    feats += [float(np.percentile(d, 25)), float(np.percentile(d, 75))]
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None, duration_s=30.0):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment22'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM,
            'signal': 's27_hpet_rtc_drift', 't_start': time.time(),
            'rep_seconds': [], 'temp_start': get_apu_temp_c(),
            'duration_s': duration_s}
    print(f"[s27] host={host} reps={reps} dur={duration_s}s", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect(duration_s=duration_s)
        vecs[r] = featurize(c)
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s27] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s "
              f"ppm={c['ppm']:+.3f}", flush=True)
    out = os.path.join(out_dir, f'{host}_s27.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s27_meta.json'), meta)
    print(f"[s27] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    # Per script <5min; 10 reps * 30s = 5 min exactly; allow override.
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    run(reps=reps, duration_s=dur)
