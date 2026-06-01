#!/usr/bin/env python3
"""S6: Thermal-zone offset spread + hwmon temp diversity.
Z2-mini has only 1 thermal_zone in ACPI but many hwmon devices.
We sample all hwmon*/temp*_input simultaneously and treat the *offset
constellation* as the per-board signature (sensor calibration spread).
"""
import os, sys, time, glob
import numpy as np

try:
    from ._common19 import thermal_guard, hostname, save_json, wait_cool
except ImportError:
    from _common19 import thermal_guard, hostname, save_json, wait_cool

def _enumerate_temps():
    paths = []
    for p in sorted(glob.glob('/sys/class/hwmon/hwmon*/temp*_input')):
        paths.append(p)
    for p in sorted(glob.glob('/sys/class/thermal/thermal_zone*/temp')):
        paths.append(p)
    return paths

def _read_t(p):
    try:
        v = int(open(p).read().strip())
        return v / 1000.0
    except Exception:
        return float('nan')

def measure(n=200, dt=0.05):
    thermal_guard()
    paths = _enumerate_temps()
    data = np.zeros((n, len(paths)), dtype=np.float64)
    for i in range(n):
        for j, p in enumerate(paths):
            data[i, j] = _read_t(p)
        time.sleep(dt)
    # Per-sensor stats; replace NaNs with zone-mean
    feats = []
    for j in range(len(paths)):
        col = data[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            feats.extend([0.0, 0.0, 0.0])
            continue
        feats.extend([col.mean(), col.std(), col.max() - col.min()])
    # Cross-sensor: pairwise offsets (signature of sensor calibration spread)
    mu = np.nanmean(data, axis=0)
    valid = mu[np.isfinite(mu)]
    if valid.size >= 2:
        offsets = valid[:, None] - valid[None, :]
        off_flat = offsets[np.triu_indices(valid.size, k=1)]
        feats.append(off_flat.mean())
        feats.append(off_flat.std())
        feats.append(np.percentile(off_flat, 90))
        feats.append(np.percentile(off_flat, 10))
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])
    return np.asarray(feats, dtype=np.float64), {'n_sensors': len(paths), 'paths': paths}

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'results', 'signature_phase19')
    os.makedirs(out_dir, exist_ok=True)
    # Probe dim
    v0, info = measure(n=10, dt=0.01)
    DIM = len(v0)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's6_thermal_spread',
            'sensor_info': info, 't_start': time.time(), 'rep_seconds': []}
    print(f"[s6] host={host} reps={reps} dim={DIM} sensors={info['n_sensors']}", flush=True)
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            v, _ = measure()
            vecs[r] = v
        except SystemExit as e:
            print(f"[s6] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s6] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s", flush=True)
    out = os.path.join(out_dir, f'{host}_s6.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s6_meta.json'), meta)
    print(f"[s6] saved {out}"); return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
