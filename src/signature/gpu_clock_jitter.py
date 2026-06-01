#!/usr/bin/env python3
"""S4 (light): GPU clock-edge idle jitter — NO stress, just passive sampling.
Replaces full per-CU shader timing (which would conflict w/ Phase 18B).
Reads hwmon GPU freq + voltage at high rate, extracts per-die DPM transition stats.
"""
import os, sys, time, glob
import numpy as np

try:
    from ._common19 import thermal_guard, hostname, save_json, wait_cool
except ImportError:
    from _common19 import thermal_guard, hostname, save_json, wait_cool

GPU_FREQ = sorted(glob.glob('/sys/class/hwmon/hwmon*/freq*_input'))
GPU_TEMP = sorted(glob.glob('/sys/class/hwmon/hwmon*/temp1_input'))
GPU_VOLT = sorted(glob.glob('/sys/class/hwmon/hwmon*/in0_input'))
DIM = 20

def _ri(p):
    try: return int(open(p).read().strip())
    except Exception: return 0

def measure(n=8000, dt=0.001):
    thermal_guard()
    f = GPU_FREQ[0] if GPU_FREQ else None
    t = GPU_TEMP[0] if GPU_TEMP else None
    v = GPU_VOLT[0] if GPU_VOLT else None
    freq = np.empty(n, dtype=np.int64)
    temp = np.empty(n, dtype=np.int64)
    volt = np.empty(n, dtype=np.int64)
    for i in range(n):
        freq[i] = _ri(f) if f else 0
        temp[i] = _ri(t) if t else 0
        volt[i] = _ri(v) if v else 0
        time.sleep(dt)
    return freq, temp, volt

def featurize(freq, temp, volt):
    df = np.diff(freq.astype(np.float64))
    nz = df[df != 0]
    n_trans = (df != 0).sum()
    feats = np.array([
        float(np.mean(freq)), float(np.std(freq)),
        float(np.median(freq)), float(np.ptp(freq)),
        float(n_trans), float(np.abs(df).mean()),
        float(nz.std()) if nz.size else 0.0,
        float(np.percentile(np.abs(df), 99)),
        float(np.mean(temp)), float(np.std(temp)),
        float(np.mean(volt)), float(np.std(volt)),
        float(np.corrcoef(freq, temp)[0,1]) if temp.std()>0 and freq.std()>0 else 0.0,
        float(np.corrcoef(freq, volt)[0,1]) if volt.std()>0 and freq.std()>0 else 0.0,
        float(len(set(freq.tolist()))),  # # unique DPM states observed
        float(np.percentile(freq, 25)),
        float(np.percentile(freq, 75)),
        float((freq == np.bincount(freq).argmax()).mean()) if freq.size and freq.max()<1e9 else 0.0,
        float(np.diff(freq).max() if freq.size>1 else 0),
        float(np.diff(freq).min() if freq.size>1 else 0),
    ], dtype=np.float64)
    return feats

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'results', 'signature_phase19')
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's4_gpu_clock_jitter',
            't_start': time.time(), 'rep_seconds': []}
    print(f"[s4] host={host} reps={reps} freq_paths={GPU_FREQ}", flush=True)
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            f, t, v = measure()
            vecs[r] = featurize(f, t, v)
        except SystemExit as e:
            print(f"[s4] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s4] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s n_trans={vecs[r,4]:.0f}", flush=True)
    out = os.path.join(out_dir, f'{host}_s4.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s4_meta.json'), meta)
    print(f"[s4] saved {out}"); return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
