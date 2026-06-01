#!/usr/bin/env python3
"""S9: Identity-Jacobian — dynamics, derivatives, accelerations, and
cross-signal sensitivity matrices.

Hypothesis: a die's first-/second-order *temporal response* to a self-induced
perturbation is more unique than its steady-state values, because aging,
thermal capacitance, and DVFS PID tuning differ across boards while the
mean values converge under regulation.

Method (no GPU stress):
  1. Generate a SOFT load profile (CPU only, sub-1s bursts of arithmetic),
     stepping through 3 amplitudes.
  2. Sample (RAPL energy, hwmon temp, hwmon freq, nanosleep jitter) at 100Hz
     for 6s per amplitude (= 18s total).
  3. Compute per-signal:
        x[t], dx/dt, d2x/dt2, time-to-half-peak, settling tau
  4. Compute cross-signal Jacobian J[i,j] = corr(dx_i/dt, x_j) and J_eigenvals.
  5. Output: per-signal dynamics (5*4=20) + Jacobian eigvals (4) + cross-corr (6).

Burst <= 60s, no GPU, gentle (4 threads, simple arithmetic).
"""
import os, sys, time, glob, multiprocessing as mp, ctypes
import numpy as np

try:
    from ._common19 import thermal_guard, hostname, save_json, wait_cool
except ImportError:
    from _common19 import thermal_guard, hostname, save_json, wait_cool

RAPL = sorted(glob.glob('/sys/class/powercap/intel-rapl*/energy_uj'))
HW_TEMP = sorted(glob.glob('/sys/class/hwmon/hwmon*/temp1_input'))
HW_FREQ = sorted(glob.glob('/sys/class/hwmon/hwmon*/freq1_input'))

DIM = 4 * 5 + 4 + 6  # 30

def _ri(p):
    try: return int(open(p).read().strip())
    except Exception: return 0

def _worker(amp, stop):
    """Soft load: amp in [0..3] => 0/0.25/0.5/0.75 duty over busy/sleep."""
    duty = amp / 3.0
    while not stop.value:
        # 5ms busy + sleep
        end = time.perf_counter() + 0.005 * duty
        x = 0.0
        while time.perf_counter() < end:
            x = x * 1.000001 + 1.0
        time.sleep(0.005 * (1 - duty))

def _nanosleep_burst(n=20, ns=2000):
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    class TS(ctypes.Structure):
        _fields_=[("s",ctypes.c_long),("ns",ctypes.c_long)]
    ts = TS(0, ns); out = np.empty(n, dtype=np.int64)
    pn = time.perf_counter_ns
    for i in range(n):
        t0 = pn(); libc.nanosleep(ctypes.byref(ts), None); out[i] = pn() - t0
    return out

def _sample_phase(amp, duration_s=5.0, dt=0.01, n_workers=4):
    """Spawn n workers at amp, collect telemetry, return time series."""
    stop = mp.Value('b', False)
    procs = []
    if amp > 0:
        for _ in range(n_workers):
            p = mp.Process(target=_worker, args=(amp, stop)); p.start(); procs.append(p)
    n = int(duration_s / dt)
    energy = np.zeros(n); temp = np.zeros(n); freq = np.zeros(n); nsj = np.zeros(n)
    e0 = _ri(RAPL[0]) if RAPL else 0
    t0 = time.perf_counter()
    for i in range(n):
        thermal_guard()
        e = _ri(RAPL[0]) if RAPL else 0
        energy[i] = (e - e0) / 1e6   # joules cumulative
        temp[i]   = _ri(HW_TEMP[0]) / 1000.0 if HW_TEMP else 0.0
        freq[i]   = _ri(HW_FREQ[0]) / 1e6 if HW_FREQ else 0.0
        nsj[i]    = float(np.median(_nanosleep_burst(n=8, ns=2000)))
        # pace
        target = t0 + (i+1) * dt
        while time.perf_counter() < target: pass
    stop.value = True
    for p in procs:
        p.join(timeout=1.0)
        if p.is_alive(): p.terminate()
    return energy, temp, freq, nsj

def featurize_signal(x, dt=0.01):
    """5 features per signal: peak, settling-tau, d/dt-max, d2/dt2-max, half-rise-time."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 5: return np.zeros(5)
    dx = np.gradient(x, dt)
    d2x = np.gradient(dx, dt)
    peak = float(np.percentile(np.abs(x - x[0]), 95))
    # settling tau: time after peak to reach 1/e of peak deviation
    abs_dev = np.abs(x - x.mean())
    peak_idx = int(np.argmax(abs_dev))
    target = abs_dev[peak_idx] / np.e
    tau_idx = peak_idx
    for k in range(peak_idx, len(x)):
        if abs_dev[k] < target:
            tau_idx = k; break
    tau = (tau_idx - peak_idx) * dt
    # half-rise time
    half = abs_dev[peak_idx] * 0.5
    hr = 0
    for k in range(peak_idx):
        if abs_dev[k] >= half:
            hr = k * dt; break
    return np.array([peak, tau, float(np.abs(dx).max()),
                     float(np.abs(d2x).max()), hr], dtype=np.float64)

def measure():
    # Step through amps 0, 1, 2, 3 for ~5s each
    series = []
    for amp in [0, 1, 2, 3]:
        e, t, f, n = _sample_phase(amp, duration_s=5.0)
        series.append((e, t, f, n))
    # Concat and compute features on the FULL transient
    e_all = np.concatenate([s[0] for s in series])
    t_all = np.concatenate([s[1] for s in series])
    f_all = np.concatenate([s[2] for s in series])
    n_all = np.concatenate([s[3] for s in series])
    f1 = featurize_signal(e_all)
    f2 = featurize_signal(t_all)
    f3 = featurize_signal(f_all)
    f4 = featurize_signal(n_all)
    # Jacobian: corr(dx_i/dt, x_j) for i,j in [e,t,f,n]
    dt = 0.01
    sigs = [e_all, t_all, f_all, n_all]
    derivs = [np.gradient(x, dt) for x in sigs]
    J = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            di = derivs[i]; sj = sigs[j]
            if di.std() > 1e-9 and sj.std() > 1e-9:
                J[i, j] = float(np.corrcoef(di, sj)[0, 1])
    # Eigenvalues of (J + J.T)/2 (symmetric part, real)
    eig = np.linalg.eigvalsh((J + J.T) / 2)
    # Cross-correlations between original signals (6 unique pairs)
    crosses = []
    for i in range(4):
        for j in range(i+1, 4):
            if sigs[i].std() > 1e-9 and sigs[j].std() > 1e-9:
                crosses.append(float(np.corrcoef(sigs[i], sigs[j])[0,1]))
            else:
                crosses.append(0.0)
    feats = np.concatenate([f1, f2, f3, f4, eig, np.asarray(crosses)])
    return feats

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'results', 'signature_phase19')
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's9_jacobian_dynamics',
            't_start': time.time(), 'rep_seconds': []}
    print(f"[s9] host={host} reps={reps} dim={DIM}", flush=True)
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            vecs[r] = measure()
        except SystemExit as e:
            print(f"[s9] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s9] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s eigvals={vecs[r,20:24]}", flush=True)
    out = os.path.join(out_dir, f'{host}_s9.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s9_meta.json'), meta)
    print(f"[s9] saved {out}"); return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
