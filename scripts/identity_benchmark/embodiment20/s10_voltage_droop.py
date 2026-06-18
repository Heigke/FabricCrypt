#!/usr/bin/env python3
"""S10: Voltage droop / load-step response.

Probe RAPL energy_uj and (if present) USB-C VBUS voltage during a controlled
on/off load step. Per-die VRM compensation + cap ESR yields a unique transient.
"""
import os, sys, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common20 import (thermal_guard, hostname, save_json, wait_cool,
                      read_int, find_hwmon, get_apu_temp_c)

DIM = 18
RAPL = sorted(glob.glob('/sys/class/powercap/intel-rapl*/energy_uj'))[:1]
VBUS_HWMON = find_hwmon('hp')  # ucsi voltage / curr
if not VBUS_HWMON:
    # fallback: any hwmon with in0_input
    for h in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
        if os.path.exists(os.path.join(h, 'in0_input')):
            VBUS_HWMON = h; break


def _busy(burst_s):
    """CPU burst on a single thread for burst_s seconds."""
    t0 = time.time()
    x = 0
    # tight integer loop
    while time.time() - t0 < burst_s:
        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return x


def sample(duration_s=4.0, hz=200, burst_at=1.0, burst_dur=1.5):
    """Sample energy + vbus during a load step at burst_at .. burst_at+burst_dur."""
    dt = 1.0 / hz
    n = int(duration_s * hz)
    ts = np.empty(n); ev = np.empty(n); vb = np.empty(n); cu = np.empty(n)
    vfile = os.path.join(VBUS_HWMON, 'in0_input') if VBUS_HWMON else None
    cfile = os.path.join(VBUS_HWMON, 'curr1_input') if VBUS_HWMON else None
    efile = RAPL[0] if RAPL else None
    t_start = time.time()
    busy_until = t_start + burst_at + burst_dur
    busy_from = t_start + burst_at
    for i in range(n):
        now = time.time()
        ts[i] = now - t_start
        ev[i] = read_int(efile) if efile else 0
        vb[i] = read_int(vfile) if vfile else 0
        cu[i] = read_int(cfile) if cfile else 0
        # interleave a short integer burst while in burst window
        if busy_from <= now < busy_until:
            _busy(min(0.5 * dt, max(0.0, busy_until - now)))
        # pacing
        target = t_start + (i+1) * dt
        slack = target - time.time()
        if slack > 0: time.sleep(slack)
    return ts, ev, vb, cu


def featurize(ts, ev, vb, cu, burst_at=1.0, burst_dur=1.5):
    de = np.diff(ev.astype(np.float64))
    # mask pre/burst/post windows
    burst = (ts[1:] >= burst_at) & (ts[1:] < burst_at + burst_dur)
    pre = ts[1:] < burst_at
    post = ts[1:] >= burst_at + burst_dur
    def stat(a):
        if a.size == 0: return 0.0, 0.0
        return float(a.mean()), float(a.std())
    e_pre_m, e_pre_s = stat(de[pre])
    e_b_m, e_b_s = stat(de[burst])
    e_post_m, e_post_s = stat(de[post])

    # voltage / current droop stats (if available)
    vb = vb.astype(np.float64); cu = cu.astype(np.float64)
    v_pre = vb[ts < burst_at]
    v_b = vb[(ts >= burst_at) & (ts < burst_at + burst_dur)]
    v_post = vb[ts >= burst_at + burst_dur]
    def med(a): return float(np.median(a)) if a.size else 0.0
    droop = med(v_pre) - med(v_b)
    recover = med(v_post) - med(v_b)

    # Settling time: samples post-burst until v within 5% of pre median
    settled_n = 0
    pre_med = med(v_pre)
    if pre_med > 0:
        tol = 0.05 * pre_med
        for k, t in enumerate(ts[ts >= burst_at + burst_dur]):
            if abs(vb[ts >= burst_at + burst_dur][k] - pre_med) < tol:
                settled_n = k
                break

    feats = np.array([
        e_pre_m, e_pre_s, e_b_m, e_b_s, e_post_m, e_post_s,
        e_b_m / (e_pre_m + 1.0),               # burst / idle ratio (per-die response)
        float(np.percentile(de[burst], 90)) if burst.any() else 0.0,
        float(np.percentile(de[burst], 10)) if burst.any() else 0.0,
        med(v_pre), med(v_b), med(v_post),
        droop, recover, float(settled_n),
        med(cu[ts < burst_at]),                # current pre
        med(cu[(ts >= burst_at) & (ts < burst_at + burst_dur)]),  # current burst
        med(cu[ts >= burst_at + burst_dur]),   # current post
    ], dtype=np.float64)
    return feats


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment20'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's10_voltage_droop',
            'rapl': RAPL, 'vbus_hwmon': VBUS_HWMON, 't_start': time.time(),
            'rep_seconds': [], 'temp_start': get_apu_temp_c()}
    print(f"[s10] host={host} reps={reps} rapl={bool(RAPL)} vbus={bool(VBUS_HWMON)}", flush=True)
    for r in range(reps):
        wait_cool(target_c=58, timeout_s=60)
        thermal_guard()
        t0 = time.time()
        try:
            ts, ev, vb, cu = sample()
            vecs[r] = featurize(ts, ev, vb, cu)
        except SystemExit as e:
            print(f"[s10] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s10] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s", flush=True)
    out = os.path.join(out_dir, f'{host}_s10.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s10_meta.json'), meta)
    print(f"[s10] saved {out}"); return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
