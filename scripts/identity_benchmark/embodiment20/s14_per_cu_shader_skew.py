#!/usr/bin/env python3
"""S14: per-CU shader instruction-latency skew on gfx1151 / gfx1100.

Compiles `s14_per_cu_shader_skew.hip` and runs it under
HSA_OVERRIDE_GFX_VERSION=11.0.0. Parses (hw_id, cycles) pairs, groups by CU,
computes per-CU mean cycles, and emits a stable per-die vector.
"""
import os, sys, time, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common20 import (thermal_guard, hostname, save_json, wait_cool,
                      get_apu_temp_c)

DIM = 24
SRC = os.path.join(HERE, 's14_per_cu_shader_skew.hip')
BIN = os.path.join(HERE, 's14_per_cu_shader_skew')


def ensure_built():
    if (os.path.exists(BIN) and os.path.getmtime(BIN) >
        os.path.getmtime(SRC)):
        return
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    # gfx1100 binary is forward-compat with gfx1151 under override
    subprocess.check_call([
        '/usr/bin/hipcc', '-O2', '--offload-arch=gfx1100',
        SRC, '-o', BIN], env=env)


def measure(blocks=2048):
    """Returns array of (cu_id, cycles)."""
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    p = subprocess.run([BIN, str(blocks)], capture_output=True, text=True,
                       env=env, timeout=30)
    rows = []
    for ln in p.stdout.splitlines():
        parts = ln.split()
        if len(parts) == 2:
            try:
                rows.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
    return np.asarray(rows, dtype=np.int64) if rows else np.zeros((0,2), dtype=np.int64)


def featurize(arr):
    if arr.size == 0:
        return np.zeros(DIM, dtype=np.float64)
    # RDNA3 HW_ID1: [9:6] simd, [13:10] wgp, [17:14] sa, [19:18] se
    hw = arr[:, 0]
    cyc = arr[:, 1].astype(np.float64)
    wgp_id = (hw >> 10) & 0xF
    sa_id  = (hw >> 14) & 0xF
    se_id  = (hw >> 18) & 0x3
    # combine SE/SA/WGP for unique location (WGP = workgroup processor; each
    # contains 2 CUs on RDNA3)
    loc = (se_id.astype(np.int64) << 8) | (sa_id.astype(np.int64) << 4) | wgp_id
    uniq = np.unique(loc)
    per_loc_mean = np.array([cyc[loc == u].mean() for u in uniq])
    per_loc_std = np.array([cyc[loc == u].std() if (loc == u).sum() > 1 else 0.0
                            for u in uniq])
    feats = [
        float(cyc.mean()), float(cyc.std()),
        float(np.percentile(cyc, 10)), float(np.percentile(cyc, 50)),
        float(np.percentile(cyc, 90)),
        float(uniq.size),                    # # distinct CU locations seen
        float(per_loc_mean.std()),           # cross-CU latency dispersion
        float(per_loc_mean.max() - per_loc_mean.min()),
        float(per_loc_std.mean()),
    ]
    # top-8 ranked CU means + bottom-7 — stable per-die "shape"
    sorted_means = np.sort(per_loc_mean)
    n = sorted_means.size
    pick = list(range(min(8, n))) + list(range(max(0, n-7), n))
    pad = []
    for i in pick:
        pad.append(float(sorted_means[i]))
    feats += pad
    feats = feats[:DIM]
    while len(feats) < DIM:
        feats.append(0.0)
    return np.asarray(feats, dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment20'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's14_per_cu_skew',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    try:
        ensure_built()
    except Exception as e:
        meta['build_error'] = str(e)
        save_json(os.path.join(out_dir, f'{host}_s14_meta.json'), meta)
        print(f"[s14] BUILD FAIL: {e}", flush=True)
        return None
    print(f"[s14] host={host} reps={reps}", flush=True)
    for r in range(reps):
        wait_cool(target_c=58, timeout_s=60)
        thermal_guard()
        t0 = time.time()
        try:
            arr = measure(blocks=2048)
            if r == 0: meta['n_rows_rep0'] = int(arr.shape[0])
            vecs[r] = featurize(arr)
        except SystemExit as e:
            print(f"[s14] abort rep {r}: {e}"); vecs = vecs[:r]; break
        except Exception as e:
            print(f"[s14] err rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s14] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s "
              f"mean_cyc={vecs[r,0]:.0f} cross_cu_std={vecs[r,6]:.1f}", flush=True)
    out = os.path.join(out_dir, f'{host}_s14.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s14_meta.json'), meta)
    print(f"[s14] saved {out}"); return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
