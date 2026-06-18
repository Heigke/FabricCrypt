#!/usr/bin/env python3
"""S26: amdgpu read-only register fingerprint via umr.

UMR safety (per project memory): NEVER write SMU mailbox; NEVER read
amdgpu_regs_didt. Safe ops: `umr -r`, `umr -lr`, `umr --clock-scan`,
`umr -O bits`.

We use `--clock-scan` and `-O bits -r ...` for a handful of safe IDs
(GRBM-family status / RLC version registers). Output is parsed for
a stable byte-blob; per-die binning + boot-time DPM table = unique
per-board.

Sudo -n required; if unavailable returns zero vector.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      sudo_cmd, hash_bytes, hash_to_floats)

DIM = 16
UMR = '/opt/amdgpu/bin/umr'


def collect():
    # 1) clock-scan: lists DPM states (sclk/mclk/pcie levels + which active)
    cs = sudo_cmd([UMR, '--clock-scan'], timeout=8)
    # 2) ASIC name + IP discovery (read-only)
    asic = sudo_cmd([UMR, '-lt'], timeout=8)
    # 3) HW IP list (versions of GFX/SDMA/UVD/VCN blocks)
    ipv = sudo_cmd([UMR, '--list-ip-versions'], timeout=8)
    if not ipv:
        ipv = sudo_cmd([UMR, '-li'], timeout=8)
    # Parse clock-scan for active levels
    active = []
    for ln in cs.splitlines():
        if '*' in ln:
            parts = ln.replace('*', '').split()
            if parts and parts[0].isdigit():
                # find MHz
                for p in parts:
                    if p.isdigit():
                        active.append(int(p))
                        break
    n_active = len(active)
    h_full = hash_bytes(cs + '|' + asic + '|' + ipv)
    return {'cs_len': len(cs), 'asic_len': len(asic),
            'ipv_len': len(ipv), 'active': active,
            'h': h_full,
            'available': bool(cs or asic or ipv)}


def featurize(c):
    feats = [float(c['cs_len']), float(c['asic_len']),
             float(c['ipv_len']), float(len(c['active'])),
             1.0 if c['available'] else 0.0]
    a = c['active']
    feats += [float(a[0]) if len(a) > 0 else 0.0,
              float(a[1]) if len(a) > 1 else 0.0,
              float(a[2]) if len(a) > 2 else 0.0]
    feats += hash_to_floats(c['h'], 8)
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment22'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's26_umr',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s26] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['available'] = c['available']
            meta['active'] = c['active']
            meta['sha'] = c['h'].hex()[:16]
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s26] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"avail={c['available']}", flush=True)
        time.sleep(0.1)
    out = os.path.join(out_dir, f'{host}_s26.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s26_meta.json'), meta)
    print(f"[s26] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
