#!/usr/bin/env python3
"""S20: ACPI / PCI device-enumeration fingerprint.

`lspci -mn` gives one line per device (machine-readable). Concatenated &
hashed = stable per-board signature reflecting BIOS quirks + slot
population. Variance across reps should be zero on a single host.
"""
import os, sys, time, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      run_cmd, hash_bytes, hash_to_floats)

DIM = 16


def collect():
    out = run_cmd(['lspci', '-mn'], timeout=8)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    # also -t for tree
    tree = run_cmd(['lspci', '-t'], timeout=8)
    n_devs = len(lines)
    # vendor counts
    vendors = {}
    for ln in lines:
        parts = ln.split()
        # parts[1] = "class"  parts[2] = "vendor:device"
        if len(parts) >= 3:
            tok = parts[2].strip('"')
            v = tok.split(':')[0] if ':' in tok else tok
            vendors[v] = vendors.get(v, 0) + 1
    n_vendors = len(vendors)
    canonical = '\n'.join(sorted(lines))
    h_full = hash_bytes(canonical)
    h_tree = hash_bytes(tree)
    return {'n_devs': n_devs, 'n_vendors': n_vendors,
            'h_full': h_full, 'h_tree': h_tree,
            'len_full': len(canonical), 'len_tree': len(tree)}


def featurize(c):
    feats = [float(c['n_devs']), float(c['n_vendors']),
             float(c['len_full']), float(c['len_tree'])]
    feats += hash_to_floats(c['h_full'], 8)
    feats += hash_to_floats(c['h_tree'], 4)
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment22'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's20_acpi_pci',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s20] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['n_devs'] = c['n_devs']
            meta['n_vendors'] = c['n_vendors']
            meta['sha_full'] = c['h_full'].hex()[:16]
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s20] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"devs={c['n_devs']}", flush=True)
        time.sleep(0.05)
    out = os.path.join(out_dir, f'{host}_s20.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s20_meta.json'), meta)
    print(f"[s20] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
