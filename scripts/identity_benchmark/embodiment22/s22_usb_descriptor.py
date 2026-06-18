#!/usr/bin/env python3
"""S22: USB device descriptor + tree fingerprint.

`lsusb -t` gives the bus tree (controllers, devices, speeds). Combined with
`lsusb` plain text (Vendor:Product per device) and per-device sysfs
descriptors, we get a stable per-board fingerprint. USB controller IPs &
embedded-device VID/PIDs vary between OEM SKUs and revisions.
"""
import os, sys, time, glob, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      read_str, run_cmd, hash_bytes, hash_to_floats)

DIM = 16


def collect():
    tree = run_cmd(['lsusb', '-t'], timeout=8)
    plain = run_cmd(['lsusb'], timeout=8)
    # parse plain for VID:PID count
    vidpids = []
    for ln in plain.splitlines():
        # "Bus 001 Device 002: ID 1d6b:0002 ..."
        parts = ln.split()
        for p in parts:
            if ':' in p and len(p) == 9 and all(c in '0123456789abcdef:'
                                                 for c in p.lower()):
                vidpids.append(p)
                break
    n_devs = len(vidpids)
    # speeds via sysfs
    speeds = []
    for d in glob.glob('/sys/bus/usb/devices/*/speed'):
        v = read_str(d)
        try: speeds.append(float(v))
        except ValueError: pass
    speeds = np.asarray(speeds)
    # count root hubs (entries starting with "/:" or "Bus" in tree)
    n_roots = sum(1 for ln in tree.splitlines() if ln.startswith('/:'))
    canonical = '\n'.join(sorted(vidpids)) + '\n' + tree
    h = hash_bytes(canonical)
    return {'n_devs': n_devs, 'n_roots': n_roots,
            'speeds': speeds, 'h': h, 'len_tree': len(tree),
            'vidpids': vidpids}


def featurize(c):
    feats = [float(c['n_devs']), float(c['n_roots']),
             float(c['len_tree'])]
    s = c['speeds']
    if s.size:
        feats += [float(s.mean()), float(s.std()),
                  float(s.max()), float(s.sum())]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    feats += hash_to_floats(c['h'], 9)
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment22'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's22_usb',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s22] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['n_devs'] = c['n_devs']; meta['n_roots'] = c['n_roots']
            meta['vidpids'] = c['vidpids']; meta['sha'] = c['h'].hex()[:16]
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s22] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"devs={c['n_devs']} roots={c['n_roots']}", flush=True)
        time.sleep(0.05)
    out = os.path.join(out_dir, f'{host}_s22.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s22_meta.json'), meta)
    print(f"[s22] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
