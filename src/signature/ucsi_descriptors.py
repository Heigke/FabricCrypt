#!/usr/bin/env python3
"""S25: UCSI power-supply / USB-C PD descriptor fingerprint.

/sys/class/power_supply/ucsi-source-psy-* exposes one entry per Type-C
port. Each PD controller exposes voltage_max, voltage_min, voltage_now,
current_max, current_now, usb_type (PD/Type-C/...). Per-board variation
comes from: number of ports, PD controller model, factory-calibrated
voltage limits, current advertisement.

Read-only sysfs reads only.
"""
import os, sys, time, glob
import numpy as np

try:
    from ._common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      read_int, read_str, hash_bytes, hash_to_floats)
except ImportError:
    from _common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      read_int, read_str, hash_bytes, hash_to_floats)

DIM = 16
BASE = '/sys/class/power_supply'

FIELDS = ['voltage_max', 'voltage_min', 'voltage_now',
          'current_max', 'current_now', 'online']


def collect():
    ports = sorted(glob.glob(os.path.join(BASE, 'ucsi-source-psy-*')))
    n_ports = len(ports)
    feats = {f: [] for f in FIELDS}
    usb_types = []
    for p in ports:
        for f in FIELDS:
            v = read_int(os.path.join(p, f))
            feats[f].append(v)
        usb_types.append(read_str(os.path.join(p, 'usb_type')))
    sig = '|'.join(usb_types) + '|' + str(n_ports)
    h = hash_bytes(sig)
    return {'n_ports': n_ports, 'fields': feats,
            'usb_types': usb_types, 'h': h}


def featurize(c):
    feats = [float(c['n_ports'])]
    for f in FIELDS:
        a = np.asarray(c['fields'][f], dtype=np.float64)
        if a.size:
            # Scale large voltage/current values (uV, uA) down
            feats += [float(a.sum() / 1e6),
                      float(a.max() / 1e6),
                      float(a.std() / 1e6)]
        else:
            feats += [0.0, 0.0, 0.0]
    feats += hash_to_floats(c['h'], 4)
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'results', 'signature_phase22')
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's25_ucsi',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s25] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['n_ports'] = c['n_ports']
            meta['usb_types'] = c['usb_types']
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s25] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"ports={c['n_ports']}", flush=True)
        time.sleep(0.05)
    out = os.path.join(out_dir, f'{host}_s25.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s25_meta.json'), meta)
    print(f"[s25] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
