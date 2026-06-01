#!/usr/bin/env python3
"""S24: Kernel boot timing fingerprint.

`journalctl -k --boot=0` (kernel ring buffer of current boot) prints
`[ TIME ]` per message. We extract subsystem ready-times (PCI bus,
SATA/NVMe init, USB enumeration, ALSA, amdgpu) from the first ~1500
lines. Per-board boot-order quirks (slot population, EFI vendor) cause
distinct timing patterns.

For the SAME boot reps are identical (within-host stability ~0).
Cross-host: distinct devices => distinct prints => large KS-D.
"""
import os, sys, time, re
import numpy as np

try:
    from ._common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      run_cmd, hash_bytes, hash_to_floats)
except ImportError:
    from _common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      run_cmd, hash_bytes, hash_to_floats)

DIM = 16
TS_RE = re.compile(r'\[\s*(\d+\.\d+)\]')

# Subsystem keywords; first-occurrence timestamp = ready-time proxy
TAGS = ['PCI', 'usb', 'nvme', 'ahci', 'amdgpu', 'snd',
        'thermal', 'cpufreq', 'systemd', 'EXT4', 'Bluetooth',
        'iwlwifi']


def collect():
    txt = run_cmd(['journalctl', '-k', '--boot=0', '--no-pager',
                   '--output=short-monotonic', '-n', '2000'], timeout=15)
    if not txt:
        # Fallback to dmesg
        txt = run_cmd(['dmesg', '--ctime', '--no-pager'], timeout=8)
    lines = txt.splitlines()[:1500]
    n_lines = len(lines)
    first_ts = {}
    all_ts = []
    for ln in lines:
        m = TS_RE.search(ln)
        if not m: continue
        t = float(m.group(1))
        all_ts.append(t)
        for tag in TAGS:
            if tag in ln and tag not in first_ts:
                first_ts[tag] = t
    h = hash_bytes('\n'.join(lines[:200]))  # hash first 200 lines
    return {'n_lines': n_lines, 'first_ts': first_ts,
            'last_ts': max(all_ts) if all_ts else 0.0,
            'h': h}


def featurize(c):
    feats = [float(c['n_lines']), float(c['last_ts'])]
    for tag in TAGS:
        feats.append(float(c['first_ts'].get(tag, 0.0)))
    feats += hash_to_floats(c['h'], 2)
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'results', 'signature_phase22')
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's24_boot',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s24] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['n_lines'] = c['n_lines']
            meta['last_ts'] = c['last_ts']
            meta['first_ts'] = c['first_ts']
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s24] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"lines={c['n_lines']} last={c['last_ts']:.1f}s",
              flush=True)
        time.sleep(0.1)
    out = os.path.join(out_dir, f'{host}_s24.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s24_meta.json'), meta)
    print(f"[s24] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
