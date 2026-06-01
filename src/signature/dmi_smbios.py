#!/usr/bin/env python3
"""S23: DMI / SMBIOS deep fingerprint.

Per-board manufacturing variance is encoded in SMBIOS Type 0 (BIOS),
Type 1 (System), Type 2 (Baseboard), Type 3 (Chassis), Type 4 (CPU),
Type 16/17 (memory device). Serials, UUIDs, asset tags, mfg dates ->
extremely high-entropy per-host fingerprint.

Uses `sudo -n dmidecode -t N` per type. If sudo fails, falls back to
/sys/class/dmi/id/* which is partial but non-privileged.
"""
import os, sys, time, glob
import numpy as np

try:
    from ._common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      read_str, sudo_cmd, hash_bytes, hash_to_floats)
except ImportError:
    from _common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      read_str, sudo_cmd, hash_bytes, hash_to_floats)

DIM = 18
TYPES = [0, 1, 2, 3, 4, 16, 17, 19]  # add type 19 (memory array mapped address)


def collect():
    per_type = {}
    total_len = 0
    sudo_ok = False
    for t in TYPES:
        out = sudo_cmd(['dmidecode', '-t', str(t)], timeout=8)
        if out and 'SMBIOS' in out:
            sudo_ok = True
        per_type[t] = out
        total_len += len(out)
    # Fallback: /sys/class/dmi/id
    sysfs_blob = ''
    if not sudo_ok:
        for f in sorted(glob.glob('/sys/class/dmi/id/*')):
            v = read_str(f)
            if v: sysfs_blob += os.path.basename(f) + '=' + v + '\n'
        total_len = len(sysfs_blob)
    per_type_hashes = {t: hash_bytes(per_type[t]) for t in TYPES}
    full = '\n'.join(per_type[t] for t in TYPES) if sudo_ok else sysfs_blob
    h_full = hash_bytes(full)
    return {'per_type_hashes': per_type_hashes,
            'h_full': h_full, 'total_len': total_len,
            'sudo_ok': sudo_ok, 'sysfs_blob': sysfs_blob}


def featurize(c):
    feats = [float(c['total_len']), 1.0 if c['sudo_ok'] else 0.0]
    feats += hash_to_floats(c['h_full'], 8)
    # 1 byte from each type's hash
    for t in TYPES:
        h = c['per_type_hashes'][t]
        feats.append(float(h[0]) if h else 0.0)
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), 'results', 'signature_phase22')
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's23_dmi',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s23] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['sudo_ok'] = c['sudo_ok']
            meta['total_len'] = c['total_len']
            meta['sha'] = c['h_full'].hex()[:16]
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s23] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"sudo={c['sudo_ok']} len={c['total_len']}", flush=True)
        time.sleep(0.05)
    out = os.path.join(out_dir, f'{host}_s23.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s23_meta.json'), meta)
    print(f"[s23] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
