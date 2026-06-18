#!/usr/bin/env python3
"""S21: PCIe link speed/width degradation per device.

For every /sys/bus/pci/devices/* with link properties, compare current vs
max speed & width. Per-die SerDes margin yields a stable but per-board
degradation pattern (which links and how far).
"""
import os, sys, time, glob, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common22 import (thermal_guard, hostname, save_json, get_apu_temp_c,
                      read_str, hash_bytes, hash_to_floats)

DIM = 16
PCI = '/sys/bus/pci/devices'
SPEED_MAP = {'2.5 GT/s PCIe': 2.5, '5.0 GT/s PCIe': 5.0,
             '8.0 GT/s PCIe': 8.0, '16.0 GT/s PCIe': 16.0,
             '32.0 GT/s PCIe': 32.0, '2.5 GT/s': 2.5, '5.0 GT/s': 5.0,
             '8.0 GT/s': 8.0, '16.0 GT/s': 16.0, '32.0 GT/s': 32.0}


def parse_speed(s):
    return SPEED_MAP.get(s.strip(), 0.0)


def collect():
    devs = sorted(glob.glob(os.path.join(PCI, '*')))
    n = 0; n_dg_speed = 0; n_dg_width = 0
    speed_ratios = []; width_ratios = []
    speed_deltas = []; width_deltas = []
    dg_devices = []
    for d in devs:
        cs = read_str(os.path.join(d, 'current_link_speed'))
        ms = read_str(os.path.join(d, 'max_link_speed'))
        cw = read_str(os.path.join(d, 'current_link_width'))
        mw = read_str(os.path.join(d, 'max_link_width'))
        if not cs or not ms: continue
        n += 1
        sv = parse_speed(cs); mv = parse_speed(ms)
        try: cwv = int(cw); mwv = int(mw)
        except (ValueError, TypeError): cwv = mwv = 0
        if mv > 0:
            speed_ratios.append(sv / mv)
            speed_deltas.append(mv - sv)
            if sv < mv:
                n_dg_speed += 1
                dg_devices.append(os.path.basename(d) + ':speed')
        if mwv > 0:
            width_ratios.append(cwv / mwv)
            width_deltas.append(mwv - cwv)
            if cwv < mwv:
                n_dg_width += 1
                dg_devices.append(os.path.basename(d) + ':width')
    return {'n': n, 'n_dg_speed': n_dg_speed, 'n_dg_width': n_dg_width,
            'speed_ratios': np.asarray(speed_ratios),
            'width_ratios': np.asarray(width_ratios),
            'speed_deltas': np.asarray(speed_deltas),
            'width_deltas': np.asarray(width_deltas),
            'dg_devices': dg_devices}


def featurize(c):
    def stat(a):
        if a.size == 0: return [0.0, 0.0, 0.0]
        return [float(a.mean()), float(a.std()), float(a.min())]
    feats = [float(c['n']), float(c['n_dg_speed']), float(c['n_dg_width'])]
    feats += stat(c['speed_ratios'])
    feats += stat(c['width_ratios'])
    feats += stat(c['speed_deltas'])
    # hash over the dg-device list (stable per-board)
    sig = ','.join(sorted(c['dg_devices']))
    h = hash_bytes(sig)
    feats += hash_to_floats(h, 4)
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment22'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM,
            'signal': 's21_pcie_link', 't_start': time.time(),
            'rep_seconds': [], 'temp_start': get_apu_temp_c()}
    print(f"[s21] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['n'] = c['n']; meta['dg_speed'] = c['n_dg_speed']
            meta['dg_width'] = c['n_dg_width']
            meta['dg_devices'] = c['dg_devices']
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s21] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"n={c['n']} dg_s={c['n_dg_speed']} dg_w={c['n_dg_width']}",
              flush=True)
        time.sleep(0.05)
    out = os.path.join(out_dir, f'{host}_s21.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s21_meta.json'), meta)
    print(f"[s21] saved {out}")
    return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
