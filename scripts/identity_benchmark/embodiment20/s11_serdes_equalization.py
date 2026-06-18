#!/usr/bin/env python3
"""S11: PCIe SerDes equalization / Link-Control fingerprint.

Per-die PCIe PHY trains equalization taps once at link-up. The trained values
plus the Link Capabilities/Status registers form a stable per-board signature
across boots. We read what's accessible without privileged setpci-extended
calls: lspci -vv text + per-device sysfs files (current_link_speed/width,
max_link_speed/width, aer counters).

Root-free path: walk /sys/bus/pci/devices/* — each device exposes
current_link_speed, max_link_speed, current_link_width, max_link_width.
Sum/concat across all devices = stable per-board topology+training proxy.
"""
import os, sys, time, glob, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common20 import (thermal_guard, hostname, save_json, wait_cool,
                      read_str, get_apu_temp_c)

DIM = 16
PCI_ROOT = '/sys/bus/pci/devices'

SPEED_MAP = {'2.5 GT/s PCIe': 2.5, '5.0 GT/s PCIe': 5.0,
             '8.0 GT/s PCIe': 8.0, '16.0 GT/s PCIe': 16.0,
             '32.0 GT/s PCIe': 32.0, '2.5 GT/s': 2.5, '5.0 GT/s': 5.0,
             '8.0 GT/s': 8.0, '16.0 GT/s': 16.0}


def collect():
    devs = sorted(glob.glob(os.path.join(PCI_ROOT, '*')))
    speeds_cur, speeds_max, widths_cur, widths_max = [], [], [], []
    classes, vendors = [], []
    aer_corr = aer_fatal = aer_nonfatal = 0
    n_devs = 0
    n_downgraded = 0
    for d in devs:
        n_devs += 1
        cs = read_str(os.path.join(d, 'current_link_speed'))
        ms = read_str(os.path.join(d, 'max_link_speed'))
        cw = read_str(os.path.join(d, 'current_link_width'))
        mw = read_str(os.path.join(d, 'max_link_width'))
        if cs:
            sv = SPEED_MAP.get(cs.strip(), 0.0)
            mv = SPEED_MAP.get(ms.strip(), 0.0)
            speeds_cur.append(sv); speeds_max.append(mv)
            try:
                cwv = int(cw); mwv = int(mw)
                widths_cur.append(cwv); widths_max.append(mwv)
                if sv < mv or cwv < mwv: n_downgraded += 1
            except (ValueError, TypeError):
                pass
        cls = read_str(os.path.join(d, 'class'))
        ven = read_str(os.path.join(d, 'vendor'))
        if cls: classes.append(cls)
        if ven: vendors.append(ven)
        # AER counters (root-free for correctable)
        try:
            v = read_str(os.path.join(d, 'aer_dev_correctable'))
            for ln in v.splitlines():
                parts = ln.split()
                if len(parts) == 2:
                    try: aer_corr += int(parts[1])
                    except ValueError: pass
        except Exception:
            pass
    return {
        'n_devs': n_devs,
        'speeds_cur': np.asarray(speeds_cur, dtype=np.float64),
        'speeds_max': np.asarray(speeds_max, dtype=np.float64),
        'widths_cur': np.asarray(widths_cur, dtype=np.float64),
        'widths_max': np.asarray(widths_max, dtype=np.float64),
        'classes': classes, 'vendors': vendors,
        'aer_corr': aer_corr,
        'n_downgraded': n_downgraded,
    }


def featurize(c):
    def stat(a):
        if a.size == 0: return [0.0, 0.0, 0.0, 0.0]
        return [float(a.sum()), float(a.mean()), float(a.std()), float(a.max())]
    feats = []
    feats += stat(c['speeds_cur'])
    feats += stat(c['widths_cur'])
    # ratio + downgrade count
    cw_sum = float(c['widths_cur'].sum())
    mw_sum = float(c['widths_max'].sum()) or 1.0
    feats += [cw_sum / mw_sum, float(c['n_devs']), float(c['n_downgraded']),
              float(c['aer_corr'])]
    # topology hash → int → 4 bytes as 4 floats (stable per-board ordering)
    sig = ','.join(c['classes']) + '|' + ','.join(c['vendors'])
    h = hashlib.md5(sig.encode()).digest()
    for i in range(4):
        feats.append(float(h[i]))
    return np.asarray(feats, dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment20'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's11_serdes_eq',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s11] host={host} reps={reps}", flush=True)
    for r in range(reps):
        thermal_guard()
        t0 = time.time()
        c = collect()
        vecs[r] = featurize(c)
        if r == 0:
            meta['n_devs'] = c['n_devs']
            meta['n_downgraded'] = c['n_downgraded']
            meta['aer_corr'] = c['aer_corr']
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s11] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.2f}s "
              f"n_devs={c['n_devs']} dg={c['n_downgraded']}", flush=True)
        time.sleep(0.05)
    out = os.path.join(out_dir, f'{host}_s11.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s11_meta.json'), meta)
    print(f"[s11] saved {out}"); return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
