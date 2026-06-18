#!/usr/bin/env python3
"""S5: PCIe AER counters + link state observation.
Ikaros has only one thermal zone, but many PCIe devices. We sample:
 - per-device aer_dev_correctable / aer_dev_fatal counters (sysfs)
 - link speed/width (current vs max)
 - completion timeout settings
Background drift over 60s gives per-board signature (AER background rate is
sensitive to SI margin which differs per-die VRM tolerance).
"""
import os, sys, time, glob, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common19 import thermal_guard, hostname, save_json, wait_cool

AER_FIELDS = ['RxErr', 'BadTLP', 'BadDLLP', 'Rollover', 'Timeout', 'NonFatalErr', 'CorrIntErr', 'HeaderOF']
PCI_BASE = '/sys/bus/pci/devices'

def _list_aer_devs():
    devs = []
    for d in sorted(glob.glob(os.path.join(PCI_BASE, '*'))):
        if os.path.exists(os.path.join(d, 'aer_dev_correctable')):
            devs.append(d)
    return devs[:8]  # cap

def _read_aer(path):
    out = {}
    try:
        for line in open(path):
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                try: out[parts[0]] = int(parts[1])
                except ValueError: pass
    except Exception:
        pass
    return out

def _read_int(p):
    try: return int(open(p).read().strip())
    except Exception: return 0

def _link_info(dev):
    """Return (cur_speed_GTs, cur_width, max_speed_GTs, max_width)."""
    def _s(name):
        try: return float(open(os.path.join(dev, name)).read().split()[0].rstrip('GT/s'))
        except Exception: return 0.0
    def _w(name):
        try: return int(open(os.path.join(dev, name)).read().strip())
        except Exception: return 0
    return _s('current_link_speed'), _w('current_link_width'), _s('max_link_speed'), _w('max_link_width')

def measure(window_s=20):
    thermal_guard()
    devs = _list_aer_devs()
    if not devs:
        return np.zeros(8 * (len(AER_FIELDS) + 2)), {'n_devs': 0}
    snap0 = []
    for d in devs:
        snap0.append({
            'corr': _read_aer(os.path.join(d, 'aer_dev_correctable')),
            'fatal': _read_aer(os.path.join(d, 'aer_dev_fatal')),
            'link': _link_info(d),
            't': time.time(),
        })
    time.sleep(window_s)
    snap1 = []
    for d in devs:
        snap1.append({
            'corr': _read_aer(os.path.join(d, 'aer_dev_correctable')),
            'fatal': _read_aer(os.path.join(d, 'aer_dev_fatal')),
            'link': _link_info(d),
            't': time.time(),
        })
    # features per device: 8 AER deltas + link ratio + link width
    feats = []
    for i in range(min(8, len(devs))):
        for f in AER_FIELDS:
            d0 = snap0[i]['corr'].get(f, 0)
            d1 = snap1[i]['corr'].get(f, 0)
            feats.append(d1 - d0)
        cur_s, cur_w, max_s, max_w = snap1[i]['link']
        feats.append(cur_s)
        feats.append(cur_w)
    # pad
    target = 8 * (len(AER_FIELDS) + 2)
    while len(feats) < target: feats.append(0.0)
    return np.asarray(feats[:target], dtype=np.float64), {'n_devs': len(devs), 'devs': [os.path.basename(d) for d in devs]}

DIM = 8 * (len(AER_FIELDS) + 2)

def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's5_pcie_aer',
            't_start': time.time(), 'rep_seconds': []}
    print(f"[s5] host={host} reps={reps} dim={DIM}", flush=True)
    devinfo = None
    for r in range(reps):
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            v, info = measure()
            vecs[r] = v
            if devinfo is None: devinfo = info
        except SystemExit as e:
            print(f"[s5] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s5] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s sum={vecs[r].sum():.1f}", flush=True)
    meta['devinfo'] = devinfo
    out = os.path.join(out_dir, f'{host}_s5.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s5_meta.json'), meta)
    print(f"[s5] saved {out}"); return out

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
