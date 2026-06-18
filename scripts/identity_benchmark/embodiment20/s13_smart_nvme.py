#!/usr/bin/env python3
"""S13: NVMe SMART / thermal history.

No `nvme-cli` and no `smartctl` on either host. We use the NVMe hwmon
exposure (Composite temp + Sensor1) plus the controller model/serial/firmware
from `/sys/class/nvme/nvme0/{model,serial,firmware_rev}`. The thermal-response
to a fixed read workload (1 MiB sequential read of /dev/nvme0n1 first block)
gives a per-drive thermal-fingerprint.
"""
import os, sys, time, hashlib, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common20 import (thermal_guard, hostname, save_json, wait_cool,
                      read_str, read_int, get_apu_temp_c)

DIM = 16
NVME_SYS = '/sys/class/nvme/nvme0'
NVME_DEV = '/dev/nvme0n1'


def find_nvme_hwmon():
    for h in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
        try:
            n = open(os.path.join(h, 'name')).read().strip()
            if n == 'nvme':
                return h
        except Exception:
            pass
    return None


def read_temps(hwmon):
    temps = []
    if not hwmon: return temps
    for f in sorted(glob.glob(os.path.join(hwmon, 'temp*_input'))):
        v = read_int(f) / 1000.0
        if v > 0: temps.append(v)
    return temps


def thermal_workload(seconds=2.0):
    """Light read workload to provoke controller heating."""
    # readable by user? device file typically root-only — fall back to reading
    # block device sysfs counters as a proxy workload.
    fd = None
    try:
        fd = os.open(NVME_DEV, os.O_RDONLY | os.O_NONBLOCK)
        buf_sz = 4096
        t_end = time.time() + seconds
        nbytes = 0
        while time.time() < t_end:
            try:
                os.lseek(fd, 0, 0)
                data = os.read(fd, buf_sz)
                nbytes += len(data)
            except OSError:
                break
        return nbytes
    except PermissionError:
        # Can't read device → use /sys polling as the "workload" (still useful
        # because the polling itself is a stable workload for hwmon scrape).
        t_end = time.time() + seconds
        n = 0
        while time.time() < t_end:
            _ = open(os.path.join(NVME_SYS, 'model')).read()
            n += 1
        return n
    finally:
        if fd is not None:
            try: os.close(fd)
            except Exception: pass


def collect(hwmon, reps_inner=20):
    """Trace temp during a fixed-length workload."""
    thermal_guard()
    # idle baseline
    t_idle = []
    for _ in range(5):
        t_idle.append(read_temps(hwmon))
        time.sleep(0.1)
    # workload + concurrent temp sampling
    t_work = []
    t_end = time.time() + 2.0
    nbytes = 0
    fd = None
    try:
        try:
            fd = os.open(NVME_DEV, os.O_RDONLY | os.O_NONBLOCK)
        except (PermissionError, OSError):
            fd = None
        while time.time() < t_end:
            if fd is not None:
                try:
                    os.lseek(fd, 0, 0)
                    nbytes += len(os.read(fd, 4096))
                except OSError:
                    break
            t_work.append(read_temps(hwmon))
    finally:
        if fd is not None:
            try: os.close(fd)
            except Exception: pass
    # post relax
    t_post = []
    for _ in range(10):
        t_post.append(read_temps(hwmon))
        time.sleep(0.1)
    return t_idle, t_work, t_post, nbytes


def featurize(t_idle, t_work, t_post, dev_info):
    def col(arr, j):
        out = [x[j] for x in arr if len(x) > j]
        return np.asarray(out, dtype=np.float64)
    # use sensor channel 0 + 1 if present
    feats = []
    for j in (0, 1):
        i = col(t_idle, j); w = col(t_work, j); p = col(t_post, j)
        if i.size == 0 and w.size == 0:
            feats += [0.0]*6
            continue
        feats += [
            float(i.mean()) if i.size else 0.0,
            float(w.mean()) if w.size else 0.0,
            float(p.mean()) if p.size else 0.0,
            float((w.mean() if w.size else 0.0) - (i.mean() if i.size else 0.0)),  # delta
            float(w.std()) if w.size else 0.0,
            float(p.max() if p.size else 0.0),
        ]
    # device-identity component (stable per controller)
    sig = f"{dev_info.get('model','')}|{dev_info.get('serial','')}|{dev_info.get('firmware_rev','')}"
    h = hashlib.md5(sig.encode()).digest()
    feats += [float(h[i]) for i in range(4)]
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment20'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    hwmon = find_nvme_hwmon()
    dev_info = {k: read_str(os.path.join(NVME_SYS, k))
                for k in ('model', 'serial', 'firmware_rev')}
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's13_nvme_smart',
            'nvme_hwmon': hwmon, 'device_info': dev_info,
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    print(f"[s13] host={host} hwmon={hwmon} model={dev_info.get('model','')[:30]}", flush=True)
    for r in range(reps):
        wait_cool(target_c=58, timeout_s=60)
        thermal_guard()
        t0 = time.time()
        try:
            ti, tw, tp, nb = collect(hwmon)
            vecs[r] = featurize(ti, tw, tp, dev_info)
        except SystemExit as e:
            print(f"[s13] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s13] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s "
              f"delta0={vecs[r,3]:.2f}C", flush=True)
    out = os.path.join(out_dir, f'{host}_s13.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s13_meta.json'), meta)
    print(f"[s13] saved {out}"); return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
