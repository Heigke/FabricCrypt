#!/usr/bin/env python3
"""S12: DDR PHY training residual + per-channel memory latency.

(a) Read DMI Type-17 (memory devices) - stable per-DIMM serial/manufacturer.
(b) Cross-channel stride latency: walk a buffer > L3 with a channel-spanning
    stride; per-channel UMC training causes µs-scale per-die latency offsets.

Path (a) needs `dmidecode`; on most distros the DMI table can also be read via
sysfs (`/sys/firmware/dmi/tables/DMI`) without sudo. We try sysfs first.
"""
import os, sys, time, struct, hashlib, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common20 import (thermal_guard, hostname, save_json, wait_cool,
                      read_str, get_apu_temp_c)

DIM = 18
DMI_FILE = '/sys/firmware/dmi/tables/DMI'


def _scan_type17(blob):
    """Return list of (handle, size, fields) for SMBIOS Type-17 (memory device).

    Lightweight parser — extracts a stable hash of structure body bytes; we do
    NOT parse fields fully (vendor-specific). The hash is constant per-DIMM
    across boots."""
    out = []
    i = 0
    n = len(blob)
    while i + 4 <= n:
        t = blob[i]
        length = blob[i+1]
        if length < 4 or i + length > n:
            break
        body = blob[i:i+length]
        # skip strings: walk past terminating 0x00 0x00
        j = i + length
        while j + 1 < n and not (blob[j] == 0 and blob[j+1] == 0):
            j += 1
        strings = blob[i+length:j]
        if t == 17:  # memory device
            h = hashlib.sha256(body + b'|' + strings).hexdigest()
            out.append((h, length, len(strings)))
        if t == 127:  # end-of-table
            break
        i = j + 2
    return out


def collect_dimm():
    """Hash of all Type-17 structures (vendor + serial + size + speed)."""
    try:
        with open(DMI_FILE, 'rb') as f:
            blob = f.read()
    except PermissionError:
        return None
    except Exception:
        return None
    structs = _scan_type17(blob)
    if not structs:
        return None
    big = '|'.join(h for h, _, _ in structs)
    return {
        'n_dimm': len(structs),
        'sha256': hashlib.sha256(big.encode()).hexdigest(),
        'total_body_bytes': sum(L for _, L, _ in structs),
        'total_str_bytes': sum(S for _, _, S in structs),
    }


def measure_latency(buf_mb=128, n_chases=200_000):
    """Build a random-permutation pointer chase over a buf_mb buffer.
    The mean access time is dominated by DRAM Trcd+Trp+Tcl + channel routing.
    Per-channel UMC training adds a tiny but board-stable bias.
    """
    thermal_guard()
    sz = buf_mb * 1024 * 1024
    n = sz // 8
    arr = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(0xC0FFEE)
    rng.shuffle(arr)
    # build permutation: arr[i] is the next index
    # pointer chase: idx = arr[idx]
    idx = 0
    iters = min(n_chases, n // 2)
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        idx = arr[idx]
    t1 = time.perf_counter_ns()
    mean_ns = (t1 - t0) / iters
    # sink to defeat DCE
    return mean_ns, int(idx)


def featurize(dimm, lat_runs):
    feats = []
    # DIMM hash projected to 8 floats
    if dimm:
        h = bytes.fromhex(dimm['sha256'])[:16]
        for i in range(8):
            feats.append(float(h[i]))
        feats += [float(dimm['n_dimm']),
                  float(dimm['total_body_bytes']),
                  float(dimm['total_str_bytes'])]
    else:
        feats += [0.0]*11
    # Latency stats
    a = np.asarray(lat_runs, dtype=np.float64)
    feats += [
        float(a.mean()), float(a.std()),
        float(np.percentile(a, 10)), float(np.percentile(a, 50)),
        float(np.percentile(a, 90)),
        float(a.max() - a.min()),
        float(a.size),
    ]
    return np.asarray(feats[:DIM], dtype=np.float64)


def run(reps=10, out_dir=None):
    host = hostname()
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment20'))
    os.makedirs(out_dir, exist_ok=True)
    vecs = np.zeros((reps, DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': DIM, 'signal': 's12_ddr_residual',
            't_start': time.time(), 'rep_seconds': [],
            'temp_start': get_apu_temp_c()}
    dimm = collect_dimm()
    meta['dimm'] = dimm
    print(f"[s12] host={host} reps={reps} dimm_ok={bool(dimm)}", flush=True)
    for r in range(reps):
        wait_cool(target_c=58, timeout_s=60)
        thermal_guard()
        t0 = time.time()
        try:
            # 4 latency samples per rep so we can extract variance
            lat_runs = []
            for _ in range(4):
                ns, _ = measure_latency(buf_mb=64, n_chases=120_000)
                lat_runs.append(ns)
            vecs[r] = featurize(dimm, lat_runs)
        except SystemExit as e:
            print(f"[s12] abort rep {r}: {e}"); vecs = vecs[:r]; break
        meta['rep_seconds'].append(time.time() - t0)
        print(f"[s12] rep {r+1}/{reps} {meta['rep_seconds'][-1]:.1f}s "
              f"lat_mean={vecs[r,11]:.1f}ns", flush=True)
    out = os.path.join(out_dir, f'{host}_s12.npz')
    np.savez(out, vec=vecs, host=host, dim=DIM)
    save_json(os.path.join(out_dir, f'{host}_s12_meta.json'), meta)
    print(f"[s12] saved {out}"); return out


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run(reps=reps)
