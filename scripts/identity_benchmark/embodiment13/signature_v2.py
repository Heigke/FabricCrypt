#!/usr/bin/env python3
"""Phase 13 Task A — signature_v2 extractor.

Produces a 290-dim per-die fingerprint vector from 5 verified signals
(Phase 12+12B per-die signatures):

  Block 1 (TSC inter-core, 7 targets x 5 stats = 35)
  Block 2 (Cacheline pingpong, 7 pairs x 5 stats = 35)
  Block 3 (DRAM refresh-probing histogram, 200 bins)
  Block 4 (Syscall p99.9 tail, 10 percentile features)
  Block 5 (NVMe queue tail, 10 percentile features)

Output: per-rep npz: vec (R, 290), labels (R,), block_starts (5,)

Strict thermal: abort 68, pause 63, cool 50. Each rep ~30-40s.
"""
import os, sys, time, struct, ctypes, subprocess, math, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common13 import (thermal_guard, wait_cool, save_json, hostname,
                      compile_c, get_apu_temp_c)

# 7 TSC targets (covering different CCD/CCX positions on Strix-Halo 16-core)
TSC_TARGETS = [1, 2, 4, 7, 8, 12, 15]
# 7 cacheline-pingpong pairs (same set used in Phase 12B)
CL_PAIRS = [(0,1), (0,2), (0,4), (0,7), (0,8), (0,15), (0,16)]
# DRAM histogram: 200 bins, log-spaced over ns latency
DRAM_NBINS = 200
DRAM_BIN_EDGES = np.logspace(1.0, 4.0, DRAM_NBINS + 1)  # 10ns..10us
# Block dims
DIMS = {'tsc': 35, 'cacheline': 35, 'dram': 200, 'nanosleep': 10, 'nvme': 10}
TOTAL_DIM = sum(DIMS.values())  # 290
BLOCK_STARTS = {'tsc': 0, 'cacheline': 35, 'dram': 70,
                'nanosleep': 270, 'nvme': 280}

# ---- helpers ----
def pcts(arr, ps=(50, 90, 99, 99.9, 99.99)):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0: return [0.0]*len(ps)
    return [float(np.percentile(a, p)) for p in ps]

def features_5stats(arr):
    """Return [p50, p90, p99, std, mean] of an array."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0: return [0.0]*5
    return [float(np.percentile(a, 50)),
            float(np.percentile(a, 90)),
            float(np.percentile(a, 99)),
            float(np.std(a)),
            float(np.mean(a))]

# ---- block extractors ----
def block_tsc(binp, n_per_target=2000):
    feats = []
    for tgt in TSC_TARGETS:
        thermal_guard()
        p = subprocess.run([binp, str(tgt), str(n_per_target)],
                           capture_output=True, check=True)
        data = struct.unpack(f'{len(p.stdout)//8}q', p.stdout)
        offsets = [data[2*i+1] - data[2*i] for i in range(n_per_target)]
        feats.extend(features_5stats(offsets))
    return np.asarray(feats, dtype=np.float64)

def block_cacheline(binp, iters=4000):
    feats = []
    for (a,b) in CL_PAIRS:
        thermal_guard()
        p = subprocess.run([binp, str(a), str(b), str(iters)],
                           capture_output=True, check=True)
        rtt = struct.unpack(f'{len(p.stdout)//8}Q', p.stdout)
        feats.extend(features_5stats(rtt))
    return np.asarray(feats, dtype=np.float64)

def block_dram(n_samples=100000, size_mb=128):
    """DRAM walk -> histogram (log-binned, 200 bins) of access latencies (ns)."""
    thermal_guard()
    import random
    SIZE = size_mb * 1024 * 1024
    arr = bytearray(SIZE)
    rng = random.Random(0xBEEF + os.getpid())
    # warm map
    for i in range(0, SIZE, 4096):
        arr[i] = 1
    positions = [rng.randrange(0, SIZE-64) & ~63 for _ in range(n_samples)]
    samples = np.empty(n_samples, dtype=np.int64)
    perf = time.perf_counter_ns
    for i, p in enumerate(positions):
        t0 = perf()
        _ = arr[p]
        samples[i] = perf() - t0
    hist, _ = np.histogram(samples, bins=DRAM_BIN_EDGES)
    # normalise to fraction (avoid scale dominating)
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        hist = hist.astype(np.float64)
    return hist  # (200,)

def block_nanosleep(n=20000, sleep_ns=1000):
    """10 features: 10 percentiles from p50..p99.99."""
    thermal_guard()
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    class TS(ctypes.Structure):
        _fields_=[("s",ctypes.c_long),("ns",ctypes.c_long)]
    ts = TS(0, sleep_ns)
    out = np.empty(n, dtype=np.int64)
    perf = time.perf_counter_ns
    for i in range(n):
        t0 = perf()
        libc.nanosleep(ctypes.byref(ts), None)
        out[i] = perf() - t0
    ps = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99, 99.999, 99.9999]
    return np.asarray([np.percentile(out, p) for p in ps], dtype=np.float64)

def block_nvme(n=10000, path='/usr/bin/python3'):
    thermal_guard()
    fd = os.open(path, os.O_RDONLY)
    try:
        out = np.empty(n, dtype=np.int64)
        perf = time.perf_counter_ns
        for i in range(n):
            os.lseek(fd, 0, 0)
            t0 = perf()
            os.read(fd, 4096)
            out[i] = perf() - t0
    finally:
        os.close(fd)
    ps = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99, 99.999, 99.9999]
    return np.asarray([np.percentile(out, p) for p in ps], dtype=np.float64)

# ---- main per-rep ----
def extract_one(tsc_bin, cl_bin):
    v = np.zeros(TOTAL_DIM, dtype=np.float64)
    v[BLOCK_STARTS['tsc']:BLOCK_STARTS['tsc']+DIMS['tsc']]                = block_tsc(tsc_bin)
    v[BLOCK_STARTS['cacheline']:BLOCK_STARTS['cacheline']+DIMS['cacheline']] = block_cacheline(cl_bin)
    v[BLOCK_STARTS['dram']:BLOCK_STARTS['dram']+DIMS['dram']]             = block_dram()
    v[BLOCK_STARTS['nanosleep']:BLOCK_STARTS['nanosleep']+DIMS['nanosleep']] = block_nanosleep()
    v[BLOCK_STARTS['nvme']:BLOCK_STARTS['nvme']+DIMS['nvme']]             = block_nvme()
    return v

def main(reps=10, out_dir=None):
    if out_dir is None:
        out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
            'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment13'))
    os.makedirs(out_dir, exist_ok=True)
    host = hostname()
    # compile
    tsc_src = os.path.join(HERE, 'tsc_inter_core.c')
    tsc_bin = os.path.join(HERE, 'tsc_inter_core')
    cl_src  = os.path.join(HERE, 'cacheline_pingpong.c')
    cl_bin  = os.path.join(HERE, 'cacheline_pingpong')
    if not os.path.exists(tsc_bin): compile_c(tsc_src, tsc_bin)
    if not os.path.exists(cl_bin):  compile_c(cl_src, cl_bin)

    vecs = np.zeros((reps, TOTAL_DIM), dtype=np.float64)
    meta = {'host': host, 'reps': reps, 'dim': TOTAL_DIM,
            'block_starts': BLOCK_STARTS, 'dims': DIMS,
            't_start': time.time(), 'apu_temp_start_c': get_apu_temp_c(),
            'rep_temps_c': [], 'rep_seconds': []}
    print(f"[sig_v2] host={host} reps={reps} dim={TOTAL_DIM} temp={get_apu_temp_c():.1f}C", flush=True)
    for r in range(reps):
        wait_cool(target_c=55, timeout_s=120)
        t0 = time.time()
        try:
            vecs[r] = extract_one(tsc_bin, cl_bin)
        except SystemExit as e:
            print(f"[sig_v2] thermal abort during rep {r}: {e}", flush=True)
            vecs = vecs[:r]
            break
        meta['rep_temps_c'].append(get_apu_temp_c())
        dt = time.time() - t0
        meta['rep_seconds'].append(dt)
        print(f"[sig_v2] rep {r+1}/{reps} done {dt:.1f}s temp={get_apu_temp_c():.1f}C", flush=True)
    meta['t_end'] = time.time()
    meta['apu_temp_end_c'] = get_apu_temp_c()
    out_npz = os.path.join(out_dir, f'{host}_sig_v2.npz')
    np.savez(out_npz, vec=vecs, host=host, dim=TOTAL_DIM,
             block_starts=np.array([BLOCK_STARTS[k] for k in ['tsc','cacheline','dram','nanosleep','nvme']]))
    save_json(os.path.join(out_dir, f'{host}_sig_v2_meta.json'), meta)
    print(f"[sig_v2] saved {out_npz} shape={vecs.shape}", flush=True)
    return out_npz

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(reps=reps, out_dir=out_dir)
