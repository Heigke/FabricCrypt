"""Task E: RDRAND latency variance via Python ctypes inline.

We bracket RDRAND with clock_gettime calls. (RDTSC bracketing in pure Python
not available without C extension; clock_gettime MONOTONIC_RAW gives ~20 ns
resolution which is enough to see RDRAND latency variance.)

1M calls is too many in pure-python; use 200k which still gives strong KS power.
"""
import os
import sys
import time
import ctypes
import ctypes.util
import numpy as np
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from common import thermal_guard, save_json, hostname

C_SRC = r'''
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdlib.h>
#include <x86intrin.h>

int main(int argc, char** argv) {
    int N = atoi(argv[1]);
    uint64_t *lat = (uint64_t*)malloc(sizeof(uint64_t)*N);
    unsigned long long v;
    unsigned int ok;
    uint64_t t0,t1;
    unsigned aux;
    // warmup
    for (int i=0;i<10000;i++) ok=_rdrand64_step(&v);
    for (int i=0;i<N;i++) {
        t0 = __rdtscp(&aux);
        ok = _rdrand64_step(&v);
        t1 = __rdtscp(&aux);
        lat[i] = t1 - t0;
        (void)ok;
    }
    // print as binary to stdout
    fwrite(lat, sizeof(uint64_t), N, stdout);
    free(lat);
    return 0;
}
'''


def build_and_run(n):
    tmp = tempfile.mkdtemp(prefix='rdrand_')
    src = os.path.join(tmp, 'r.c')
    binp = os.path.join(tmp, 'r')
    with open(src, 'w') as f:
        f.write(C_SRC)
    r = subprocess.run(['gcc', '-O2', '-mrdrnd', '-o', binp, src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("compile fail:", r.stderr)
        sys.exit(1)
    res = subprocess.run([binp, str(n)], capture_output=True)
    if res.returncode != 0:
        print("run fail")
        sys.exit(1)
    return np.frombuffer(res.stdout, dtype=np.uint64).copy()


def summarize(lat):
    return {
        'n': int(lat.size),
        'mean_cyc': float(lat.mean()),
        'std_cyc': float(lat.std()),
        'p50': float(np.percentile(lat, 50)),
        'p90': float(np.percentile(lat, 90)),
        'p99': float(np.percentile(lat, 99)),
        'p99_9': float(np.percentile(lat, 99.9)),
        'p99_99': float(np.percentile(lat, 99.99)),
        'min': int(lat.min()),
        'max': int(lat.max()),
    }


def main():
    N = 1_000_000
    host = hostname()
    print(f"[E] host={host} RDRAND N={N}")
    thermal_guard()
    t0 = time.time()
    lat = build_and_run(N)
    print(f"[E] done in {time.time()-t0:.1f}s, got {lat.size} samples")
    thermal_guard()

    out = {
        'host': host,
        'N': int(lat.size),
        'rdrand_cycles': summarize(lat),
        'raw_samples_cyc': lat[::10].tolist(),  # 100k subsample
    }
    out_path = f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment12/task_E_rdrand_{host}.json"
    save_json(out_path, out)


if __name__ == '__main__':
    main()
