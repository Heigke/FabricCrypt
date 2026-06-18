"""Phase 14C Task D — signature acquisition timing.

Goal: 1000 fresh-nonce reads, all <5 ms each.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P13 = os.path.abspath(os.path.join(HERE, '..', 'embodiment13'))
sys.path.insert(0, P13)

from common13 import thermal_guard, hostname, save_json
from nonce_signature import NonceSig, fresh_nonce


def main():
    host = hostname()
    out_dir = os.path.abspath(os.path.join(
        HERE, '..', '..', '..', 'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14c'))
    os.makedirs(out_dir, exist_ok=True)

    sig = NonceSig(host=host)
    rng = np.random.default_rng(0)
    # warm
    for _ in range(50): sig.read(fresh_nonce(rng))

    N = 1000
    times = np.empty(N, dtype=np.float64)
    for i in range(N):
        if (i % 16) == 0: thermal_guard()
        nb = fresh_nonce(rng)
        t0 = time.perf_counter()
        _ = sig.read(nb)
        times[i] = (time.perf_counter() - t0) * 1e3  # ms

    stats = {
        'host': host,
        'n': N,
        'mean_ms':  float(times.mean()),
        'median_ms': float(np.median(times)),
        'p95_ms':   float(np.percentile(times, 95)),
        'p99_ms':   float(np.percentile(times, 99)),
        'max_ms':   float(times.max()),
        'min_ms':   float(times.min()),
        'budget_ms': 5.0,
        'pass_p99':  bool(np.percentile(times, 99) < 5.0),
        'pass_max':  bool(times.max() < 5.0),
    }
    save_json(os.path.join(out_dir, f'{host}_timing.json'), stats)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
