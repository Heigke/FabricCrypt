"""Phase 14D — collect 500 LiveSig samples at the current governor.

Usage: collect_sigs.py <gov_label>     # e.g. powersave / performance
Saves: results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d/ikaros_sigs_<gov>.npz
"""
import os, sys, time, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P14B = os.path.abspath(os.path.join(HERE, '..', 'embodiment14b'))
sys.path.insert(0, P14B)

from signature_live import LiveSig

OUT = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                   'results', 'IDENTITY_BENCHMARK_2026-05-30',
                                   'embodiment14d'))
os.makedirs(OUT, exist_ok=True)


def collect(n=500, gov_label='unknown'):
    sig = LiveSig()
    # warm
    for _ in range(20): sig.read()
    arr = np.zeros((n, 32), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n):
        arr[i] = sig.read()
        if i % 50 == 0:
            time.sleep(0.005)
    dt = time.perf_counter() - t0
    out_path = os.path.join(OUT, f'ikaros_sigs_{gov_label}.npz')
    np.savez(out_path, sigs=arr, host='ikaros', gov=gov_label, nonce=b'')
    print(f"[collect] {n} sigs in {dt:.2f}s -> {out_path}")
    print(f"[collect] mean={arr.mean():.3f} std={arr.std():.3f} "
          f"per_dim_std={arr.std(0).mean():.3f}")
    return out_path


if __name__ == "__main__":
    gov_label = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
    collect(500, gov_label)
