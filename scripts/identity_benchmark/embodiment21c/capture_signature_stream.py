"""Phase 21C — capture chip signature stream (1 Hz, ~30 min default).

LIGHT thermal load (just reads sensors). Save to .npz for replay on zgx.

Usage:
  python capture_signature_stream.py --duration_s 1800 --out signature_stream_ikaros.npz
"""
from __future__ import annotations
import os, sys, time, argparse, socket
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'embodiment21b'))
from _common import LiveSig, temp_c, hostname  # noqa: E402


def capture(duration_s=1800, hz=1.0, out_path='signature_stream.npz',
            abort_c=60.0):
    period = 1.0 / hz
    sig = LiveSig(nonce=(b'phase21c_' + hostname().encode())[:64])
    # Warm up internal stats
    for _ in range(5):
        sig.read()
    vecs = []
    temps = []
    times = []
    t0 = time.time()
    next_t = t0
    n = int(duration_s * hz)
    print(f"[21c/cap] host={hostname()} duration_s={duration_s} hz={hz} "
          f"n_target={n} T={temp_c():.1f}C abort_c={abort_c}", flush=True)
    aborted = False
    while True:
        now = time.time()
        if (now - t0) >= duration_s:
            break
        t = temp_c()
        if t >= abort_c:
            print(f"[21c/cap] THERMAL ABORT T={t:.1f}C >= {abort_c}", flush=True)
            aborted = True
            break
        v = sig.read()
        vecs.append(v.astype(np.float32))
        temps.append(float(t))
        times.append(float(now - t0))
        if len(vecs) % 60 == 0:
            print(f"[21c/cap] n={len(vecs)} t={now-t0:.0f}s T={t:.1f}C "
                  f"|sig|={np.linalg.norm(v):.2f}", flush=True)
        next_t += period
        sleep_s = next_t - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
    arr = np.stack(vecs).astype(np.float32)
    temps = np.asarray(temps, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez(out_path, sig=arr, temp_c=temps, t_rel=times,
             host=hostname(), aborted=aborted, hz=hz)
    print(f"[21c/cap] WROTE {out_path} shape={arr.shape} "
          f"mean|sig|={np.linalg.norm(arr,axis=1).mean():.2f} "
          f"T_range=[{temps.min():.1f},{temps.max():.1f}]C aborted={aborted}",
          flush=True)
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration_s', type=int, default=1800)
    ap.add_argument('--hz', type=float, default=1.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--abort_c', type=float, default=60.0)
    args = ap.parse_args()
    capture(args.duration_s, args.hz, args.out, args.abort_c)
