"""H7 die-PARAMETERIZED computation — proof of concept on REAL CPPC keys (track 2 core).

The reframe that works: the die's reproducible per-core CPPC fingerprint becomes the WEIGHTS of a fixed nonlinear
reservoir the task needs. Train a linear readout on die-A's reservoir; test on die-A's kernel (match) vs die-B's
kernel (wrong die) vs shuffled key. If die-A >> die-B/shuffle, the COMPUTATION is die-bound — räkna-unikt at the
system level, via the channel that is genuinely reproducible (static binning, intra=1.0, 75% distinct).
Uses fmax_enroll_{host}_r1.npz CPPC keys. CPU-only, no thermal load.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
rng = np.random.default_rng(0)


def die_key(host):
    d = np.load(OUT/f"fmax_enroll_{host}_r1.npz")
    hp = d["runs"].mean(0)[:, 0]              # per-core CPPC highest_perf (32-dim, firmware-constant)
    return (hp - hp.mean())/(hp.std()+1e-9)


def reservoir_states(X, key, Dh=256):
    """Fixed random reservoir whose per-hidden gain+bias are PARAMETERIZED by the die key."""
    Din = X.shape[1]
    Win = rng.standard_normal((Dh, Din))*0.6
    # map die key (NCPU) -> hidden dim by tiling/interp; it sets per-unit gain and bias (the die's imprint)
    k = np.interp(np.linspace(0, 1, Dh), np.linspace(0, 1, len(key)), key)
    gain = 1.0 + 0.5*k; bias = 0.4*np.roll(k, 7)
    pre = X @ Win.T * gain[None, :] + bias[None, :]
    return np.tanh(pre)


def make_task(n=4000, Din=8):
    X = rng.standard_normal((n, Din))
    # nonlinear target that NEEDS the reservoir (products + threshold)
    y = np.sign(X[:, 0]*X[:, 1] + 0.7*np.tanh(3*X[:, 2]) - 0.5*X[:, 3]*X[:, 4])
    return X, (y > 0).astype(float)


def fit_readout(H, y):
    Hb = np.concatenate([H, np.ones((len(H), 1))], 1)
    w, *_ = np.linalg.lstsq(Hb, 2*y-1, rcond=None)
    return w
def acc(H, y, w):
    Hb = np.concatenate([H, np.ones((len(H), 1))], 1)
    return float(((Hb@w > 0).astype(float) == y).mean())


def main():
    hosts = [Path(p).stem.split("fmax_enroll_")[1].rsplit("_r", 1)[0]
             for p in sorted(glob.glob(str(OUT/"fmax_enroll_*_r1.npz")))]
    hosts = sorted(set(hosts))
    print("dies with CPPC keys:", hosts)
    if len(hosts) < 2:
        print("need 2 dies enrolled (run h7_fmax_enroll on both)"); return
    A, B = hosts[0], hosts[1]
    kA, kB = die_key(A), die_key(B)
    kS = rng.permutation(kA.copy())          # shuffled key (same values, scrambled assignment)
    print(f"key cos(A,B)={float(kA@kB/(np.linalg.norm(kA)*np.linalg.norm(kB))):+.3f}", flush=True)

    Xtr, ytr = make_task(); Xte, yte = make_task(3000)
    HA_tr = reservoir_states(Xtr, kA); w = fit_readout(HA_tr, ytr)   # readout trained on die-A kernel

    res = {}
    for name, key in [("die-A (match)", kA), ("die-B (wrong die)", kB), ("shuffled key", kS), ("zero key", kA*0)]:
        H = reservoir_states(Xte, key)
        res[name] = acc(H, yte, w)
    base = res["die-A (match)"]
    print(f"\nreadout trained on {A}; test accuracy with each kernel:")
    for k, v in res.items():
        drop = base - v
        print(f"  {k:22s} acc={v:.3f}  drop={drop:+.3f}", flush=True)
    real = (base - res["die-B (wrong die)"] > 0.05) and (base - res["shuffled key"] > 0.05)
    print(f"\nVERDICT: {'die-PARAMETERIZED computation REAL — die-A works, wrong-die/shuffle break it' if real else 'kernels too similar — no die-binding'}", flush=True)
    print(f"  (die-A {base:.3f} vs die-B {res['die-B (wrong die)']:.3f} vs shuffle {res['shuffled key']:.3f})", flush=True)


if __name__ == "__main__":
    main()
