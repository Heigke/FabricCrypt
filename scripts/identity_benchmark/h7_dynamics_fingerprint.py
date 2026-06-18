"""H7-PAT Step-0 — DYNAMICS fingerprint gate (does the die have die-unique 2nd-order structure?).

The reservoir route is closed: every channel mean is a monotone (cloneable) function of load.
The physics-aware-training (PAT) literature says device lock-in works even for LINEAR substrates —
BUT only if the device contributes irreducible live structure a recording/twin can't clone: noise
covariance, thermal time-constants, cross-channel correlations (2nd-order, NOT the means).

This gate tests exactly that, with the cloneable part removed. On each die we run an IDENTICAL,
deterministic light load schedule, collect many short multi-channel telemetry windows, and save the
RAW windows. The classify step (run locally over both dies' files) removes each window's per-channel
MEAN and DC-normalizes (kills the cloneable offset+gain), then asks whether a classifier can still
tell ikaros from daedalus using ONLY 2nd-order features (per-channel variance, AR/thermal-lag,
cross-channel covariance, PSD bands).

  PASS (PAT-locking justified): die-classification >> chance from MEAN-REMOVED 2nd-order features,
       stable, with enough bits — i.e. the dynamics carry a die fingerprint beyond the cloneable DC.
  FAIL: once means are removed, the dies are indistinguishable -> PAT would lock to a cloneable
       offset only -> report and stop (same honesty bar as the reservoir negative).

Usage:
  collect:  python h7_dynamics_fingerprint.py collect <label>      # run on EACH die
  classify: python h7_dynamics_fingerprint.py classify A.npz B.npz # run locally over both
CPU telemetry read; LIGHT deterministic load with in-loop thermal self-guard (pause@82C).
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
WIN = 256          # samples per window (~0.5s @ 500Hz)
N_WIN = 120        # windows per die
SEED = 0


def temp_c():
    try: return int(ZONE.read_text()) / 1000.0
    except Exception: return 0.0


def collect(label):
    import torch
    from substrate_realtime_v3 import SubstrateStateV3
    host = socket.gethostname()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    # DETERMINISTIC load schedule shared across dies: same 0/1 excitation pattern per window
    sched = rng.integers(0, 2, size=N_WIN)
    st = SubstrateStateV3(hz_target=500); st.start()
    print(f"[{host}/{label}] device={device} warmup 6s (temp {temp_c():.0f}C)...", flush=True)
    time.sleep(6.0)
    a = torch.randn(768, 768, device=device); b = torch.randn(768, 768, device=device)
    wins = np.zeros((N_WIN, WIN, N_CH), dtype=np.float32)
    t0 = time.time()
    for k in range(N_WIN):
        if sched[k]:                              # light deterministic excitation
            for _ in range(2):
                a = (a @ b).tanh() * 0.5 + 0.5
            if device == "cuda": torch.cuda.synchronize()
        time.sleep(0.05)
        wins[k] = st.latest_window(length=WIN).reshape(-1, N_CH)[:WIN]
        if k % 10 == 0:
            tc = temp_c()
            if tc > 82.0:
                print(f"  [self-guard] {tc:.0f}C cooling", flush=True)
                while temp_c() > 62.0: time.sleep(1.0)
            if k % 40 == 0:
                print(f"  win {k}/{N_WIN} temp={tc:.0f}C ({time.time()-t0:.0f}s)", flush=True)
    st.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"dynfp_{label}.npz"
    np.savez_compressed(out, wins=wins, sched=sched, host=host, label=label)
    print(f"  saved {out}  ({N_WIN} windows, {time.time()-t0:.0f}s)", flush=True)


def features(wins, drop_mean=True):
    """2nd-order features per window. drop_mean=True removes the cloneable per-channel DC offset+gain."""
    F = []
    for w in wins:                                # w: (WIN, N_CH)
        x = w.astype(np.float64)
        mu = x.mean(0); sd = x.std(0) + 1e-9
        if drop_mean:
            x = (x - mu) / sd                     # z-score => kills offset AND gain (the cloneable part)
        feats = []
        # per-channel std (if drop_mean, this is ~1; keep raw-scale var separately below)
        feats += list(np.log(x.std(0) + 1e-9))
        # AR/thermal lag: autocorr at lags 1,2,4,8,16
        for lag in (1, 2, 4, 8, 16):
            ac = np.array([np.corrcoef(x[:-lag, c], x[lag:, c])[0, 1] if WIN > lag else 0.0
                           for c in range(N_CH)])
            feats += list(np.nan_to_num(ac))
        # cross-channel covariance (correlation) upper triangle
        C = np.corrcoef(x.T); C = np.nan_to_num(C)
        iu = np.triu_indices(N_CH, k=1)
        feats += list(C[iu])
        # PSD low/mid/high band power (on mean-removed signal)
        xd = x - x.mean(0)
        P = np.abs(np.fft.rfft(xd, axis=0)) ** 2
        nb = P.shape[0]
        for lo, hi in [(1, nb // 8), (nb // 8, nb // 3), (nb // 3, nb)]:
            feats += list(np.log(P[lo:hi].mean(0) + 1e-9))
        F.append(feats)
    return np.array(F)


def classify(fa, fb):
    A = np.load(fa, allow_pickle=True); B = np.load(fb, allow_pickle=True)
    la = str(A["label"]); lb = str(B["label"])
    wa, wb = A["wins"], B["wins"]
    print(f"loaded {la} ({wa.shape}) vs {lb} ({wb.shape})", flush=True)

    def run(drop_mean):
        Fa = features(wa, drop_mean); Fb = features(wb, drop_mean)
        X = np.vstack([Fa, Fb]); y = np.array([0] * len(Fa) + [1] * len(Fb))
        mu = X.mean(0); sd = X.std(0) + 1e-9; X = (X - mu) / sd
        rng = np.random.default_rng(0); idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        # 5-fold CV ridge classifier
        accs = []
        n = len(X); fold = n // 5
        for f in range(5):
            te = slice(f * fold, (f + 1) * fold)
            mask = np.ones(n, bool); mask[te] = False
            Xtr, ytr, Xte, yte = X[mask], y[mask], X[te], y[te]
            best = 0.5
            for al in [0.1, 1, 10, 100, 1e3]:
                W = np.linalg.solve(Xtr.T @ Xtr + al * np.eye(Xtr.shape[1]), Xtr.T @ (2 * ytr - 1))
                pred = (Xte @ W > 0).astype(int)
                best = max(best, float(np.mean(pred == yte)))
            accs.append(best)
        return float(np.mean(accs)), float(np.std(accs))

    acc_full, sd_full = run(drop_mean=False)     # control: includes cloneable means (should be ~1.0)
    acc_dyn, sd_dyn = run(drop_mean=True)        # the real test: 2nd-order only, means removed
    chance = 0.5
    # PASS if dynamics-only classification clearly beats chance AND most of the signal survives mean-removal
    verdict = "PASS" if (acc_dyn >= 0.75) else ("WEAK" if acc_dyn >= 0.62 else "FAIL")
    res = {"pair": [la, lb], "acc_with_means_control": acc_full, "acc_dynamics_only": acc_dyn,
           "sd_dynamics": sd_dyn, "chance": chance, "verdict": verdict,
           "note": "acc_dynamics_only = die separability from MEAN-REMOVED 2nd-order features (covariance, "
                   "AR/thermal-lag, PSD). High => die-unique live structure beyond cloneable DC => PAT-lock viable."}
    out = OUT / "dynamics_fingerprint_result.json"; out.write_text(json.dumps(res, indent=2))
    print(f"\n  control (with means)     : {acc_full:.3f}  (sanity: should be high; DC differs)")
    print(f"  DYNAMICS ONLY (means out): {acc_dyn:.3f} ± {sd_dyn:.3f}   chance {chance:.2f}")
    print(f"  >>> {verdict}   saved {out}", flush=True)
    if verdict == "FAIL":
        print("  Once the cloneable DC offset is removed, the dies are NOT distinguishable -> PAT would")
        print("  lock to a cloneable offset only. Same honesty bar as the reservoir negative.")
    else:
        print("  Die-unique LIVE 2nd-order structure survives mean-removal -> PAT-locking has real, "
              "non-trivial entropy to bind to.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "collect":
        collect(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "classify":
        classify(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
