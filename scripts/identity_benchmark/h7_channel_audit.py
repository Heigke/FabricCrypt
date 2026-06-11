"""Deep channel audit — measure 5 embodiment properties per channel:

  1. Cross-host discriminativity (Cohen's d ikaros vs daedalus)
  2. Self-predictability (ridge R², persistence R²)
  3. Compute-load coupling (Δ per channel between busy GPU vs idle GPU)
  4. Spoof-resistance (REAL-vs-matched-spectrum-spoof classifiability via higher moments)
  5. Non-trivial predictability gap (ridge R² − persistence R²)

A channel is EMBODIMENT-BEARING only if:
  - (1) > 1.0 (clear cross-host signal) OR not measurable at this stage
  - (3) > 0.5σ (compute-load actually moves the channel — closed loop exists)
  - (4) > 0.85 (spoof-resistant: matched-spectrum can't fake it)
  - (5) > 0.05 (model adds value over persistence — channel has dynamics, not just constant)

Drop channels failing 2+ criteria. Report final embodiment-bearing channel set.
"""
from __future__ import annotations
import json, sys, time, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import GlobalNorm

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/IDENTITY_H7_2026-06-09"
STATS = OUT_DIR / "global_substrate_stats.npz"
CH_NAMES = ["C07_xtal","C09_pm1","C20_lat_x","C20_logtl","C11_drift",
            "C05_e0_rt","C06_fast","C09_pm3","C09_pm5","C20_lat_e"]
N_CH = 10


def collect(seconds, label=""):
    """Collect live substrate. Returns (T, 10) numpy."""
    print(f"  [{label}] collecting {seconds}s live...")
    state = SubstrateStateV3(hz_target=500); state.start()
    time.sleep(seconds)
    # Take full 4096-step buffer
    w = state.latest_window(length=4096)
    state.stop()
    return w


def per_ch_r2(pred, target):
    ss_res = ((pred - target)**2).mean(axis=0)
    ss_tot = ((target - target.mean(axis=0))**2).mean(axis=0) + 1e-9
    return 1 - ss_res / ss_tot


def matched_spoof(w, rng):
    """μ-σ-φ matched AR(1) spoof per channel."""
    out = np.zeros_like(w)
    for c in range(w.shape[1]):
        x = w[:, c]
        mu, sg = x.mean(), x.std()
        phi = np.corrcoef(x[:-1], x[1:])[0,1] if x.std() > 0 else 0
        eps = rng.normal(0, sg*np.sqrt(max(1e-6, 1-phi**2)), len(x))
        y = np.zeros_like(x); y[0] = x[0]
        for t in range(1, len(x)):
            y[t] = phi*y[t-1] + eps[t]
        y = y - y.mean() + mu
        out[:, c] = y
    return out


def gpu_busy_thread(seconds):
    """Run a tight loop on ROCm-PyTorch (or CPU fallback) to create compute load."""
    end = time.time() + seconds
    if torch.cuda.is_available():
        x = torch.randn(2048, 2048, device="cuda")
        while time.time() < end:
            x = torch.matmul(x, x.T); torch.cuda.synchronize()
            x = (x - x.mean()) / (x.std() + 1e-3)
    else:
        # CPU fallback
        import numpy as _np
        x = _np.random.randn(1024, 1024)
        while time.time() < end:
            x = x @ x.T
            x = (x - x.mean()) / (x.std() + 1e-3)


def main():
    norm = GlobalNorm(STATS)
    rng = np.random.default_rng(7)

    print("=" * 70)
    print("DEEP CHANNEL AUDIT — 5 embodiment properties per channel")
    print("=" * 70)

    # 1. Idle window (baseline)
    print("\n[1/4] IDLE 60s collection")
    w_idle_raw = collect(60, "idle")
    w_idle = norm(w_idle_raw)

    # 2. Busy GPU window
    print("\n[2/4] BUSY-GPU 60s collection (with concurrent compute load)")
    import threading
    t = threading.Thread(target=gpu_busy_thread, args=(65,), daemon=True)
    t.start()
    time.sleep(2)   # let load ramp
    w_busy_raw = collect(60, "busy")
    w_busy = norm(w_busy_raw)
    t.join(timeout=10)

    # 3. Cross-host data — use saved daedalus replay (6-ch padded, sub-optimal but only option for now)
    print("\n[3/4] CROSS-HOST data (use saved daedalus replay, 6-ch padded)")
    try:
        da = np.load(OUT_DIR / "substrate_replay_daedalus.npz")["windows"]
        pad = np.zeros((*da.shape[:-1], 4), dtype=da.dtype)
        da10 = np.concatenate([da, pad], axis=-1)
        da_seq = np.concatenate([norm(w) for w in da10], axis=0)
    except Exception as e:
        print(f"  WARN: no daedalus replay ({e}); cross-host props degraded")
        da_seq = None

    print(f"\n[4/4] computing 5 properties × {N_CH} channels...")

    results = {n: {} for n in CH_NAMES}

    # === PROP 1: Cross-host discriminativity (Cohen's d) ===
    if da_seq is not None:
        for i, n in enumerate(CH_NAMES):
            ik = w_idle[:, i]; dz = da_seq[:, i]
            sp = np.sqrt(((len(ik)-1)*ik.std(ddof=1)**2 + (len(dz)-1)*dz.std(ddof=1)**2) / (len(ik)+len(dz)-2))
            d = (ik.mean() - dz.mean()) / (sp + 1e-9)
            results[n]["cross_host_d"] = abs(d)
    else:
        for n in CH_NAMES:
            results[n]["cross_host_d"] = None

    # === PROP 2: Self-predictability ===
    big = w_idle
    N = len(big); split = int(0.8 * N)
    # persistence
    r2_pers = per_ch_r2(big[split:-1], big[split+1:])
    # ridge 32-lag
    def make_lag(seq, lags=32):
        feats, targs = [], []
        for t in range(lags, len(seq)-1):
            feats.append(seq[t-lags:t].flatten()); targs.append(seq[t])
        return np.array(feats), np.array(targs)
    Xl_tr, Yl_tr = make_lag(big[:split], lags=32)
    Xl_te, Yl_te = make_lag(big[split:], lags=32)
    lam = 1e-2 * len(Xl_tr)
    W = np.linalg.solve(Xl_tr.T @ Xl_tr + lam*np.eye(Xl_tr.shape[1]), Xl_tr.T @ Yl_tr)
    pred_te = Xl_te @ W
    r2_ridge = per_ch_r2(pred_te, Yl_te)
    for i, n in enumerate(CH_NAMES):
        results[n]["r2_persistence"] = float(r2_pers[i])
        results[n]["r2_ridge"] = float(r2_ridge[i])
        results[n]["r2_gap"] = float(r2_ridge[i] - max(r2_pers[i], 0))

    # === PROP 3: Compute-load coupling (busy vs idle delta per channel) ===
    for i, n in enumerate(CH_NAMES):
        # Compare distributions: idle vs busy
        ik = w_idle[:, i]; bk = w_busy[:, i]
        sp = np.sqrt(((len(ik)-1)*ik.std(ddof=1)**2 + (len(bk)-1)*bk.std(ddof=1)**2) / (len(ik)+len(bk)-2))
        d_load = (bk.mean() - ik.mean()) / (sp + 1e-9)
        results[n]["compute_load_d"] = abs(d_load)
        # Also report variance ratio
        results[n]["var_ratio_busy_idle"] = float(bk.std() / (ik.std() + 1e-6))

    # === PROP 4: Spoof-resistance ===
    # Higher-moment features per channel
    def hi_features(seq, win=64):
        F = []
        for i in range(0, len(seq) - win, win):
            w = seq[i:i+win]
            F.append([w.mean(), w.std(),
                      stats.skew(w) if w.std()>0 else 0,
                      stats.kurtosis(w) if w.std()>0 else 0,
                      np.corrcoef(w[:-1], w[1:])[0,1] if w.std()>0 else 0])
        return np.array(F)

    for i, n in enumerate(CH_NAMES):
        real_seq = w_idle[:, i]
        spoof_seq = matched_spoof(w_idle, rng)[:, i]
        F_real = hi_features(real_seq); F_spoof = hi_features(spoof_seq)
        if len(F_real) < 4 or len(F_spoof) < 4:
            results[n]["spoof_resistance"] = None; continue
        X = np.concatenate([F_real, F_spoof]); y = np.concatenate([np.zeros(len(F_real)), np.ones(len(F_spoof))])
        mu, sd = X.mean(0), X.std(0)+1e-6
        Z = (X - mu) / sd
        if len(X) >= 10:
            clf = LogisticRegression(max_iter=2000)
            try:
                acc = cross_val_score(clf, Z, y, cv=3).mean()
                results[n]["spoof_resistance"] = float(acc)
            except Exception:
                results[n]["spoof_resistance"] = None
        else:
            results[n]["spoof_resistance"] = None

    # Print table
    print(f"\n=== CHANNEL AUDIT RESULTS ===\n")
    print(f"{'channel':14s} {'host_d':>7s} {'r2_pers':>8s} {'r2_ridge':>9s} {'r2_gap':>7s} {'load_d':>7s} {'varratio':>9s} {'spoofresist':>12s}")
    print("-" * 96)
    for i, n in enumerate(CH_NAMES):
        r = results[n]
        host = f"{r['cross_host_d']:+.2f}" if r['cross_host_d'] is not None else "  n/a"
        sp_r = f"{r['spoof_resistance']:.3f}" if r['spoof_resistance'] is not None else "n/a"
        print(f"{n:14s} {host:>7s} {r['r2_persistence']:+.3f} {r['r2_ridge']:+.3f} {r['r2_gap']:+.3f} "
              f"{r['compute_load_d']:+.2f} {r['var_ratio_busy_idle']:>9.2f} {sp_r:>12s}")

    # Score each channel as embodiment-bearing
    print(f"\n=== EMBODIMENT-BEARING VERDICT ===")
    print(f"Criteria: (host_d > 1.0) + (load_d > 0.5) + (spoof_resist > 0.85) + (r2_gap > 0.05)")
    print()
    keepers = []
    for i, n in enumerate(CH_NAMES):
        r = results[n]
        criteria = {
            "discr": (r["cross_host_d"] is None) or (r["cross_host_d"] > 1.0),
            "load":  r["compute_load_d"] > 0.5,
            "spoof": (r["spoof_resistance"] is not None) and (r["spoof_resistance"] > 0.85),
            "dyn":   r["r2_gap"] > 0.05,
        }
        pass_count = sum(criteria.values())
        status = "★ KEEP" if pass_count >= 3 else ("? marginal" if pass_count == 2 else "✗ DROP")
        flags = "".join("●" if criteria[k] else "○" for k in ["discr","load","spoof","dyn"])
        print(f"  ch{i} {n:14s} {flags} [{pass_count}/4] {status}")
        if pass_count >= 3:
            keepers.append((i, n))

    print(f"\n=> EMBODIMENT-BEARING CHANNELS: {len(keepers)} of {N_CH}")
    for i, n in keepers:
        print(f"  ch{i} {n}")

    out_path = OUT_DIR / "channel_audit_2026-06-10.json"
    out_path.write_text(json.dumps({"results": results, "keepers": keepers}, indent=2, default=str))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
