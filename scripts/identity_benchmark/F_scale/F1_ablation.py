"""F1 — Scale-up of F (self-referential identity) with ablation of substrate
feature subsets and two strong controls (sw_matched, shuffle).

Setup mirrors scripts/identity_benchmark/novel/F_self_referential.py but:
  * SEEDS = 30 (CAP to 20 if wall budget breached — driven by env var F1_SEEDS)
  * Five feature configs:
       rtn_only      : feature vec = per-CU RTN buckets
       spc_only      : feature vec = top-k eigvals of spatial-corr
       both          : RTN + spatial-corr (current F implementation)
       sw_matched    : SW-RNG matched in (mean, std) per device — same dim as `both`
       shuffle       : `both` features cross-paired with WRONG device label
                       (so the model is told identity X while running on Y)
  * Per ablation × seed × {train,eval} ∈ {ikaros, daedalus}^2
  * Pre-registered gate: aware_gap_mean > sw_matched_gap_mean + 2 * combined_std
                          AND aware_gap_mean > shuffle_gap_mean + 2 * combined_std

We do NOT re-run a substrate-naive baseline here — F base already covers that
and the comparison of interest is now "real-substrate-features vs control-substrate-features".
Each ablation has its OWN baseline-naive run for fair gap subtraction.
"""
from __future__ import annotations
from pathlib import Path
import json, os, sys, time
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "identity_benchmark" / "phase2"))

from narma10_reservoir import (  # noqa: E402
    narma10, build_esn, run_esn, train_ridge, predict, nrmse, ESNConfig,
)
from _substrate_hooks import SubstrateSampler  # noqa: E402

DATA = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"
OUT_DIR = DATA / "F_scale"
OUT = OUT_DIR / "F1_ablation.json"
DEVICES = ["ikaros", "daedalus"]
SEEDS = list(range(int(os.environ.get("F1_SEEDS", "30"))))
WALL_CAP_S = float(os.environ.get("F1_WALL_CAP_S", "3300"))  # 55 min safety

# ----- per-device substrate feature builders -----

def _load(device):
    z = np.load(DATA / device / "raw_idle.npz")
    cyc = np.asarray(z["cu_ts_cyc"], dtype=np.float64)
    rtn = np.asarray(z["rtn"], dtype=np.float64)
    sp  = np.asarray(z["spatial_corr"], dtype=np.float64)
    return cyc, rtn, sp


def feat_rtn_only(device):
    _, rtn, _ = _load(device)
    return rtn.reshape(8, 10).mean(axis=1).astype(np.float64)  # (8,)


def feat_spc_only(device):
    _, _, sp = _load(device)
    eigs = np.sort(np.linalg.eigvalsh(0.5 * (sp + sp.T)))[::-1][:8]
    return (eigs / (np.abs(eigs).max() + 1e-9)).astype(np.float64)  # (8,)


def feat_both(device):
    cyc, rtn, sp = _load(device)
    cu_med = np.median(cyc, axis=1)
    cu_med = (cu_med - cu_med.mean()) / (cu_med.std() + 1e-9)
    cu_buckets = cu_med.reshape(8, 10).mean(axis=1)
    rtn_buckets = rtn.reshape(8, 10).mean(axis=1)
    eigs = np.sort(np.linalg.eigvalsh(0.5 * (sp + sp.T)))[::-1][:8]
    eigs = eigs / (np.abs(eigs).max() + 1e-9)
    return np.concatenate([cu_buckets, rtn_buckets, eigs]).astype(np.float64)  # (24,)


def feat_sw_matched(device, seed):
    """Same shape as `both` (24,), mean+std matched to that device's `both`
    feature vector, but drawn from numpy RNG. Reproducible per (device, seed).
    """
    ref = feat_both(device)
    rng = np.random.default_rng(hash((device, "swmatched", seed)) & 0xFFFFFFFF)
    z = rng.standard_normal(ref.shape[0])
    return (z - z.mean()) / (z.std() + 1e-9) * ref.std() + ref.mean()


def feat_shuffle(device, seed):
    """Cross-paired with WRONG device — preserves "real" feature statistics
    but breaks the identity binding."""
    other = "daedalus" if device == "ikaros" else "ikaros"
    return feat_both(other)  # purely deterministic; seed unused


FEATURE_FNS = {
    "rtn_only":   lambda dev, seed: feat_rtn_only(dev),
    "spc_only":   lambda dev, seed: feat_spc_only(dev),
    "both":       lambda dev, seed: feat_both(dev),
    "sw_matched": feat_sw_matched,
    "shuffle":    feat_shuffle,
}


def run_one(seed, train_dev, eval_dev, feat_key, substrate_aware,
            T_train=2000, T_test=500):
    cfg = ESNConfig(n=128, seed=seed)
    W, Win = build_esn(cfg)
    u, y = narma10(T_train + T_test, seed=seed * 13 + 7)

    train_sub = SubstrateSampler(train_dev, seed=seed + 100)
    X_train = run_esn(u, W, Win, cfg, train_sub)
    if substrate_aware:
        f_train = FEATURE_FNS[feat_key](train_dev, seed)
        Xtr_full = np.concatenate([X_train, np.tile(f_train, (X_train.shape[0], 1))], axis=1)
    else:
        Xtr_full = X_train

    wash = 100
    Xtr = Xtr_full[wash:T_train]; ytr = y[wash:T_train]
    Wout = train_ridge(Xtr, ytr)

    eval_sub = SubstrateSampler(eval_dev, seed=seed + 999)
    X_eval = run_esn(u, W, Win, cfg, eval_sub)
    if substrate_aware:
        f_eval = FEATURE_FNS[feat_key](eval_dev, seed)
        Xev_full = np.concatenate([X_eval, np.tile(f_eval, (X_eval.shape[0], 1))], axis=1)
    else:
        Xev_full = X_eval

    Xte = Xev_full[T_train:]; yte = y[T_train:]
    yhat = predict(Xte, Wout)
    return float(nrmse(yte, yhat))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    n_done = 0
    feats = list(FEATURE_FNS.keys())
    n_total = len(feats) * 2 * len(SEEDS) * len(DEVICES) * len(DEVICES)
    print(f"[F1] feats={feats} seeds={len(SEEDS)} -> {n_total} runs", flush=True)
    aborted = False
    for feat_key in feats:
        for aware in (False, True):
            for train_dev in DEVICES:
                for eval_dev in DEVICES:
                    for s in SEEDS:
                        if time.time() - t0 > WALL_CAP_S:
                            print(f"[F1] WALL CAP reached at {n_done}/{n_total} runs", flush=True)
                            aborted = True
                            break
                        nr = run_one(s, train_dev, eval_dev, feat_key, aware)
                        rows.append({
                            "feat": feat_key, "aware": aware,
                            "train": train_dev, "eval": eval_dev,
                            "seed": s, "nrmse": nr,
                        })
                        n_done += 1
                        if n_done % 40 == 0:
                            print(f"[F1] {n_done}/{n_total}  elapsed={time.time()-t0:.0f}s", flush=True)
                    if aborted: break
                if aborted: break
            if aborted: break
        if aborted: break

    summary = {"per_run": rows, "feats": feats, "n_seeds": len(SEEDS),
               "wall_s": time.time() - t0, "aborted_at_cap": aborted}

    # per-(feat, aware) gap stats
    gaps = {}
    for feat_key in feats:
        for aware in (False, True):
            diffs = []
            for train_dev in DEVICES:
                for s in SEEDS:
                    same = [r["nrmse"] for r in rows
                            if r["feat"] == feat_key and r["aware"] == aware
                            and r["train"] == train_dev and r["eval"] == train_dev
                            and r["seed"] == s]
                    diff = [r["nrmse"] for r in rows
                            if r["feat"] == feat_key and r["aware"] == aware
                            and r["train"] == train_dev and r["eval"] != train_dev
                            and r["seed"] == s]
                    if same and diff:
                        diffs.append(diff[0] - same[0])
            if diffs:
                arr = np.array(diffs)
                gaps[f"{feat_key}|aware={aware}"] = {
                    "n": int(arr.size),
                    "gap_mean": float(arr.mean()),
                    "gap_std":  float(arr.std()),
                    "gap_median": float(np.median(arr)),
                    "gap_iqr":   [float(np.percentile(arr, 25)),
                                  float(np.percentile(arr, 75))],
                }
    summary["gaps"] = gaps

    # Pre-registered gate: substrate-aware "both" gap vs controls
    def _z(a, b):
        if a is None or b is None: return None
        sa = np.sqrt(a["gap_std"]**2 + b["gap_std"]**2 + 1e-12)
        return (a["gap_mean"] - b["gap_mean"]) / (sa + 1e-12)

    g = lambda k: gaps.get(k)
    verdict_block = {}
    for feat in ("rtn_only", "spc_only", "both"):
        aware = g(f"{feat}|aware=True")
        naive = g(f"{feat}|aware=False")
        sw    = g("sw_matched|aware=True")
        shuf  = g("shuffle|aware=True")
        verdict_block[feat] = {
            "aware_gap":  aware,
            "naive_gap":  naive,
            "z_vs_naive":      _z(aware, naive),
            "z_vs_sw_matched": _z(aware, sw),
            "z_vs_shuffle":    _z(aware, shuf),
            "gate_passed": bool(
                aware and sw and shuf
                and aware["gap_mean"] > sw["gap_mean"] + 2 * np.sqrt(aware["gap_std"]**2 + sw["gap_std"]**2 + 1e-12)
                and aware["gap_mean"] > shuf["gap_mean"] + 2 * np.sqrt(aware["gap_std"]**2 + shuf["gap_std"]**2 + 1e-12)
            ),
        }
    summary["verdict"] = verdict_block
    summary["headline"] = {
        "any_feat_gate_passed": any(v["gate_passed"] for v in verdict_block.values()),
        "passers": [k for k, v in verdict_block.items() if v["gate_passed"]],
    }

    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"headline": summary["headline"],
                      "verdict": {k: {kk: vv for kk, vv in v.items()
                                      if kk.startswith("z_") or kk == "gate_passed"}
                                  for k, v in verdict_block.items()}}, indent=2))
    print(f"\n[F1] wrote {OUT}  ({summary['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
