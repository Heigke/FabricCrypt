"""ANGLE F — Self-referential identity (model knows substrate).

Trains a tiny NARMA-10 reservoir whose readout is conditioned on a *substrate
feature vector* (per-CU ΔVth proxy + RTN-rate, summary statistics from
Phase 1b raw_idle.npz). Compares two conditions:

  - SUBSTRATE_AWARE:   features = [reservoir_state, substrate_features]
  - BASELINE:          features = [reservoir_state]

During inference we test each trained model on (a) its OWN device's substrate
and (b) the OTHER device's substrate.

Hypothesis (gate F-DISCOVERY): the substrate-aware model should degrade MORE
when fed the wrong substrate than the baseline degrades (i.e. there is a
substrate-specific commitment that breaks under transplant). Gap > 2σ above
baseline gap.
"""
from __future__ import annotations
from pathlib import Path
import json
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "identity_benchmark" / "phase2"))

from narma10_reservoir import (
    narma10, build_esn, run_esn, train_ridge, predict, nrmse, ESNConfig,
)
from _substrate_hooks import SubstrateSampler

DATA = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"
OUT = DATA / "novel" / "F_results.json"
DEVICES = ["ikaros", "daedalus"]


def substrate_feature_vector(device: str) -> np.ndarray:
    """Compress per-device substrate to a small fixed-size feature vector.
    Inputs: per-CU median cycle time (ΔVth proxy), per-CU RTN rate,
    spatial-corr eigenvalue spectrum (top-k).
    Output: shape (D,) where D is small (~24).
    """
    z = np.load(DATA / device / "raw_idle.npz")
    cyc = np.asarray(z["cu_ts_cyc"], dtype=np.float64)   # (80, 500)
    rtn = np.asarray(z["rtn"], dtype=np.float64)         # (80,)
    sp = np.asarray(z["spatial_corr"], dtype=np.float64) # (80,80)

    cu_med = np.median(cyc, axis=1)                      # (80,)
    cu_med = (cu_med - cu_med.mean()) / (cu_med.std() + 1e-9)
    # Aggregate per-CU into 8-bucket summary (preserve some spatial info)
    cu_buckets = cu_med.reshape(8, 10).mean(axis=1)      # (8,)
    rtn_buckets = rtn.reshape(8, 10).mean(axis=1)        # (8,)
    eigs = np.sort(np.linalg.eigvalsh(0.5 * (sp + sp.T)))[::-1][:8]  # top-8
    eigs = eigs / (np.abs(eigs).max() + 1e-9)
    feat = np.concatenate([cu_buckets, rtn_buckets, eigs]).astype(np.float64)
    return feat


def run_one(seed: int, train_device: str, eval_device: str,
            substrate_aware: bool, T_train=2000, T_test=500) -> dict:
    """Train ESN on a substrate-driven sequence from train_device, optionally
    appending substrate features to the readout. Evaluate on eval_device.

    The reservoir uses the eval device's substrate at inference (the
    "ENVIRONMENT" the model finds itself in); the substrate FEATURE VECTOR is
    the device the model is TOLD it's running on. For a self-referential
    model, those two should match; we test what happens when they don't.
    """
    cfg = ESNConfig(n=128, seed=seed)
    W, Win = build_esn(cfg)
    u, y = narma10(T_train + T_test, seed=seed * 13 + 7)

    train_sub = SubstrateSampler(train_device, seed=seed + 100)
    X_train = run_esn(u, W, Win, cfg, train_sub)

    # Build feature matrix: optionally append substrate features
    if substrate_aware:
        feat_train = substrate_feature_vector(train_device)
        F_train = np.tile(feat_train, (X_train.shape[0], 1))
        Xfull_train = np.concatenate([X_train, F_train], axis=1)
    else:
        Xfull_train = X_train

    wash = 100
    Xtr = Xfull_train[wash:T_train]
    ytr = y[wash:T_train]
    Wout = train_ridge(Xtr, ytr)

    # Eval: run reservoir with eval_device substrate, condition the feature
    # vector on eval_device. (For "wrong substrate" runs the reservoir
    # environment AND the told-identity flip together — most natural variant.)
    eval_sub = SubstrateSampler(eval_device, seed=seed + 999)
    X_eval = run_esn(u, W, Win, cfg, eval_sub)
    if substrate_aware:
        feat_eval = substrate_feature_vector(eval_device)
        F_eval = np.tile(feat_eval, (X_eval.shape[0], 1))
        Xfull_eval = np.concatenate([X_eval, F_eval], axis=1)
    else:
        Xfull_eval = X_eval

    Xte = Xfull_eval[T_train:]
    yte = y[T_train:]
    yhat = predict(Xte, Wout)
    return {
        "seed": seed,
        "train_device": train_device,
        "eval_device": eval_device,
        "substrate_aware": substrate_aware,
        "nrmse": nrmse(yte, yhat),
    }


def main():
    SEEDS = list(range(10))
    rows = []
    for aware in (False, True):
        for train_dev in DEVICES:
            for eval_dev in DEVICES:
                for s in SEEDS:
                    r = run_one(s, train_dev, eval_dev, aware)
                    rows.append(r)
    # Compute degradation gap per (model_type, train_dev)
    summary = {"per_run": rows, "gaps": {}}
    for aware in (False, True):
        for train_dev in DEVICES:
            own = [r["nrmse"] for r in rows
                   if r["substrate_aware"] == aware
                   and r["train_device"] == train_dev
                   and r["eval_device"] == train_dev]
            other = [r["nrmse"] for r in rows
                     if r["substrate_aware"] == aware
                     and r["train_device"] == train_dev
                     and r["eval_device"] != train_dev]
            own = np.array(own); other = np.array(other)
            gap = other.mean() - own.mean()
            # paired difference per seed for std
            per_seed_diff = other - own
            summary["gaps"][f"aware={aware}|train={train_dev}"] = {
                "own_mean": float(own.mean()), "own_std": float(own.std()),
                "other_mean": float(other.mean()), "other_std": float(other.std()),
                "gap": float(gap),
                "per_seed_diff_mean": float(per_seed_diff.mean()),
                "per_seed_diff_std": float(per_seed_diff.std()),
                "n": len(own),
            }
    # Overall comparison: substrate-aware gap vs baseline gap (pooled over devices)
    aware_diffs = []
    base_diffs = []
    for train_dev in DEVICES:
        for s in SEEDS:
            own_a = [r["nrmse"] for r in rows if r["substrate_aware"] and r["train_device"]==train_dev and r["eval_device"]==train_dev and r["seed"]==s][0]
            other_a = [r["nrmse"] for r in rows if r["substrate_aware"] and r["train_device"]==train_dev and r["eval_device"]!=train_dev and r["seed"]==s][0]
            own_b = [r["nrmse"] for r in rows if not r["substrate_aware"] and r["train_device"]==train_dev and r["eval_device"]==train_dev and r["seed"]==s][0]
            other_b = [r["nrmse"] for r in rows if not r["substrate_aware"] and r["train_device"]==train_dev and r["eval_device"]!=train_dev and r["seed"]==s][0]
            aware_diffs.append(other_a - own_a)
            base_diffs.append(other_b - own_b)
    aware_diffs = np.array(aware_diffs); base_diffs = np.array(base_diffs)
    # z-score: how many σ above the BASELINE gap is the AWARE gap
    pooled_std = np.sqrt((aware_diffs.std()**2 + base_diffs.std()**2) / 2 + 1e-12)
    z = (aware_diffs.mean() - base_diffs.mean()) / (pooled_std + 1e-12)
    discovery = bool(z > 2.0 and aware_diffs.mean() > base_diffs.mean())
    summary["overall"] = {
        "aware_gap_mean": float(aware_diffs.mean()),
        "aware_gap_std": float(aware_diffs.std()),
        "baseline_gap_mean": float(base_diffs.mean()),
        "baseline_gap_std": float(base_diffs.std()),
        "z_score": float(z),
        "discovery_gate_passed": discovery,
        "gate_definition": "z>2 AND aware_gap>baseline_gap",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
