"""ANGLE J — Split-brain co-dependence.

One 128-neuron NARMA-10 reservoir SPLIT across two physical devices:
  - 64 neurons driven by ikaros substrate
  - 64 neurons driven by daedalus substrate
Ridge readout trained jointly on the COMBINED 128-d state vector.

Conditions (per seed):
  intact            — both halves use their native substrate
  ikaros_killed     — ikaros half replaced by zeros
  daedalus_killed   — daedalus half replaced by zeros
  swapped           — ikaros half uses daedalus sub, daedalus half uses ikaros sub
  fungible_baseline — both halves use the SAME (ikaros) substrate; "kill" test on right half
                      Acts as the control: if halves were truly fungible the
                      killed/intact gap should match.

Gate J-DISCOVERY:
  severed-NRMSE > intact-NRMSE by >2σ AND
  swap-NRMSE > swap_to_zero-NRMSE (so swap isn't simply information loss).
"""
from __future__ import annotations
from pathlib import Path
import json
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "identity_benchmark" / "phase2"))

from narma10_reservoir import (
    narma10, build_esn, train_ridge, predict, nrmse, ESNConfig,
)
from _substrate_hooks import SubstrateSampler

DATA = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"
OUT = DATA / "novel" / "J_results.json"


def run_split(u: np.ndarray, W, Win, cfg: ESNConfig,
              sub_left, sub_right) -> np.ndarray:
    """Run a 128-neuron ESN where the first 64 neurons (left half) get
    perturbations from sub_left and the second 64 (right half) from sub_right.
    Either can be None (no perturbation) or a numpy array of zeros applied.
    """
    T = len(u)
    n = cfg.n
    half = n // 2
    x = np.zeros(n)
    X = np.zeros((T, n))
    for t in range(T):
        pre = W @ x + Win[:, 0] * u[t]
        # Left half
        if sub_left == "ZERO":
            pre[:half] = 0.0
        elif sub_left is not None:
            gain = sub_left.rtn_perturbation(half)
            noise = sub_left.spatial_noise(half, scale=cfg.substrate_strength)
            pre[:half] = pre[:half] * gain + noise
        # Right half
        if sub_right == "ZERO":
            pre[half:] = 0.0
        elif sub_right is not None:
            gain = sub_right.rtn_perturbation(half)
            noise = sub_right.spatial_noise(half, scale=cfg.substrate_strength)
            pre[half:] = pre[half:] * gain + noise
        x_new = np.tanh(pre)
        # If a half is zeroed at pre-stage, also zero its state to fully sever
        if sub_left == "ZERO":
            x_new[:half] = 0.0
        if sub_right == "ZERO":
            x_new[half:] = 0.0
        x = (1 - cfg.leak) * x + cfg.leak * x_new
        X[t] = x
    return X


def one_seed(seed: int, T_train=2000, T_test=500) -> dict:
    cfg = ESNConfig(n=128, seed=seed)
    W, Win = build_esn(cfg)
    u, y = narma10(T_train + T_test, seed=seed * 13 + 7)
    wash = 100

    # Build samplers
    sub_ik = SubstrateSampler("ikaros",   seed=seed + 100)
    sub_da = SubstrateSampler("daedalus", seed=seed + 200)
    sub_ik2 = SubstrateSampler("ikaros",  seed=seed + 300)  # second copy for fungible baseline

    conditions = {
        "intact":            (sub_ik, sub_da),
        "ikaros_killed":     ("ZERO", sub_da),
        "daedalus_killed":   (sub_ik, "ZERO"),
        "swapped":           (sub_da, sub_ik),         # left half now sees daedalus sub
        "swap_to_zero":      (sub_ik, "ZERO"),         # alias for daedalus_killed — used only as reference
        "fungible_baseline": (sub_ik, sub_ik2),        # both halves "ikaros-like"
    }

    out = {}
    for name, (L, R) in conditions.items():
        # Train readout on THIS condition's state and evaluate on holdout
        # (We train per-condition; transplant tests are name=intact vs swapped
        # which share the architecture but differ in substrate routing.)
        X = run_split(u, W, Win, cfg, L, R)
        Wout = train_ridge(X[wash:T_train], y[wash:T_train])
        yhat = predict(X[T_train:], Wout)
        out[name] = float(nrmse(y[T_train:], yhat))

    # Also: train on INTACT, then evaluate on swapped/killed without retraining
    # (this is the actual "transplant" test, more honest)
    X_intact = run_split(u, W, Win, cfg, sub_ik, sub_da)
    Wout_intact = train_ridge(X_intact[wash:T_train], y[wash:T_train])
    transplant = {}
    for name, (L, R) in conditions.items():
        X_e = run_split(u, W, Win, cfg, L, R)
        yhat = predict(X_e[T_train:], Wout_intact)
        transplant[name] = float(nrmse(y[T_train:], yhat))

    return {"seed": seed, "per_condition_retrain": out, "transplant": transplant}


def main():
    SEEDS = list(range(10))
    rows = [one_seed(s) for s in SEEDS]

    def collect(field, cond):
        return np.array([r[field][cond] for r in rows])

    summary = {"per_seed": rows, "stats": {}}
    for field in ("per_condition_retrain", "transplant"):
        s = {}
        for cond in ("intact", "ikaros_killed", "daedalus_killed",
                     "swapped", "swap_to_zero", "fungible_baseline"):
            vals = collect(field, cond)
            s[cond] = {"mean": float(vals.mean()), "std": float(vals.std()),
                       "n": len(vals)}
        summary["stats"][field] = s

    # Gate: severed > intact by >2σ on TRANSPLANT condition
    t = summary["stats"]["transplant"]
    intact = collect("transplant", "intact")
    severed = collect("transplant", "ikaros_killed")  # use one severed condition
    diff = severed - intact
    z_sever = diff.mean() / (diff.std() + 1e-12) * np.sqrt(len(diff))
    swap = collect("transplant", "swapped")
    swap_to_zero = collect("transplant", "swap_to_zero")
    swap_diff = swap - swap_to_zero
    # Discovery requires substrate confusion (swap) to be WORSE than mere
    # information loss (swap_to_zero), AND severance to be >2σ vs intact.
    discovery = bool(z_sever > 2.0 and swap.mean() > swap_to_zero.mean())
    # Also compare to fungible baseline
    fungible_intact = collect("transplant", "fungible_baseline")
    summary["overall"] = {
        "intact_nrmse": float(intact.mean()),
        "severed_nrmse": float(severed.mean()),
        "severance_z": float(z_sever),
        "swapped_nrmse": float(swap.mean()),
        "swap_to_zero_nrmse": float(swap_to_zero.mean()),
        "confusion_minus_info_loss": float(swap.mean() - swap_to_zero.mean()),
        "fungible_intact_nrmse": float(fungible_intact.mean()),
        "discovery_gate_passed": discovery,
        "gate_definition": "severance_z>2 AND swap>swap_to_zero",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
