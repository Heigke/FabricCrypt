"""TM3 — DS-N10 sine classification at T=85°C.

The IiiNetLUT was solved at T=27°C and has no T parameter. We approximate
the BSIM4 thermal effect (verified in TM1) by patching the LUT call:
  - shift VG1 by ΔVG ≈ +|ΔVth(T)|  (kt1=-0.11 → +22 mV at 85°C)
  - scale Inet by mobility factor (T/Tnom)^ute = (358.15/300.15)^(-1.5) ≈ 0.793

Compares BASELINE NSRAM reservoir at T=27°C vs T=85°C across SEEDS={0..4}.

Verify: acc drop < 5 pp → robust ; > 10 pp → fragile.

Outputs:
  results/TM3_dsn10_hot/summary.json
  results/TM3_dsn10_hot/acc_vs_T.png
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "TM3_dsn10_hot"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from DS_N10_reservoir import NSRAMReservoir, sine_dataset, ridge_train, ridge_predict  # noqa: E402
from S2b_transient import IiiNetLUT  # noqa: E402


class ThermalLUTAdapter:
    """Wrap IiiNetLUT to apply approximate BSIM4 thermal corrections."""
    def __init__(self, lut: IiiNetLUT, T_C: float, kt1: float = -0.11,
                 ute: float = -1.5, tnom_C: float = 27.0):
        self.lut = lut
        self.T_C = T_C
        Tnom = tnom_C + 273.15
        T = T_C + 273.15
        TRatio = T / Tnom
        self.dVth = kt1 * (TRatio - 1.0)             # negative for T>Tnom
        self.dVG_shift = -self.dVth                   # higher G needed to reach same Vth
        self.mu_factor = TRatio ** ute                 # <1 for T>Tnom
        # forward attrs
        for a in ("vg1_lo", "vg1_hi", "vg2_lo", "vg2_hi", "vd_lo", "vd_hi",
                  "vb_lo", "vb_hi", "vg1_axis", "vg2_axis", "vd_axis",
                  "vb_axis", "Inet", "Id"):
            setattr(self, a, getattr(lut, a))

    def __call__(self, VG1, VG2, Vd, Vb):
        # Both gates see the same Vth shift → shift both up by dVG
        VG1_eff = np.asarray(VG1, dtype=np.float64) - self.dVG_shift
        VG2_eff = np.asarray(VG2, dtype=np.float64) - self.dVG_shift
        # Inet scales with mobility (subthreshold ~ mu·exp(...))
        return self.mu_factor * self.lut(VG1_eff, VG2_eff, Vd, Vb)


def make_reservoir_at_T(seed, T_C, N=2000, n_readout=256):
    r = NSRAMReservoir(N=N, n_readout=n_readout, seed=seed)
    if abs(T_C - 27.0) > 1e-6:
        r.lut = ThermalLUTAdapter(r.lut, T_C=T_C)
    return r


def run_sine(reservoir, seed):
    X, y = sine_dataset(n_classes=4, n_per_class=30, snippet_len=150, seed=seed)
    n = len(y); n_train = int(0.7 * n)
    feats = np.zeros((n, reservoir.n_readout), dtype=np.float64)
    for i in range(n):
        reservoir.reset()
        s = reservoir.run(X[i])
        feats[i] = s.mean(axis=0)
    n_classes = int(y.max() + 1)
    Y_oh = np.zeros((n, n_classes)); Y_oh[np.arange(n), y] = 1.0
    W = ridge_train(feats[:n_train], Y_oh[:n_train], alpha=1e-2)
    Yhat = ridge_predict(feats[n_train:], W)
    return float((Yhat.argmax(axis=1) == y[n_train:]).mean())


SEEDS = [0, 1, 2, 3, 4]
T_LIST = [27.0, 85.0]
results = {T: [] for T in T_LIST}
t0 = time.time()
for seed in SEEDS:
    for T in T_LIST:
        r = make_reservoir_at_T(seed, T)
        acc = run_sine(r, seed)
        results[T].append(acc)
        print(f"seed={seed} T={T:.0f}°C acc={acc:.3f}", flush=True)

elapsed = time.time() - t0

acc_27 = np.array(results[27.0])
acc_85 = np.array(results[85.0])
drop_pp = (acc_27.mean() - acc_85.mean()) * 100.0

# Plot
fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
ax.boxplot([acc_27, acc_85], labels=["27°C", "85°C"])
ax.axhline(0.978, color="r", ls="--", label="DS-N10 published 97.8%")
ax.set_ylabel("accuracy")
ax.set_title(f"TM3 — Sine 4-class at T={T_LIST[0]} vs {T_LIST[1]} °C  Δ={drop_pp:.1f}pp")
ax.legend(fontsize=8)
fig.savefig(OUT / "acc_vs_T.png", dpi=130)
plt.close(fig)

summary = {
    "N_cells": 2000, "n_readout": 256, "n_seeds": len(SEEDS),
    "wall_s": elapsed,
    "T_27C_acc": {"mean": float(acc_27.mean()), "std": float(acc_27.std()),
                   "seeds": acc_27.tolist()},
    "T_85C_acc": {"mean": float(acc_85.mean()), "std": float(acc_85.std()),
                   "seeds": acc_85.tolist()},
    "drop_pp": float(drop_pp),
    "gate": {
        "robust_lt_5pp": bool(drop_pp < 5.0),
        "fragile_gt_10pp": bool(drop_pp > 10.0),
    },
    "thermal_corrections_applied": {
        "kt1": -0.11, "ute": -1.5, "tnom_C": 27.0,
        "dVth_85C_mV": -22.0, "mu_factor_85C": 0.793,
    },
    "note": "LUT solved at T=27°C; T=85°C approximated by VG-shift + mu rescale.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
