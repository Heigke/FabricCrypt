"""TM2 — Temperature sweep on z372 cell-wide RMSE fit.

Re-runs the z372 snapback demo at T ∈ {25, 50, 75, 85, 100} °C using
identical R-46 best params (no re-fitting). Measures cell-wide RMSE
(median across the 3 VG1 branches) at each T.

Output:
  results/TM2_t_sweep/rmse_vs_T.png
  results/TM2_t_sweep/summary.json
"""
from __future__ import annotations
import json
import sys
import os
import importlib.util
import re
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nsram"))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results" / "TM2_t_sweep"
OUT.mkdir(parents=True, exist_ok=True)
DATA = REPO / "data/sebas_2026_04_22"

# Import helpers from z372
sp = importlib.util.spec_from_file_location("z372", REPO / "scripts/z372_snapback_demo.py")
z372 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z372)

from nsram.bsim4_port.nsram_cell_2T import forward_2t  # noqa: E402

T_LIST = [25.0, 50.0, 75.0, 85.0, 100.0]
TARGETS = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]
x_best = [1889.88, 1.8447, 9.1722,
          1092.27, 1.5152, 9.8983,
           417.63, 0.9036, 6.7846]
per_vg1 = {0.2: (x_best[0], x_best[1], 10**x_best[2]),
           0.4: (x_best[3], x_best[4], 10**x_best[5]),
           0.6: (x_best[6], x_best[7], 10**x_best[8])}


def run_at_T(T_C):
    cfg, M1, M2, bjt = z372.build_base()
    cfg.T_C = T_C
    # Recompute size-dep with the new T
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    sebas_rows = z372.load_sebas_params()
    rmses = []
    per_branch = {}
    nan_branches = 0
    for vg1, vg2 in TARGETS:
        Vd_m, Id_m, _ = z372.load_measured(vg1, vg2)
        Bf, iii, Rs = per_vg1[vg1]
        bjt.Bf = Bf; cfg.iii_body_gain = iii; cfg.vnwell_Rs = Rs
        row = z372.find_or_impute_row(sebas_rows, vg1, vg2)
        P_M1, P_M2 = z372.make_overrides(row)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
        with z372.patch_sd_scaled(sd_M1, P_M1), z372.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(vg1, dtype=torch.float64),
                             VG2=torch.tensor(vg2, dtype=torch.float64),
                             warm_start=True)
        Id_p = np.abs(out["Id"].detach().cpu().numpy())
        mask = (Id_m > 1e-15) & (Id_p > 1e-15) & np.isfinite(Id_p)
        if mask.sum() < 3:
            rmse = float("nan"); nan_branches += 1
        else:
            rmse = float(np.sqrt(np.mean(
                (np.log10(Id_p[mask]) - np.log10(Id_m[mask]))**2)))
        per_branch[vg1] = rmse
        if np.isfinite(rmse):
            rmses.append(rmse)
    median = float(np.median(rmses)) if rmses else float("nan")
    return median, per_branch, nan_branches


results = []
for T in T_LIST:
    med, branches, nan_b = run_at_T(T)
    print(f"T={T:5.1f}°C  median_RMSE={med:.3f} dec  branches={branches}  nan={nan_b}",
          flush=True)
    results.append({"T_C": T, "rmse_median_dec": med, "per_branch": branches,
                    "n_nan_branches": nan_b})

rmse_arr = np.array([r["rmse_median_dec"] for r in results])
ok_arr = np.isfinite(rmse_arr)
monotonic = bool(np.all(np.diff(rmse_arr[ok_arr]) >= -0.05))  # non-decreasing within 0.05 noise
delta_per_10C = []
for i in range(1, len(T_LIST)):
    if np.isfinite(rmse_arr[i]) and np.isfinite(rmse_arr[i-1]):
        d_T = T_LIST[i] - T_LIST[i-1]
        if d_T > 0:
            delta_per_10C.append((rmse_arr[i] - rmse_arr[i-1]) / d_T * 10.0)

# Plot
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
ax.plot(T_LIST, rmse_arr, "o-", lw=2)
ax.set_xlabel("T (°C)"); ax.set_ylabel("median cell-wide RMSE (dec)")
ax.set_title("TM2 — Cell-wide fit RMSE vs Temperature")
ax.grid(alpha=0.4)
fig.savefig(OUT / "rmse_vs_T.png", dpi=130)
plt.close(fig)

summary = {
    "T_C_list": T_LIST,
    "rmse_median_dec_per_T": [float(x) for x in rmse_arr],
    "per_branch_per_T": results,
    "gates": {
        "no_nan": int(np.sum(~ok_arr)) == 0,
        "monotonic_increase": monotonic,
        "delta_rmse_per_10C_dec": [float(x) for x in delta_per_10C],
        "delta_rmse_per_10C_mean": float(np.mean(delta_per_10C)) if delta_per_10C else float("nan"),
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
