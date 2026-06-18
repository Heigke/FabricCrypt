"""z382 — Test A: Static high-T sweep on z372 cell-wide fit + fold metric.

Re-runs the 3-branch (VG1 ∈ {0.2, 0.4, 0.6}) R-46 best fit at
T ∈ {25, 50, 75, 100, 125, 150} °C. For each (T, VG1), measure:
  - cell-wide RMSE in log10(Id) dec
  - "fold" jump magnitude: max single-step drop in log10(Id) within the
    snapback region (Vd >= 1.4 V) along the measured Vd grid.

Hypothesis 1: thermal physics is the missing piece. If elevated T
produces a > 0.5 dec fold at VG1=0.6 in the MODEL prediction, this
confirms that an isothermal pyport at 25 °C systematically misses the
fold seen in slow-DC silicon measurements (which self-heat).

Output: results/z382_thermal_static/{summary.json, plot.png, run.log}
"""
from __future__ import annotations
import json
import os
import sys
import importlib.util
import time
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
OUT = REPO / "results" / "z382_thermal_static"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "run.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# Fresh log
LOG_PATH.write_text("")

# Import helpers from z372 (build_base, load_measured, load_sebas_params,
# find_or_impute_row, make_overrides, patch_sd_scaled)
sp = importlib.util.spec_from_file_location("z372", REPO / "scripts/z372_snapback_demo.py")
z372 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(z372)

from nsram.bsim4_port.nsram_cell_2T import forward_2t  # noqa: E402


T_LIST = [25.0, 50.0, 75.0, 100.0, 125.0, 150.0]
TARGETS = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]
SNAPBACK_VD_MIN = 1.4  # V

# R-46 best per-VG1 params (Bf, iii_body_gain, log10(Rs))
x_best = [1889.88, 1.8447, 9.1722,
          1092.27, 1.5152, 9.8983,
           417.63, 0.9036, 6.7846]
PER_VG1 = {
    0.2: (x_best[0], x_best[1], 10 ** x_best[2]),
    0.4: (x_best[3], x_best[4], 10 ** x_best[5]),
    0.6: (x_best[6], x_best[7], 10 ** x_best[8]),
}


def fold_magnitude(Vd: np.ndarray, Id: np.ndarray, vd_min: float = SNAPBACK_VD_MIN) -> dict:
    """Maximum single-step DROP in log10(|Id|) for Vd >= vd_min along
    increasing Vd. Positive value = fold/snapback present (Id decreased
    as Vd increased).

    Returns dict with: fold_dec, vd_at_fold, n_points_in_region.
    """
    Id = np.abs(np.asarray(Id))
    Vd = np.asarray(Vd)
    order = np.argsort(Vd)
    Vd = Vd[order]
    Id = Id[order]
    mask = (Vd >= vd_min) & (Id > 1e-18) & np.isfinite(Id)
    if mask.sum() < 2:
        return {"fold_dec": float("nan"), "vd_at_fold": float("nan"), "n_points": int(mask.sum())}
    Vd_r = Vd[mask]
    Id_r = Id[mask]
    log_I = np.log10(Id_r)
    # drop[i] = log_I[i] - log_I[i+1]  (positive = decrease as Vd grows)
    drops = log_I[:-1] - log_I[1:]
    if drops.size == 0:
        return {"fold_dec": 0.0, "vd_at_fold": float("nan"), "n_points": int(mask.sum())}
    idx = int(np.argmax(drops))
    return {
        "fold_dec": float(drops[idx]),
        "vd_at_fold": float(Vd_r[idx]),
        "n_points": int(mask.sum()),
    }


def run_one_T(T_C: float) -> dict:
    cfg, M1, M2, bjt = z372.build_base()
    cfg.T_C = T_C
    # Force fresh size-dep at the new T
    cfg.invalidate()
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    sebas_rows = z372.load_sebas_params()

    branch_data = {}
    rmses = []
    for vg1, vg2 in TARGETS:
        Vd_m, Id_m, _fn = z372.load_measured(vg1, vg2)
        Bf, iii, Rs = PER_VG1[vg1]
        bjt.Bf = Bf
        cfg.iii_body_gain = iii
        cfg.vnwell_Rs = Rs
        row = z372.find_or_impute_row(sebas_rows, vg1, vg2)
        P_M1, P_M2 = z372.make_overrides(row)
        Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
        try:
            with z372.patch_sd_scaled(sd_M1, P_M1), z372.patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(
                    cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                    VG1=torch.tensor(vg1, dtype=torch.float64),
                    VG2=torch.tensor(vg2, dtype=torch.float64),
                    warm_start=True,
                )
            Id_p = np.abs(out["Id"].detach().cpu().numpy())
        except Exception as e:
            log(f"  EXC at T={T_C} VG1={vg1}: {e}")
            Id_p = np.full_like(Id_m, np.nan)

        mask = (Id_m > 1e-15) & (Id_p > 1e-15) & np.isfinite(Id_p)
        if mask.sum() < 3:
            rmse = float("nan")
        else:
            rmse = float(np.sqrt(np.mean(
                (np.log10(Id_p[mask]) - np.log10(Id_m[mask])) ** 2)))

        fold_m = fold_magnitude(Vd_m, Id_m)
        fold_p = fold_magnitude(Vd_m, Id_p)
        branch_data[vg1] = {
            "rmse_dec": rmse,
            "fold_meas_dec": fold_m["fold_dec"],
            "fold_meas_vd": fold_m["vd_at_fold"],
            "fold_pred_dec": fold_p["fold_dec"],
            "fold_pred_vd": fold_p["vd_at_fold"],
            "Vd": Vd_m.tolist(),
            "Id_meas": Id_m.tolist(),
            "Id_pred": [float(x) for x in Id_p.tolist()],
        }
        if np.isfinite(rmse):
            rmses.append(rmse)
        log(f"  VG1={vg1}: RMSE={rmse:.3f} dec  fold_pred={fold_p['fold_dec']:.3f} dec "
            f"@ Vd={fold_p['vd_at_fold']:.3f}  (meas fold={fold_m['fold_dec']:.3f})")

    rmse_med = float(np.median(rmses)) if rmses else float("nan")
    return {"T_C": T_C, "rmse_median_dec": rmse_med, "branches": branch_data}


def main():
    t0 = time.time()
    log(f"z382 thermal static sweep — T = {T_LIST} °C")
    log("R-46 best params: " + ", ".join(f"{v:.3f}" for v in x_best))

    results = []
    for T in T_LIST:
        log(f"--- T = {T:.1f} °C ---")
        r = run_one_T(T)
        results.append(r)
        log(f"  median RMSE = {r['rmse_median_dec']:.3f} dec")

    # Aggregate: fold magnitudes vs T for each VG1
    fold_by_vg1 = {vg1: [] for vg1, _ in TARGETS}
    rmse_by_vg1 = {vg1: [] for vg1, _ in TARGETS}
    for r in results:
        for vg1 in fold_by_vg1:
            fold_by_vg1[vg1].append(r["branches"][vg1]["fold_pred_dec"])
            rmse_by_vg1[vg1].append(r["branches"][vg1]["rmse_dec"])

    max_fold_vg1_06 = float(np.nanmax(fold_by_vg1[0.6]))
    any_fold_gt_05 = any(
        np.nanmax(fold_by_vg1[vg1]) > 0.5 for vg1 in fold_by_vg1
    )
    fold_at_25 = fold_by_vg1[0.6][0]
    fold_max = max_fold_vg1_06

    # Gates
    gates = {
        "INFRA_no_nan_or_exc": not any(
            not np.isfinite(r["rmse_median_dec"]) for r in results
        ),
        "DISCOVERY_fold_gt_0p5_at_VG1_0p6": bool(max_fold_vg1_06 > 0.5),
        "DISCOVERY_fold_gt_0p5_any_branch": bool(any_fold_gt_05),
        "AMBITIOUS_rmse_lt_0p5_and_fold_gt_1p5": bool(
            (min(r["rmse_median_dec"] for r in results
                 if np.isfinite(r["rmse_median_dec"])) < 0.5)
            and max_fold_vg1_06 > 1.5
        ),
        "KILL_SHOT_no_fold_at_any_T": bool(
            all(f < 0.5 or not np.isfinite(f) for f in fold_by_vg1[0.6])
        ),
        "fold_VG1_0p6_at_T25": fold_at_25,
        "fold_VG1_0p6_max": fold_max,
        "fold_VG1_0p6_argmax_T_C": float(T_LIST[int(np.nanargmax(fold_by_vg1[0.6]))]),
    }

    # Plots: 2x3 grid (3 VG1 × 2 rows: fold-vs-T and example curves)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, (vg1, _) in enumerate(TARGETS):
        ax = axes[0, col]
        meas_fold = results[0]["branches"][vg1]["fold_meas_dec"]
        ax.plot(T_LIST, fold_by_vg1[vg1], "o-", lw=2, label="pyport model")
        ax.axhline(meas_fold, ls="--", color="k",
                   label=f"measured fold = {meas_fold:.2f} dec")
        ax.axhline(0.5, ls=":", color="r", label="0.5 dec gate")
        ax.set_xlabel("T (°C)")
        ax.set_ylabel("fold magnitude (dec)")
        ax.set_title(f"VG1={vg1}: max log10(Id) drop @ Vd≥{SNAPBACK_VD_MIN}V")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)

        ax2 = axes[1, col]
        # Curves at T=25 and T_max
        b25 = results[0]["branches"][vg1]
        bMx = results[-1]["branches"][vg1]
        ax2.semilogy(b25["Vd"], np.abs(b25["Id_meas"]), "k.", ms=4, label="measured")
        ax2.semilogy(b25["Vd"], np.maximum(np.abs(b25["Id_pred"]), 1e-18),
                     "b-", lw=1.5, label=f"model T={T_LIST[0]}°C")
        ax2.semilogy(bMx["Vd"], np.maximum(np.abs(bMx["Id_pred"]), 1e-18),
                     "r-", lw=1.5, label=f"model T={T_LIST[-1]}°C")
        ax2.set_xlabel("Vd (V)")
        ax2.set_ylabel("|Id| (A)")
        ax2.set_title(f"VG1={vg1} curves")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.4, which="both")
    fig.savefig(OUT / "plot.png", dpi=130)
    plt.close(fig)

    # Strip the bulky Vd/Id arrays from summary to keep it readable.
    results_trim = []
    for r in results:
        rt = {"T_C": r["T_C"], "rmse_median_dec": r["rmse_median_dec"], "branches": {}}
        for vg1, b in r["branches"].items():
            rt["branches"][str(vg1)] = {
                "rmse_dec": b["rmse_dec"],
                "fold_meas_dec": b["fold_meas_dec"],
                "fold_meas_vd": b["fold_meas_vd"],
                "fold_pred_dec": b["fold_pred_dec"],
                "fold_pred_vd": b["fold_pred_vd"],
            }
        results_trim.append(rt)

    summary = {
        "test": "z382_thermal_static",
        "T_C_list": T_LIST,
        "targets": TARGETS,
        "fold_by_vg1_per_T": {str(k): v for k, v in fold_by_vg1.items()},
        "rmse_by_vg1_per_T": {str(k): v for k, v in rmse_by_vg1.items()},
        "results": results_trim,
        "gates": gates,
        "wall_time_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE. wall={summary['wall_time_s']:.1f}s")
    log(f"GATES: {json.dumps(gates, indent=2)}")


if __name__ == "__main__":
    main()
