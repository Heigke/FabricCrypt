"""z383 — Test B: Self-heating coupled DC sweep.

For each Vd point we solve for the self-consistent junction temperature:

    T_j = T_amb + |Vd · Id| · R_th

where R_th is a lumped thermal resistance (K/W). We use Picard iteration
on T_j, recomputing the BSIM4 size-dep tables at each step (because
mobility, Vth, Js all shift with T).

We do NOT alter the pyport solver itself; instead, we wrap
solve_2t_steady_state in a per-Vd loop that varies cfg.T_C between calls.
Warm-starts for (Vsint, Vb) cascade across the Vd sweep, mirroring
forward_2t's behaviour.

Hypothesis 5: real silicon snapback in slow-DC sweeps is thermal
runaway, not isothermal DC bistability. If any R_th ∈ {10, 50, 100, 500}
K/W produces a fold > 0.5 dec at VG1=0.6, this is supporting evidence.

Output: results/z383_self_heating/{summary.json, plot.png, run.log}
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
OUT = REPO / "results" / "z383_self_heating"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "run.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


LOG_PATH.write_text("")

sp = importlib.util.spec_from_file_location("z372", REPO / "scripts/z372_snapback_demo.py")
z372 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(z372)

from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state  # noqa: E402


TARGETS = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]
SNAPBACK_VD_MIN = 1.4
R_TH_LIST = [0.0, 10.0, 50.0, 100.0, 500.0]  # K/W; 0 = isothermal baseline
T_AMBIENT_C = 25.0
SELF_HEAT_MAX_ITERS = 6
SELF_HEAT_TOL_K = 0.5  # K — tighter than the Vd grid would justify

x_best = [1889.88, 1.8447, 9.1722,
          1092.27, 1.5152, 9.8983,
           417.63, 0.9036, 6.7846]
PER_VG1 = {
    0.2: (x_best[0], x_best[1], 10 ** x_best[2]),
    0.4: (x_best[3], x_best[4], 10 ** x_best[5]),
    0.6: (x_best[6], x_best[7], 10 ** x_best[8]),
}


def fold_magnitude(Vd: np.ndarray, Id: np.ndarray, vd_min: float = SNAPBACK_VD_MIN) -> dict:
    Id = np.abs(np.asarray(Id))
    Vd = np.asarray(Vd)
    order = np.argsort(Vd)
    Vd = Vd[order]; Id = Id[order]
    mask = (Vd >= vd_min) & (Id > 1e-18) & np.isfinite(Id)
    if mask.sum() < 2:
        return {"fold_dec": float("nan"), "vd_at_fold": float("nan"), "n_points": int(mask.sum())}
    log_I = np.log10(Id[mask]); Vd_r = Vd[mask]
    drops = log_I[:-1] - log_I[1:]
    if drops.size == 0:
        return {"fold_dec": 0.0, "vd_at_fold": float("nan"), "n_points": int(mask.sum())}
    idx = int(np.argmax(drops))
    return {"fold_dec": float(drops[idx]), "vd_at_fold": float(Vd_r[idx]),
            "n_points": int(mask.sum())}


def run_one_vg1(cfg, M1, M2, bjt, vg1: float, vg2: float,
                P_M1, P_M2, R_th: float, Vd_m: np.ndarray) -> dict:
    """Self-heating sweep at one (VG1, VG2) branch, one R_th."""
    Vsint_w = None
    Vb_w = None
    Id_out = np.full_like(Vd_m, np.nan, dtype=np.float64)
    Tj_out = np.full_like(Vd_m, np.nan, dtype=np.float64)
    n_iters_total = 0
    n_conv = 0

    for i, Vd_val in enumerate(Vd_m):
        Vd_i = torch.tensor([float(Vd_val)], dtype=torch.float64)
        VG1_t = torch.tensor(vg1, dtype=torch.float64)
        VG2_t = torch.tensor(vg2, dtype=torch.float64)
        # Self-heating Picard iteration on T_j
        T_j = T_AMBIENT_C
        prev_T = T_j
        Id_val = 0.0
        out = None
        for it in range(SELF_HEAT_MAX_ITERS):
            n_iters_total += 1
            cfg.T_C = T_j
            cfg.invalidate()
            # warm-start: Vsint=Vd/2, Vb=0.5 for first point, else cascade
            if Vsint_w is None:
                Vsi = (0.5 * Vd_i).clone()
                Vbi = torch.tensor([0.5], dtype=torch.float64)
            else:
                Vsi = Vsint_w.clone()
                Vbi = Vb_w.clone()
            try:
                with z372.patch_sd_scaled(cfg.size_dep_M1(M1), P_M1), \
                     z372.patch_sd_scaled(cfg.size_dep_M2(M2), P_M2):
                    out = solve_2t_steady_state(
                        cfg, model=M1, bjt=bjt,
                        Vd=Vd_i, VG1=VG1_t, VG2=VG2_t,
                        P_M1=None, P_M2=None,
                        Vsint_init=Vsi, Vb_init=Vbi,
                        model_M2=M2,
                    )
            except Exception as e:
                log(f"    EXC at Vd={Vd_val:.3f} it={it} T={T_j:.1f}: {e}")
                out = None
                break
            Id_val = float(torch.abs(out["Id"]).item())
            P_diss = abs(float(Vd_val)) * Id_val
            T_new = T_AMBIENT_C + R_th * P_diss
            # damp slightly for stability under sharp T-sensitivity
            T_j = 0.5 * prev_T + 0.5 * T_new
            if abs(T_j - prev_T) < SELF_HEAT_TOL_K:
                prev_T = T_j
                break
            prev_T = T_j

        if out is not None:
            Id_out[i] = Id_val
            Tj_out[i] = T_j
            if bool(out["converged"].all()):
                n_conv += 1
            # Cascade warm-start
            Vsint_w = out["Vsint"].detach()
            Vb_w = out["Vb"].detach()

    return {"Id": Id_out, "Tj": Tj_out,
            "n_iters_total": n_iters_total, "n_converged": n_conv,
            "n_points": int(len(Vd_m))}


def run_one_R_th(R_th: float) -> dict:
    log(f"=== R_th = {R_th} K/W ===")
    cfg, M1, M2, bjt = z372.build_base()
    sebas_rows = z372.load_sebas_params()
    branch_data = {}
    for vg1, vg2 in TARGETS:
        Vd_m, Id_m, _fn = z372.load_measured(vg1, vg2)
        Bf, iii, Rs = PER_VG1[vg1]
        bjt.Bf = Bf
        cfg.iii_body_gain = iii
        cfg.vnwell_Rs = Rs
        row = z372.find_or_impute_row(sebas_rows, vg1, vg2)
        P_M1, P_M2 = z372.make_overrides(row)

        sh = run_one_vg1(cfg, M1, M2, bjt, vg1, vg2, P_M1, P_M2, R_th, Vd_m)
        Id_p = sh["Id"]
        Tj = sh["Tj"]

        mask = (Id_m > 1e-15) & (Id_p > 1e-15) & np.isfinite(Id_p)
        if mask.sum() < 3:
            rmse = float("nan")
        else:
            rmse = float(np.sqrt(np.mean(
                (np.log10(Id_p[mask]) - np.log10(Id_m[mask])) ** 2)))

        fold_p = fold_magnitude(Vd_m, Id_p)
        fold_m = fold_magnitude(Vd_m, Id_m)
        branch_data[vg1] = {
            "rmse_dec": rmse,
            "fold_meas_dec": fold_m["fold_dec"],
            "fold_pred_dec": fold_p["fold_dec"],
            "fold_pred_vd": fold_p["vd_at_fold"],
            "Tj_max_C": float(np.nanmax(Tj)) if np.any(np.isfinite(Tj)) else float("nan"),
            "Tj_at_vd_max_C": float(Tj[-1]) if np.isfinite(Tj[-1]) else float("nan"),
            "n_converged": sh["n_converged"],
            "n_points": sh["n_points"],
            "Vd": Vd_m.tolist(),
            "Id_meas": Id_m.tolist(),
            "Id_pred": [float(x) for x in Id_p.tolist()],
            "Tj": [float(x) for x in Tj.tolist()],
        }
        log(f"  VG1={vg1}: RMSE={rmse:.3f} dec  fold_pred={fold_p['fold_dec']:.3f} dec "
            f"@ Vd={fold_p['vd_at_fold']:.3f}  Tj_max={branch_data[vg1]['Tj_max_C']:.1f}°C "
            f"conv={sh['n_converged']}/{sh['n_points']}")
    rmse_med = float(np.median([
        b["rmse_dec"] for b in branch_data.values() if np.isfinite(b["rmse_dec"])
    ])) if branch_data else float("nan")
    return {"R_th": R_th, "rmse_median_dec": rmse_med, "branches": branch_data}


def main():
    t0 = time.time()
    log(f"z383 self-heating sweep — R_th = {R_TH_LIST} K/W, T_amb = {T_AMBIENT_C} °C")

    results = []
    for R_th in R_TH_LIST:
        r = run_one_R_th(R_th)
        results.append(r)
        log(f"  R_th={R_th}: median RMSE = {r['rmse_median_dec']:.3f} dec")

    # Aggregate
    fold_by_vg1 = {vg1: [] for vg1, _ in TARGETS}
    rmse_by_vg1 = {vg1: [] for vg1, _ in TARGETS}
    Tj_max_by_vg1 = {vg1: [] for vg1, _ in TARGETS}
    for r in results:
        for vg1 in fold_by_vg1:
            fold_by_vg1[vg1].append(r["branches"][vg1]["fold_pred_dec"])
            rmse_by_vg1[vg1].append(r["branches"][vg1]["rmse_dec"])
            Tj_max_by_vg1[vg1].append(r["branches"][vg1]["Tj_max_C"])

    max_fold_06 = float(np.nanmax(fold_by_vg1[0.6]))
    any_fold_gt_05 = any(
        np.nanmax(fold_by_vg1[vg1]) > 0.5 for vg1 in fold_by_vg1
    )
    fold_at_iso = fold_by_vg1[0.6][0]

    gates = {
        "INFRA_no_nan_or_exc": not any(
            not np.isfinite(r["rmse_median_dec"]) for r in results
        ),
        "DISCOVERY_fold_gt_0p5_at_VG1_0p6": bool(max_fold_06 > 0.5),
        "DISCOVERY_fold_gt_0p5_any_branch": bool(any_fold_gt_05),
        "AMBITIOUS_rmse_lt_0p5_and_fold_gt_1p5": bool(
            min((r["rmse_median_dec"] for r in results
                 if np.isfinite(r["rmse_median_dec"])), default=float("inf")) < 0.5
            and max_fold_06 > 1.5
        ),
        "KILL_SHOT_no_fold_at_any_Rth": bool(
            all(f < 0.5 or not np.isfinite(f) for f in fold_by_vg1[0.6])
        ),
        "fold_VG1_0p6_isothermal": fold_at_iso,
        "fold_VG1_0p6_max": max_fold_06,
        "fold_VG1_0p6_argmax_Rth": float(R_TH_LIST[int(np.nanargmax(fold_by_vg1[0.6]))]),
        "Tj_max_by_Rth_VG1_0p6": Tj_max_by_vg1[0.6],
    }

    # Plot: 2x3 (3 VG1 columns; row1 = fold vs R_th, row2 = curves)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, (vg1, _) in enumerate(TARGETS):
        ax = axes[0, col]
        meas_fold = results[0]["branches"][vg1]["fold_meas_dec"]
        ax.plot(R_TH_LIST, fold_by_vg1[vg1], "o-", lw=2, label="pyport (self-heating)")
        ax.axhline(meas_fold, ls="--", color="k",
                   label=f"meas fold = {meas_fold:.2f} dec")
        ax.axhline(0.5, ls=":", color="r", label="0.5 dec gate")
        ax.set_xlabel("R_th (K/W)")
        ax.set_ylabel("fold magnitude (dec)")
        ax.set_title(f"VG1={vg1}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)

        ax2 = axes[1, col]
        b0 = results[0]["branches"][vg1]
        bMx = results[-1]["branches"][vg1]
        ax2.semilogy(b0["Vd"], np.abs(b0["Id_meas"]), "k.", ms=4, label="measured")
        ax2.semilogy(b0["Vd"], np.maximum(np.abs(b0["Id_pred"]), 1e-18),
                     "b-", lw=1.5, label=f"R_th={R_TH_LIST[0]} (iso)")
        ax2.semilogy(bMx["Vd"], np.maximum(np.abs(bMx["Id_pred"]), 1e-18),
                     "r-", lw=1.5, label=f"R_th={R_TH_LIST[-1]}")
        ax2.set_xlabel("Vd (V)")
        ax2.set_ylabel("|Id| (A)")
        ax2.set_title(f"VG1={vg1} curves")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.4, which="both")
    fig.savefig(OUT / "plot.png", dpi=130)
    plt.close(fig)

    results_trim = []
    for r in results:
        rt = {"R_th": r["R_th"], "rmse_median_dec": r["rmse_median_dec"], "branches": {}}
        for vg1, b in r["branches"].items():
            rt["branches"][str(vg1)] = {
                "rmse_dec": b["rmse_dec"],
                "fold_meas_dec": b["fold_meas_dec"],
                "fold_pred_dec": b["fold_pred_dec"],
                "fold_pred_vd": b["fold_pred_vd"],
                "Tj_max_C": b["Tj_max_C"],
                "Tj_at_vd_max_C": b["Tj_at_vd_max_C"],
                "n_converged": b["n_converged"],
                "n_points": b["n_points"],
            }
        results_trim.append(rt)

    summary = {
        "test": "z383_self_heating",
        "R_th_list": R_TH_LIST,
        "T_ambient_C": T_AMBIENT_C,
        "self_heat_max_iters": SELF_HEAT_MAX_ITERS,
        "fold_by_vg1_per_Rth": {str(k): v for k, v in fold_by_vg1.items()},
        "rmse_by_vg1_per_Rth": {str(k): v for k, v in rmse_by_vg1.items()},
        "Tj_max_by_vg1_per_Rth": {str(k): v for k, v in Tj_max_by_vg1.items()},
        "results": results_trim,
        "gates": gates,
        "wall_time_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE. wall={summary['wall_time_s']:.1f}s")
    log(f"GATES: {json.dumps(gates, indent=2)}")


if __name__ == "__main__":
    main()
