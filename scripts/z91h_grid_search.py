"""z91h — coordinate-descent grid search over residual NSRAM-2T parameters.

Replaces the failing z91h Adam fit (which got confused by silently dropped
non-converged curves and noisy gradients).  We instead sweep four
parameters one at a time around the v9 baseline:

    vnwell_Rs        ∈ {3e8, 1e9, 3e9, 1e10}                — knee position
    Bf  (BJT)        ∈ {1e3, 1e4, 5e4}                       — BJT turn-on
    alpha0_scale     ∈ {0.3, 1.0, 3.0, 10.0} × CSV ALPHA0
    beta0_global     ∈ {10, 15, 19, 25}                      — overrides CSV BETA0

Per point: run z91g forward over all 33 curves (~70 s with arclength
solver, no_grad).  Score by  median + 0.3 × p90  (compound metric).
Two coordinate-descent rounds; if a sweep does not improve, leave the
parameter at the v9 baseline.
"""
from __future__ import annotations
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91h_grid_search"
OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "log.txt"
GRID_PATH = OUT / "grid.json"
BEST_PATH = OUT / "best.json"
PLOT_PATH = OUT / "best_fit_vs_meas.png"

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


# Reuse z91f helpers
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91f)
load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt


# ---------------------------------------------------------------- logging
def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------- one evaluation
def evaluate_point(
    Rs: float,
    Bf: float,
    alpha0_scale: float,
    beta0_global: float,
    *,
    cfg: NSRAMCell2TConfig,
    model_M1: BSIM4Model,
    model_M2: BSIM4Model,
    sd_M1,
    sd_M2,
    curves: list,
    sebas_rows: list,
    return_predictions: bool = False,
) -> dict:
    cfg.vnwell_Rs = float(Rs)
    log_eps = 1e-15
    rmses: list[float] = []
    n_evaluated = 0
    n_skipped = 0
    n_total_pts = 0
    n_conv_pts = 0
    predictions: list[dict] = []

    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            n_skipped += 1
            if return_predictions:
                predictions.append({"VG1": c["VG1"], "VG2": c["VG2"],
                                    "skipped": True, "reason": "NaN row"})
            continue
        P_M1, P_M2 = make_overrides(sebas_row)
        if P_M2:
            for k in ("k1", "k2", "etab", "beta0"):
                P_M2.pop(k, None)
            if not P_M2:
                P_M2 = None
        # Apply alpha0_scale on top of CSV ALPHA0, and override beta0 globally.
        if P_M1 is None:
            P_M1 = {}
        if "alpha0" in P_M1:
            P_M1["alpha0"] = P_M1["alpha0"] * float(alpha0_scale)
        else:
            csv_a0 = sebas_row.get("ALPHA0", float("nan"))
            if not math.isnan(csv_a0):
                P_M1["alpha0"] = torch.tensor(
                    float(csv_a0) * float(alpha0_scale), dtype=torch.float64)
        P_M1["beta0"] = torch.tensor(float(beta0_global), dtype=torch.float64)
        # Mirror beta0 onto M2 too (matches BETA0_TEST behaviour in z91g).
        if P_M2 is None:
            P_M2 = {}
        P_M2["beta0"] = torch.tensor(float(beta0_global), dtype=torch.float64)

        bjt = make_bjt(sebas_row)
        bjt.Bf = float(Bf)
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        cfg.vnwell_mbjt = mbjt

        try:
            with torch.no_grad(), \
                 patch_sd_scaled(sd_M1, P_M1), \
                 patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t_arclength_grad(
                    cfg, model_M1=model_M1, model_M2=model_M2,
                    bjt=bjt, Vd_seq=c["Vd"],
                    VG1=torch.tensor(c["VG1"]),
                    VG2=torch.tensor(c["VG2"]))
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            n_skipped += 1
            if return_predictions:
                predictions.append({"VG1": c["VG1"], "VG2": c["VG2"],
                                    "skipped": True,
                                    "reason": f"forward error: {e}"})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            sq = (log_p - log_m) ** 2
            rmse = float(torch.sqrt(sq[conv].mean()))
        else:
            rmse = float("inf")
        if math.isfinite(rmse):
            rmses.append(rmse)
        n_evaluated += 1
        n_total_pts += int(len(conv))
        n_conv_pts += int(conv.sum())
        if return_predictions:
            predictions.append({
                "VG1": c["VG1"], "VG2": c["VG2"], "skipped": False,
                "log_rmse": rmse,
                "n_converged": int(conv.sum()),
                "n_total": int(len(conv)),
                "Vd": c["Vd"].numpy().tolist(),
                "Id_meas": c["Id"].numpy().tolist(),
                "Id_pred": Id_pred.numpy().tolist(),
                "converged": conv.numpy().tolist(),
            })

    if rmses:
        median = float(np.median(rmses))
        p90 = float(np.percentile(rmses, 90))
    else:
        median = float("inf")
        p90 = float("inf")
    score = median + 0.3 * p90
    return {
        "Rs": Rs, "Bf": Bf,
        "alpha0_scale": alpha0_scale, "beta0_global": beta0_global,
        "median_log_rmse": median, "p90_log_rmse": p90, "score": score,
        "n_evaluated": n_evaluated, "n_skipped": n_skipped,
        "n_conv_pts": n_conv_pts, "n_total_pts": n_total_pts,
        "predictions": predictions if return_predictions else None,
    }


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    _log("z91h grid search starting")

    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    patch_model_values(model_M1, type_n=True)
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2, type_n=True)
    _log("M1/M2 cards loaded and patched")

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn),
                              T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    curves = load_curves()
    sebas_rows = load_sebas_params()
    _log(f"{len(curves)} curves, {len(sebas_rows)} CSV rows")

    # Sweep grids per parameter
    SWEEPS = {
        "Rs":           [3e8, 1e9, 3e9, 1e10],
        "Bf":           [1e3, 1e4, 5e4],
        "alpha0_scale": [0.3, 1.0, 3.0, 10.0],
        "beta0_global": [10.0, 15.0, 19.0, 25.0],
    }
    # v9 baseline
    current = {"Rs": 1e9, "Bf": 1e4, "alpha0_scale": 1.0, "beta0_global": 19.0}

    all_evals: list[dict] = []
    cache: dict[tuple, dict] = {}

    def eval_cached(p: dict) -> dict:
        key = (p["Rs"], p["Bf"], p["alpha0_scale"], p["beta0_global"])
        if key in cache:
            return cache[key]
        ev_t0 = time.time()
        res = evaluate_point(
            p["Rs"], p["Bf"], p["alpha0_scale"], p["beta0_global"],
            cfg=cfg, model_M1=model_M1, model_M2=model_M2,
            sd_M1=sd_M1, sd_M2=sd_M2,
            curves=curves, sebas_rows=sebas_rows)
        res["elapsed_s"] = time.time() - ev_t0
        cache[key] = res
        all_evals.append(res)
        n = len(all_evals)
        _log(f"eval#{n:02d}  Rs={p['Rs']:.2e}  Bf={p['Bf']:.2e}  "
             f"a0×{p['alpha0_scale']:.2g}  b0={p['beta0_global']:.1f}  "
             f"→ med={res['median_log_rmse']:.3f}  p90={res['p90_log_rmse']:.3f}  "
             f"score={res['score']:.3f}  ({res['elapsed_s']:.0f}s, "
             f"{n_evaluated_str(res)})")
        # Progress every 5
        if n % 5 == 0:
            _log(f"  ...progress: {n} evals done, "
                 f"{(time.time()-t0)/60:.1f} min elapsed")
        # Persist running grid
        GRID_PATH.write_text(json.dumps(
            [strip_pred(e) for e in all_evals], indent=2))
        return res

    # Step 1: baseline
    base_res = eval_cached(current)
    best = dict(current)
    best_score = base_res["score"]
    _log(f"BASELINE score={best_score:.3f}  med={base_res['median_log_rmse']:.3f}  "
         f"p90={base_res['p90_log_rmse']:.3f}")

    # Two rounds of coordinate descent
    SWEEP_ORDER = ["Rs", "beta0_global", "alpha0_scale", "Bf"]
    for round_idx in range(2):
        _log(f"=== Round {round_idx+1} ===")
        improved_this_round = False
        for pname in SWEEP_ORDER:
            _log(f"-- sweeping {pname} (current best={best[pname]}) --")
            for v in SWEEPS[pname]:
                trial = dict(best)
                trial[pname] = v
                res = eval_cached(trial)
                if res["score"] < best_score - 1e-6:
                    best_score = res["score"]
                    best = dict(trial)
                    improved_this_round = True
                    _log(f"   ** NEW BEST {pname}={v}: score={best_score:.3f}")
            _log(f"   after {pname}: best={best}  score={best_score:.3f}")
        if not improved_this_round:
            _log(f"Round {round_idx+1} produced no improvement — stopping early")
            break

    _log(f"FINAL best={best}  score={best_score:.3f}")

    # Re-run best with predictions for plotting
    best_res = evaluate_point(
        best["Rs"], best["Bf"], best["alpha0_scale"], best["beta0_global"],
        cfg=cfg, model_M1=model_M1, model_M2=model_M2,
        sd_M1=sd_M1, sd_M2=sd_M2,
        curves=curves, sebas_rows=sebas_rows,
        return_predictions=True)
    _log(f"Best re-eval: median={best_res['median_log_rmse']:.3f}  "
         f"p90={best_res['p90_log_rmse']:.3f}")

    # Persist artefacts
    GRID_PATH.write_text(json.dumps(
        [strip_pred(e) for e in all_evals], indent=2))
    summary = {
        "best": best,
        "best_score": best_score,
        "best_median_log_rmse": best_res["median_log_rmse"],
        "best_p90_log_rmse": best_res["p90_log_rmse"],
        "v9_baseline_median": 1.19,
        "v9_baseline_p90": 2.88,
        "n_evaluations": len(all_evals),
        "elapsed_s": time.time() - t0,
        "metric": "median + 0.3 * p90 over converged curves",
    }
    BEST_PATH.write_text(json.dumps(summary, indent=2))
    _log(f"wrote {BEST_PATH.name}: {summary}")

    # Plot best fit
    plot_predictions(best_res, best, summary)
    _log(f"wrote {PLOT_PATH.name}")
    _log(f"DONE in {(time.time()-t0)/60:.1f} min")


def n_evaluated_str(res: dict) -> str:
    return f"{res['n_evaluated']} curves eval, {res['n_skipped']} skip"


def strip_pred(e: dict) -> dict:
    out = {k: v for k, v in e.items() if k != "predictions"}
    return out


def plot_predictions(best_res: dict, best: dict, summary: dict) -> None:
    preds = best_res["predictions"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        sel = [r for r in preds
               if not r.get("skipped") and abs(r["VG1"] - vg1) < 1e-3]
        sel.sort(key=lambda r: r["VG2"])
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(sel), 1)))
        for color, r in zip(cmap, sel):
            Vd = np.array(r["Vd"]); Im = np.array(r["Id_meas"])
            Ip = np.array(r["Id_pred"]); cm = np.array(r["converged"])
            ax.semilogy(Vd, Im, "o", ms=3, color=color, alpha=0.5)
            Ip_plot = np.where(cm, Ip, np.nan)
            ax.semilogy(Vd, Ip_plot, "-", lw=1.0, color=color)
        ax.set_title(f"VG1 = {vg1} V")
        ax.set_xlabel("Vd [V]")
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(
        f"z91h grid search — best: Rs={best['Rs']:.2e}  Bf={best['Bf']:.2e}  "
        f"α0×{best['alpha0_scale']:.2g}  β0={best['beta0_global']:.1f}\n"
        f"median log-RMSE = {summary['best_median_log_rmse']:.3f}  "
        f"p90 = {summary['best_p90_log_rmse']:.3f}  "
        f"(v9 baseline: 1.19 / 2.88)",
        fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
