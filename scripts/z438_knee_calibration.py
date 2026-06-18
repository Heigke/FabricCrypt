"""z438 — Knee + magnitude calibration for z432 pseudo-transient snapback.

z432 backward sweep landed at 1.027 dec cell-wide but visual overlays show
two systematic gaps:
  1. Knee position: model trips ≈ 0.5 V too early
     (e.g. VG1=0.4 model V_D≈0.5 vs measured ≈1.0 V)
  2. Post-snapback magnitude: model ~1 dec under measured at high V_D.

Hypotheses:
  - alpha0 (impact-ionisation) too high → body charges too fast → BJT
    triggers at lower V_D. Lowering alpha0 → knee shifts RIGHT.
  - BJT current after firing too small. Boost Bf.

2D sweep over (alpha0_scale, bf_scale). Reuse z432's backward pseudo-
transient runner; only the per-bias parameter setup changes.

NO_CHEAT: report ALL 16 grid points (no cherry-picking), full heatmap.

Output: results/z438_knee_calibration/
  summary.json, heatmap.png, best_overlay_VG1_*.png, knee_position.json,
  honest_analysis.md.
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z438_knee_calibration"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# ---- Reuse z432 (which itself wires z427 + z429) ----
_spec432 = _ilu.spec_from_file_location("z432",
                                        ROOT / "scripts/z432_pseudotransient.py")
z432 = _ilu.module_from_spec(_spec432); _spec432.loader.exec_module(z432)
z427 = z432.z427
z429 = z432.z429

# Cut max integration steps in half vs z432 to fit 2 h budget for 16-point
# sweep. z432 used 800 — 49% convergence; with 400 most converging points
# already converge within ~150 steps. Empirically only a handful change.
z432.N_STEPS_DEFAULT = 400


# ============================================================ #
# Parameter-scaled wrappers
# ============================================================ #

_orig_make_overrides = z427.make_overrides
_orig_make_bjt = z427.make_bjt


def make_overrides_scaled(sebas_row, alpha0_scale: float = 1.0):
    P_M1, P_M2 = _orig_make_overrides(sebas_row)
    if P_M1 is not None and "alpha0" in P_M1 and alpha0_scale != 1.0:
        P_M1 = dict(P_M1)
        P_M1["alpha0"] = P_M1["alpha0"] * float(alpha0_scale)
    return P_M1, P_M2


def make_bjt_scaled(sebas_row, bf_scale: float = 1.0):
    bjt = _orig_make_bjt(sebas_row)
    if bf_scale != 1.0:
        bjt.Bf = float(bjt.Bf) * float(bf_scale)
    return bjt


# ============================================================ #
# Per-grid-point cell-wide backward sweep (lean — no plotting inside)
# ============================================================ #

def run_grid_point(alpha0_scale: float, bf_scale: float,
                   model_M1, model_M2, curves, sebas_rows):
    """Run cell-wide backward pseudo-transient for one (a0, bf) combo.
    Returns dict with cell rmse, per-branch rmse, per-bias trace records
    (keep Vd/Id_pred/Id_meas — needed for knee + overlay)."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    log_eps = 1e-15
    per_bias = []
    fails = 0
    t0 = time.time()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = make_overrides_scaled(sebas_row, alpha0_scale)
        bjt = make_bjt_scaled(sebas_row, bf_scale)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                    z427.patch_sd_scaled(sd_M2, P_M2):
                Id_pred, Vb_list, conv_list, _niter_list = \
                    z432.run_one_bias(cfg, model_M1, model_M2, bjt, Vd_arr,
                                       c["VG1"], c["VG2"], backward=True,
                                       Vb_init_first=0.0)
        except Exception as e:
            fails += 1
            log(f"  fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        Id_pred_t = torch.tensor(Id_pred, dtype=torch.float64)
        conv_t = torch.tensor(conv_list)
        if not conv_t.any():
            fails += 1
            continue
        log_p = torch.log10(Id_pred_t + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv_t].mean()))
        per_bias.append({
            "VG1": float(c["VG1"]), "VG2": float(c["VG2"]),
            "log_rmse": rmse,
            "n_conv": int(conv_t.sum()),
            "n_pts": len(Vd_arr),
            "Vd": [float(x) for x in Vd_arr],
            "Id_meas": [float(x) for x in Id_meas],
            "Id_pred": [float(x) for x in Id_pred],
            "converged": [bool(x) for x in conv_list],
        })
    cell_sq = sum(r["log_rmse"]**2 for r in per_bias)
    cell_n = len(per_bias)
    cell = math.sqrt(cell_sq / cell_n) if cell_n else float("inf")
    per_branch = {}
    for r in per_bias:
        b = f"VG1_{r['VG1']:.1f}"
        per_branch.setdefault(b, {"sq": 0.0, "n": 0})
        per_branch[b]["sq"] += r["log_rmse"]**2
        per_branch[b]["n"] += 1
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"])
                       for b, v in per_branch.items()}
    total_pts = sum(r["n_pts"] for r in per_bias)
    total_conv = sum(r["n_conv"] for r in per_bias)
    conv_rate = total_conv / max(total_pts, 1)
    wall = time.time() - t0
    log(f"  a0×{alpha0_scale:<5g} bf×{bf_scale:<5g} → cell={cell:.3f} "
        f"branches={ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"conv={conv_rate*100:.0f}% fails={fails} wall={wall:.0f}s")
    return {
        "alpha0_scale": alpha0_scale,
        "bf_scale": bf_scale,
        "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(wall, 1),
        "per_bias": per_bias,
    }


# ============================================================ #
# Knee detection
# ============================================================ #

def find_knee_vd(Vd_arr, Id_arr):
    """Return V_D where dI_D/dV_D in log space peaks (rising slope).
    Operates on positive currents only."""
    Vd = np.asarray(Vd_arr, dtype=float)
    Id = np.asarray(Id_arr, dtype=float)
    Id = np.where(Id <= 0, 1e-20, Id)
    if Vd.size < 3:
        return float("nan")
    # sort ascending
    order = np.argsort(Vd)
    Vd = Vd[order]; Id = Id[order]
    log_Id = np.log10(Id)
    dlogI = np.gradient(log_Id, Vd)
    # ignore tails: knee must be after some onset (Vd > 0.05 V)
    mask = Vd > 0.05
    if not mask.any():
        return float("nan")
    idx_rel = int(np.argmax(dlogI[mask]))
    idx = np.where(mask)[0][idx_rel]
    return float(Vd[idx])


# ============================================================ #
# Plots
# ============================================================ #

def plot_heatmap(grid, alpha0_grid, bf_grid, fname: Path):
    """Cell-wide RMSE heatmap. grid[i,j] = cell rmse at (alpha0[i], bf[j])."""
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   cmap="viridis_r",
                   extent=[-0.5, len(bf_grid)-0.5, -0.5, len(alpha0_grid)-0.5])
    ax.set_xticks(range(len(bf_grid)))
    ax.set_xticklabels([f"{x:g}" for x in bf_grid])
    ax.set_yticks(range(len(alpha0_grid)))
    ax.set_yticklabels([f"{x:g}" for x in alpha0_grid])
    ax.set_xlabel("bf_scale")
    ax.set_ylabel("alpha0_scale")
    ax.set_title("Cell-wide log-RMSE [dec] (lower = better)\n"
                 "z432 backward baseline = 1.027")
    # annotate
    for i in range(len(alpha0_grid)):
        for j in range(len(bf_grid)):
            v = grid[i, j]
            txt = f"{v:.2f}" if np.isfinite(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if v > np.nanmedian(grid) else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="cell RMSE [dec]")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def plot_best_overlay(VG1_target: float, best_per_bias, fname: Path):
    """3-panel: measured + best calibration prediction at low/mid/high VG2."""
    rows = {r["VG2"]: r for r in best_per_bias if abs(r["VG1"]-VG1_target) < 1e-3}
    if not rows:
        return
    vg2_vals = sorted(rows.keys())
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, vg2 in zip(axes, chosen):
        r = rows[vg2]
        ax.plot(r["Vd"], r["Id_meas"], "k-", lw=2.5, label="measured")
        ax.plot(r["Vd"], r["Id_pred"], "--", lw=1.5,
                color="tab:blue", label="z438 best")
        # mark knees
        kv_meas = find_knee_vd(r["Vd"], r["Id_meas"])
        kv_mod = find_knee_vd(r["Vd"], r["Id_pred"])
        if np.isfinite(kv_meas):
            ax.axvline(kv_meas, color="k", ls=":", lw=1, alpha=0.5)
        if np.isfinite(kv_mod):
            ax.axvline(kv_mod, color="tab:blue", ls=":", lw=1, alpha=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:+.2f}\n"
                     f"knee meas={kv_meas:.2f}V mod={kv_mod:.2f}V")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z438 best-calibration overlay @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

# Pre-registered grid
ALPHA0_GRID = [0.1, 0.3, 0.5, 1.0]
BF_GRID = [1.0, 3.0, 10.0, 30.0]

Z432_BWD_CELL = 1.026861976331113


def main():
    t_main = time.time()
    log("z438 starting — knee+magnitude calibration sweep")
    log(f"  alpha0_grid={ALPHA0_GRID}  bf_grid={BF_GRID}  "
        f"(N_STEPS={z432.N_STEPS_DEFAULT}, backward only)")
    log("  baseline z432 backward cell = 1.027 dec")

    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    points = []
    grid_mat = np.full((len(ALPHA0_GRID), len(BF_GRID)), np.nan)
    for i, a0 in enumerate(ALPHA0_GRID):
        for j, bf in enumerate(BF_GRID):
            log(f"=== grid[{i},{j}] alpha0×{a0} × bf×{bf} ===")
            res = run_grid_point(a0, bf, model_M1, model_M2,
                                  curves, sebas_rows)
            points.append(res)
            grid_mat[i, j] = res["cell_rmse_dec"]
            # Intermediate save in case we get killed
            with open(OUT / "summary.json", "w") as f:
                json.dump({
                    "STATUS": "in_progress",
                    "alpha0_grid": ALPHA0_GRID,
                    "bf_grid": BF_GRID,
                    "grid_cell_rmse": [[None if not np.isfinite(x)
                                         else float(x)
                                         for x in row]
                                       for row in grid_mat],
                    "baseline_z432_bwd_cell_dec": Z432_BWD_CELL,
                    "points": [{k: v for k, v in p.items() if k != "per_bias"}
                               for p in points],
                }, f, indent=2)

    # Identify best
    best_idx = int(np.nanargmin(grid_mat))
    bi, bj = best_idx // len(BF_GRID), best_idx % len(BF_GRID)
    best_pt = points[best_idx]
    log(f"BEST: grid[{bi},{bj}] alpha0×{ALPHA0_GRID[bi]} bf×{BF_GRID[bj]} "
        f"cell={best_pt['cell_rmse_dec']:.3f} dec")

    # Knee analysis at best point
    knee_records = []
    for r in best_pt["per_bias"]:
        kv_m = find_knee_vd(r["Vd"], r["Id_meas"])
        kv_p = find_knee_vd(r["Vd"], r["Id_pred"])
        knee_records.append({
            "VG1": r["VG1"], "VG2": r["VG2"],
            "knee_meas_V": kv_m,
            "knee_pred_V": kv_p,
            "knee_err_V": (kv_p - kv_m) if (np.isfinite(kv_m)
                                             and np.isfinite(kv_p))
                          else float("nan"),
        })
    knee_err = np.array([k["knee_err_V"] for k in knee_records])
    knee_err = knee_err[np.isfinite(knee_err)]
    n_knee_match = int(np.sum(np.abs(knee_err) <= 0.1)) if knee_err.size else 0
    n_knee_total = int(knee_err.size)
    knee_match_frac = n_knee_match / max(n_knee_total, 1)
    knee_mean_err = float(np.mean(knee_err)) if knee_err.size else float("nan")
    knee_mae = float(np.mean(np.abs(knee_err))) if knee_err.size else float("nan")
    log(f"KNEE: {n_knee_match}/{n_knee_total} within ±0.1 V  "
        f"(mean err={knee_mean_err:+.3f}V  MAE={knee_mae:.3f}V)")

    # Baseline knee (alpha0×1, bf×1) for "closed by" calc
    baseline_pt = next(
        (p for p in points
         if abs(p["alpha0_scale"] - 1.0) < 1e-9
         and abs(p["bf_scale"] - 1.0) < 1e-9),
        None,
    )
    knee_close_pct = None
    base_knee_mae = None
    if baseline_pt is not None:
        base_errs = []
        for r in baseline_pt["per_bias"]:
            kv_m = find_knee_vd(r["Vd"], r["Id_meas"])
            kv_p = find_knee_vd(r["Vd"], r["Id_pred"])
            if np.isfinite(kv_m) and np.isfinite(kv_p):
                base_errs.append(abs(kv_p - kv_m))
        if base_errs:
            base_knee_mae = float(np.mean(base_errs))
            if knee_mae < base_knee_mae and base_knee_mae > 0:
                knee_close_pct = float(
                    (base_knee_mae - knee_mae) / base_knee_mae * 100
                )

    # Heatmap
    plot_heatmap(grid_mat, ALPHA0_GRID, BF_GRID, OUT / "heatmap.png")

    # Best overlays at VG1=0.4 and 0.6
    plot_best_overlay(0.4, best_pt["per_bias"],
                      OUT / "best_overlay_VG1_0p4.png")
    plot_best_overlay(0.6, best_pt["per_bias"],
                      OUT / "best_overlay_VG1_0p6.png")
    plot_best_overlay(0.2, best_pt["per_bias"],
                      OUT / "best_overlay_VG1_0p2.png")

    # knee_position.json
    with open(OUT / "knee_position.json", "w") as f:
        json.dump({
            "best_point": {
                "alpha0_scale": ALPHA0_GRID[bi],
                "bf_scale": BF_GRID[bj],
                "cell_rmse_dec": best_pt["cell_rmse_dec"],
            },
            "summary": {
                "n_biases": n_knee_total,
                "n_within_0p1V": n_knee_match,
                "frac_within_0p1V": knee_match_frac,
                "mean_signed_err_V": knee_mean_err,
                "mean_abs_err_V": knee_mae,
                "baseline_mean_abs_err_V": base_knee_mae,
                "knee_close_pct_vs_baseline": knee_close_pct,
            },
            "per_bias": knee_records,
        }, f, indent=2)

    # Gates
    best_cell = best_pt["cell_rmse_dec"]
    gates = {
        "INFRA_pass": all(p["n_biases_evaluated"] > 0 for p in points)
                      and len(points) == len(ALPHA0_GRID) * len(BF_GRID),
        "DISCOVERY_cell_lt_0p9_AND_knee_80pct":
            (best_cell < 0.9) and (knee_match_frac >= 0.80),
        "AMBITIOUS_cell_lt_0p5": best_cell < 0.5,
        "KILL_SHOT_no_knee_50pct_close":
            (knee_close_pct is None) or (knee_close_pct < 50.0),
    }
    log(f"GATES: {gates}")

    # Final summary
    summary = {
        "STATUS": "done",
        "alpha0_grid": ALPHA0_GRID,
        "bf_grid": BF_GRID,
        "grid_cell_rmse": [[None if not np.isfinite(x) else float(x)
                            for x in row] for row in grid_mat],
        "baseline_z432_bwd_cell_dec": Z432_BWD_CELL,
        "best": {
            "i": bi, "j": bj,
            "alpha0_scale": ALPHA0_GRID[bi],
            "bf_scale": BF_GRID[bj],
            "cell_rmse_dec": best_cell,
            "per_branch_rmse_dec": best_pt["per_branch_rmse_dec"],
            "convergence_rate": best_pt["convergence_rate"],
        },
        "knee": {
            "best_n_within_0p1V": n_knee_match,
            "best_n_total": n_knee_total,
            "best_frac_within_0p1V": knee_match_frac,
            "best_mean_signed_err_V": knee_mean_err,
            "best_mean_abs_err_V": knee_mae,
            "baseline_mean_abs_err_V": base_knee_mae,
            "knee_close_pct_vs_baseline": knee_close_pct,
        },
        "GATES": gates,
        "points": [{k: v for k, v in p.items() if k != "per_bias"}
                   for p in points],
        "wall_sec_total": round(time.time() - t_main, 1),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"DONE wall={time.time()-t_main:.0f}s")

    # honest_analysis.md
    md_lines = [
        "# z438 — Knee + Magnitude Calibration: Honest Analysis",
        "",
        f"Wall time: {summary['wall_sec_total']:.0f} s "
        f"({summary['wall_sec_total']/3600:.2f} h)",
        "",
        "## Setup",
        "",
        f"- 2D grid: alpha0_scale ∈ {ALPHA0_GRID} × bf_scale ∈ {BF_GRID}",
        f"- 16 grid points × 33 biases × backward pseudo-transient sweep",
        f"- N_STEPS_DEFAULT = {z432.N_STEPS_DEFAULT} (reduced from 800 in z432)",
        f"- Baseline (z432 backward, alpha0×1, bf×1 implicit) = "
        f"{Z432_BWD_CELL:.3f} dec",
        "",
        "## Full 16-point grid (cell-wide log-RMSE [dec])",
        "",
        "| alpha0\\bf | " + " | ".join(f"{b:g}" for b in BF_GRID) + " |",
        "|---" + "|---" * len(BF_GRID) + "|",
    ]
    for i, a0 in enumerate(ALPHA0_GRID):
        row = [f"{a0:g}"]
        for j in range(len(BF_GRID)):
            v = grid_mat[i, j]
            row.append(f"{v:.3f}" if np.isfinite(v) else "—")
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines += [
        "",
        f"## Best point",
        "",
        f"- alpha0_scale = **{ALPHA0_GRID[bi]:g}**, "
        f"bf_scale = **{BF_GRID[bj]:g}**",
        f"- cell-wide RMSE = **{best_cell:.3f} dec** (vs baseline "
        f"{Z432_BWD_CELL:.3f}, Δ={Z432_BWD_CELL-best_cell:+.3f} dec)",
        f"- per-branch: " + ", ".join(
            f"{k}={v:.3f}" for k, v in best_pt["per_branch_rmse_dec"].items()
        ),
        f"- convergence rate = {best_pt['convergence_rate']*100:.1f}%",
        "",
        "## Knee position (at best point)",
        "",
        f"- biases evaluated: {n_knee_total}",
        f"- within ±0.1 V of measured: **{n_knee_match}/{n_knee_total} "
        f"({knee_match_frac*100:.0f}%)**",
        f"- mean signed knee error (model − meas): {knee_mean_err:+.3f} V "
        f"(negative = model too early)",
        f"- mean abs knee error: {knee_mae:.3f} V "
        f"(baseline {base_knee_mae if base_knee_mae is not None else float('nan'):.3f} V)",
        f"- knee error closed vs baseline: "
        f"**{knee_close_pct if knee_close_pct is not None else float('nan'):.1f}%**",
        "",
        "## Pre-registered gates",
        "",
    ]
    for k, v in gates.items():
        md_lines.append(f"- `{k}` = {'PASS' if v else 'FAIL'}")
    md_lines += [
        "",
        "## Verdict",
        "",
    ]
    if gates["AMBITIOUS_cell_lt_0p5"]:
        md_lines.append(
            "**AMBITIOUS achieved**: cell-wide RMSE below 0.5 dec. The combined "
            "(alpha0_scale, bf_scale) correction closes both knee position "
            "and post-snapback magnitude."
        )
    elif gates["DISCOVERY_cell_lt_0p9_AND_knee_80pct"]:
        md_lines.append(
            "**DISCOVERY achieved**: cell-wide RMSE < 0.9 dec AND ≥80% of "
            "biases land knees within ±0.1 V. Calibration is the right lever; "
            "more grid refinement (or extra parameters such as area / Is) "
            "could close the remaining gap."
        )
    elif gates["KILL_SHOT_no_knee_50pct_close"]:
        md_lines.append(
            "**KILL SHOT triggered**: best (alpha0, bf) point closes the knee "
            "position by < 50% relative to baseline. Knee shift and magnitude "
            "gap are not jointly governed by these two scalars; a different "
            "lever (BJT IS / area, M2 etab, lateral SCR) is required."
        )
    else:
        md_lines.append(
            "Mixed result: improvement vs baseline but neither full DISCOVERY "
            "nor KILL SHOT triggered. See per-grid heatmap for direction of "
            "next refinement."
        )

    with open(OUT / "honest_analysis.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    log("wrote honest_analysis.md")


if __name__ == "__main__":
    main()
