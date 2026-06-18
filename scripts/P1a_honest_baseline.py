"""P1a — Honest fwd+bwd re-run of z430 V_SINT_PIN, z432 PTRAN, z443 VBIC_AVL.

For each pipeline:
  - run V_D sweep both forward (0.05 -> 2.0 V) and backward (2.0 -> 0.05 V)
  - record per-curve log-RMSE, per-branch RMSE (VG1=0.2/0.4/0.6),
    cell-wide RMSE, convergence rate, wall time
  - report averages

No 4-bias subsets. No physics changes — only adding bwd sweep loop.

Output: results/P1a_honest_baseline/summary.json
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

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/P1a_honest_baseline"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG_FH = open(OUT / "run.log", "w")
def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n"); LOG_FH.flush()


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m


z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")
z429 = _load("z429", ROOT / "scripts/z429_multisolver_debug.py")
z432 = _load("z432", ROOT / "scripts/z432_pseudotransient.py")


LOG_EPS = 1e-15


def _aggregate(per_bias):
    """Compute cell-wide, per-branch RMSE, conv rate from per-bias list."""
    cell_sq = sum(r["log_rmse"]**2 for r in per_bias)
    cell_n = len(per_bias)
    cell = math.sqrt(cell_sq / cell_n) if cell_n else float("inf")
    per_branch = {}
    for r in per_bias:
        b = f"VG1_{r['VG1']:.1f}"
        per_branch.setdefault(b, {"sq": 0.0, "n": 0})
        per_branch[b]["sq"] += r["log_rmse"]**2
        per_branch[b]["n"] += 1
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"]) for b, v in per_branch.items()}
    total_pts = sum(r["n_pts"] for r in per_bias)
    total_conv = sum(r["n_conv"] for r in per_bias)
    conv_rate = total_conv / max(total_pts, 1)
    return cell, per_branch_rmse, conv_rate


# ============================================================ #
# z430 V_SINT_PIN (hard pin, 1D Newton on V_B per Vd point)
# ============================================================ #

def run_z430_vsint_pin(model_M1, model_M2, curves, sebas_rows, direction):
    """direction in {'forward','backward'}. Uses z429.run_vsint_pinned per Vd."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    per_bias = []
    fails = 0
    t0 = time.time()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        n = len(Vd_arr)
        order = list(range(n)) if direction == "forward" else list(range(n - 1, -1, -1))
        Id_pred = [None] * n
        Vb_list = [None] * n
        conv_list = [False] * n
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for idx in order:
                    Vd_f = float(Vd_arr[idx])
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        Vd_f, float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred[idx] = abs(r["Id"])
                    Vb_list[idx] = r["Vb"]
                    conv_list[idx] = bool(r["converged"])
                    Vb_warm = r["Vb"] if r["converged"] else 0.0
        except Exception as e:
            fails += 1
            log(f"  z430.{direction} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        Id_pred_t = torch.tensor(Id_pred, dtype=torch.float64)
        conv_t = torch.tensor(conv_list)
        if not conv_t.any():
            fails += 1
            continue
        log_p = torch.log10(Id_pred_t + LOG_EPS)
        log_m = torch.log10(c["Id"] + LOG_EPS)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv_t].mean()))
        per_bias.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse,
            "n_conv": int(conv_t.sum()),
            "n_pts": n,
            "vb_max": float(max(v for v in Vb_list if v is not None)),
        })
    cell, per_branch, conv_rate = _aggregate(per_bias)
    wall = time.time() - t0
    log(f"  z430.{direction}: cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch.items()} } "
        f"conv={conv_rate*100:.1f}% fails={fails} wall={wall:.0f}s")
    return dict(direction=direction, cell_rmse_dec=cell, per_branch_rmse_dec=per_branch,
                n_biases_evaluated=len(per_bias),
                convergence_rate=conv_rate, fails=fails, wall_sec=round(wall, 1),
                per_bias=per_bias)


# ============================================================ #
# z432 PTRAN — reuse existing run_cellwide (already supports direction)
# ============================================================ #

def run_z432_ptran(model_M1, model_M2, curves, sebas_rows, direction):
    res = z432.run_cellwide("PTRAN", model_M1, model_M2, curves, sebas_rows,
                            direction=direction)
    # Strip heavy per-bias arrays; keep summary scalars + per-curve rmse
    slim = []
    for r in res["per_bias"]:
        slim.append({
            "VG1": r["VG1"], "VG2": r["VG2"],
            "log_rmse": r["log_rmse"],
            "n_conv": r["n_conv"], "n_pts": r["n_pts"],
            "vb_max": r["vb_max"],
        })
    return dict(direction=direction,
                cell_rmse_dec=res["cell_rmse_dec"],
                per_branch_rmse_dec=res["per_branch_rmse_dec"],
                n_biases_evaluated=res["n_biases_evaluated"],
                convergence_rate=res["convergence_rate"],
                fails=res["fails"],
                wall_sec=res["wall_sec"],
                per_bias=slim)


# ============================================================ #
# z443 VBIC_AVL — fwd + bwd
# ============================================================ #

def run_z443_vbic_avl(model_M1, model_M2, curves, sebas_rows, direction):
    """VBIC_AVL: AVC1=AVC2=0.5 Si defaults. Uses z429.run_vsint_pinned per Vd."""
    flags = {"use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5}
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    per_bias = []
    fails = 0
    t0 = time.time()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        n = len(Vd_arr)
        order = list(range(n)) if direction == "forward" else list(range(n - 1, -1, -1))
        Id_pred = [None] * n
        Vb_list = [None] * n
        conv_list = [False] * n
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for idx in order:
                    Vd_f = float(Vd_arr[idx])
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        Vd_f, float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred[idx] = abs(r["Id"])
                    Vb_list[idx] = r["Vb"]
                    conv_list[idx] = bool(r["converged"])
                    Vb_warm = r["Vb"] if r["converged"] else 0.0
        except Exception as e:
            fails += 1
            log(f"  z443.{direction} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        Id_pred_t = torch.tensor(Id_pred, dtype=torch.float64)
        conv_t = torch.tensor(conv_list)
        if not conv_t.any():
            fails += 1
            continue
        log_p = torch.log10(Id_pred_t + LOG_EPS)
        log_m = torch.log10(c["Id"] + LOG_EPS)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv_t].mean()))
        per_bias.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse,
            "n_conv": int(conv_t.sum()), "n_pts": n,
            "vb_max": float(max(v for v in Vb_list if v is not None)),
        })
    cell, per_branch, conv_rate = _aggregate(per_bias)
    wall = time.time() - t0
    log(f"  z443.{direction}: cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch.items()} } "
        f"conv={conv_rate*100:.1f}% fails={fails} wall={wall:.0f}s")
    return dict(direction=direction, cell_rmse_dec=cell, per_branch_rmse_dec=per_branch,
                n_biases_evaluated=len(per_bias),
                convergence_rate=conv_rate, fails=fails, wall_sec=round(wall, 1),
                per_bias=per_bias)


# ============================================================ #
# Combine fwd/bwd into pipeline summary
# ============================================================ #

def pipeline_summary(name, fwd, bwd):
    # Per-curve fwd / bwd / avg, keyed by (VG1,VG2)
    fwd_map = {(r["VG1"], r["VG2"]): r["log_rmse"] for r in fwd["per_bias"]}
    bwd_map = {(r["VG1"], r["VG2"]): r["log_rmse"] for r in bwd["per_bias"]}
    keys = sorted(set(fwd_map) | set(bwd_map))
    per_curve_RMSE_fwd = []
    per_curve_RMSE_bwd = []
    per_curve_RMSE_avg = []
    per_curve_keys = []
    for k in keys:
        f = fwd_map.get(k); b = bwd_map.get(k)
        per_curve_keys.append({"VG1": k[0], "VG2": k[1]})
        per_curve_RMSE_fwd.append(f)
        per_curve_RMSE_bwd.append(b)
        if f is not None and b is not None:
            per_curve_RMSE_avg.append(math.sqrt(0.5 * (f**2 + b**2)))
        else:
            per_curve_RMSE_avg.append(f if b is None else b)
    # Cell-wide RMS-average (quadratic mean of fwd and bwd cell)
    cw_fwd = fwd["cell_rmse_dec"]; cw_bwd = bwd["cell_rmse_dec"]
    if math.isfinite(cw_fwd) and math.isfinite(cw_bwd):
        cw_avg = math.sqrt(0.5 * (cw_fwd**2 + cw_bwd**2))
    else:
        cw_avg = float("inf")
    return {
        "pipeline": name,
        "n_curves_fwd": len(fwd["per_bias"]),
        "n_curves_bwd": len(bwd["per_bias"]),
        "per_curve_keys": per_curve_keys,
        "per_curve_RMSE_fwd": per_curve_RMSE_fwd,
        "per_curve_RMSE_bwd": per_curve_RMSE_bwd,
        "per_curve_RMSE_avg": per_curve_RMSE_avg,
        "per_branch_RMSE_fwd": fwd["per_branch_rmse_dec"],
        "per_branch_RMSE_bwd": bwd["per_branch_rmse_dec"],
        "cell_wide_fwd": cw_fwd,
        "cell_wide_bwd": cw_bwd,
        "cell_wide_avg": cw_avg,
        "convergence_rate_fwd": fwd["convergence_rate"],
        "convergence_rate_bwd": bwd["convergence_rate"],
        "fails_fwd": fwd["fails"],
        "fails_bwd": bwd["fails"],
        "wall_sec_fwd": fwd["wall_sec"],
        "wall_sec_bwd": bwd["wall_sec"],
    }


# ============================================================ #
# Main
# ============================================================ #

def main():
    t_main = time.time()
    log("P1a starting — honest fwd+bwd re-run of z430/z432/z443")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    summary = {"pipelines": {}}

    # --- z430 V_SINT_PIN ---
    log("=== z430 V_SINT_PIN forward ===")
    z430_fwd = run_z430_vsint_pin(model_M1, model_M2, curves, sebas_rows, "forward")
    log("=== z430 V_SINT_PIN backward ===")
    z430_bwd = run_z430_vsint_pin(model_M1, model_M2, curves, sebas_rows, "backward")
    summary["pipelines"]["z430_V_SINT_PIN"] = pipeline_summary("z430_V_SINT_PIN",
                                                               z430_fwd, z430_bwd)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("  partial summary.json written (z430 done)")

    # --- z432 PTRAN ---
    log("=== z432 PTRAN forward ===")
    z432_fwd = run_z432_ptran(model_M1, model_M2, curves, sebas_rows, "forward")
    log("=== z432 PTRAN backward ===")
    z432_bwd = run_z432_ptran(model_M1, model_M2, curves, sebas_rows, "backward")
    summary["pipelines"]["z432_PTRAN"] = pipeline_summary("z432_PTRAN",
                                                          z432_fwd, z432_bwd)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("  partial summary.json written (z432 done)")

    # --- z443 VBIC_AVL ---
    log("=== z443 VBIC_AVL forward ===")
    z443_fwd = run_z443_vbic_avl(model_M1, model_M2, curves, sebas_rows, "forward")
    log("=== z443 VBIC_AVL backward ===")
    z443_bwd = run_z443_vbic_avl(model_M1, model_M2, curves, sebas_rows, "backward")
    summary["pipelines"]["z443_VBIC_AVL"] = pipeline_summary("z443_VBIC_AVL",
                                                             z443_fwd, z443_bwd)

    summary["total_wall_sec"] = round(time.time() - t_main, 1)
    summary["notes"] = {
        "dataset": "Sebas 33-curve set (find_params hits ~25 biases)",
        "fwd_direction": "V_D 0.05 -> 2.0 V (warm-start V_B from prev)",
        "bwd_direction": "V_D 2.0 -> 0.05 V (warm-start V_B from prev)",
        "rmse": "log10-RMSE over converged V_D points, then quadratic mean across biases",
        "cell_wide_avg": "sqrt(0.5*(cell_fwd^2 + cell_bwd^2)), quadratic mean",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE total wall={summary['total_wall_sec']:.0f}s — wrote summary.json")

    # Print final compact table
    log("\n=== FINAL TABLE ===")
    log(f"{'pipeline':<22} {'fwd':>8} {'bwd':>8} {'avg':>8} {'conv_f':>7} {'conv_b':>7}")
    for n, p in summary["pipelines"].items():
        log(f"{n:<22} {p['cell_wide_fwd']:>8.3f} {p['cell_wide_bwd']:>8.3f} "
            f"{p['cell_wide_avg']:>8.3f} {p['convergence_rate_fwd']*100:>6.1f}% "
            f"{p['convergence_rate_bwd']*100:>6.1f}%")
    LOG_FH.close()


if __name__ == "__main__":
    main()
