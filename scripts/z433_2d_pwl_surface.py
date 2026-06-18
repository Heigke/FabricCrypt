"""z433 — 2D PWL surfaces over Sebas BSIM per-bias fits.

Hypothesis: per-row lookup in z430 makes the model bias-step-discrete
(or even unavailable at NaN rows). Build smooth 2D PWL(V_G1, V_G2)
surfaces over (etab, k1, nfactor, mbjt) using LinearNDInterpolator,
backstopped by NearestNDInterpolator outside the convex hull.

Then re-run cell-wide V_SINT_PIN with PWL-interpolated overrides. We
also fill the previously skipped NaN rows (VG1=0.4/0.6 with VG2<0)
because those biases have measured curves but no per-row fit.

Pre-registered gates (vs z430 V_SINT_PIN baseline):
  INFRA:      script runs to completion
  DISCOVERY:  VG1=0.2 branch <= 2.13 dec  (z430: 2.624 → expect ≥0.5 dec drop)
  AMBITIOUS:  cell-wide   <  1.0  dec
  KILL_SHOT:  VG1=0.2 branch unchanged (|Δ| < 0.1 dec) → 2D PWL doesn't help
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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z433_2d_pwl_surface"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# --- reuse z427 + z429 + z91f modules
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)
_spec91f = _ilu.spec_from_file_location("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = _ilu.module_from_spec(_spec91f); _spec91f.loader.exec_module(z91f)

from nsram.bsim4_port.nsram_cell_2T import forward_2t  # noqa


# ============================================================ #
# 2D PWL surface construction
# ============================================================ #

PARAM_FIELDS = ["ETAB", "K1", "NFACTOR", "mbjt", "ALPHA0", "BETA0", "IS", "area"]


def build_pwl_surfaces(sebas_rows):
    """Build LinearND + Nearest fallback for each parameter over (VG1,VG2).

    Returns dict[name] -> (lin_interp, near_interp, points, values).
    Uses only rows with non-NaN K1 (valid fits). For mbjt, also include
    NaN-K1 rows because mbjt is defined for those (0.001 vs 1.0).
    """
    surfaces = {}
    # Valid rows for fit params (have K1)
    valid = [r for r in sebas_rows
             if not math.isnan(r.get("K1", float("nan")))]
    pts_valid = np.array([[r["VG1"], r["VG2"]] for r in valid])
    # All rows for mbjt (always defined)
    all_rows = [r for r in sebas_rows
                if not math.isnan(r.get("mbjt", float("nan")))]
    pts_all = np.array([[r["VG1"], r["VG2"]] for r in all_rows])

    for f in PARAM_FIELDS:
        if f == "mbjt":
            pts = pts_all
            vals = np.array([r[f] for r in all_rows], dtype=float)
        else:
            pts = pts_valid
            vals = np.array([float(r.get(f, float("nan"))) for r in valid])
        # Skip if all NaN
        if np.all(np.isnan(vals)):
            surfaces[f] = None
            continue
        # Filter out NaN values for this param
        mask = ~np.isnan(vals)
        p_use = pts[mask]
        v_use = vals[mask]
        try:
            lin = LinearNDInterpolator(p_use, v_use)
        except Exception as e:
            log(f"  LinearND fail for {f}: {e}; using Nearest only")
            lin = None
        near = NearestNDInterpolator(p_use, v_use)
        surfaces[f] = {"lin": lin, "near": near, "pts": p_use, "vals": v_use}
        log(f"  surface[{f}]: n_pts={len(p_use)} range=[{v_use.min():.4g},{v_use.max():.4g}]")
    return surfaces


def interp_param(surfaces, name, VG1, VG2):
    """Linear interp w/ Nearest fallback outside hull (returns NaN-safe float)."""
    s = surfaces.get(name)
    if s is None:
        return None
    if s["lin"] is not None:
        v = float(s["lin"](VG1, VG2))
        if not math.isnan(v):
            return v
    return float(s["near"](VG1, VG2))


def make_overrides_pwl(surfaces, VG1, VG2):
    """Like z91f.make_overrides but uses 2D PWL surfaces. Always returns
    overrides (never None) because nearest-fallback always resolves."""
    P_M1 = {}
    etab = interp_param(surfaces, "ETAB", VG1, VG2)
    if etab is not None:
        P_M1["etab"] = torch.tensor(etab, dtype=torch.float64)
    k1 = interp_param(surfaces, "K1", VG1, VG2)
    if k1 is not None:
        P_M1["k1"] = torch.tensor(k1, dtype=torch.float64)
    alpha0 = interp_param(surfaces, "ALPHA0", VG1, VG2)
    if alpha0 is not None:
        P_M1["alpha0"] = torch.tensor(alpha0, dtype=torch.float64)
    beta0 = interp_param(surfaces, "BETA0", VG1, VG2)
    if beta0 is not None:
        P_M1["beta0"] = torch.tensor(beta0, dtype=torch.float64)

    P_M2 = {}
    nfac = interp_param(surfaces, "NFACTOR", VG1, VG2)
    if nfac is not None:
        P_M2["nfactor"] = torch.tensor(nfac, dtype=torch.float64)
    # Static M2 deltas from z91f
    for k, v in z91f.M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=torch.float64)

    return P_M1 or None, P_M2 or None


def make_bjt_pwl(surfaces, VG1, VG2):
    """BJT with PWL-interpolated mbjt/area/IS."""
    from nsram.bsim4_port.bjt import GummelPoonNPN
    bjt = GummelPoonNPN.from_sebas_card()
    IS_v = interp_param(surfaces, "IS", VG1, VG2)
    if IS_v is not None and not math.isnan(IS_v):
        bjt.Is = float(IS_v)
    area = interp_param(surfaces, "area", VG1, VG2)
    if area is None or math.isnan(area):
        area = 1e-6
    mbjt = interp_param(surfaces, "mbjt", VG1, VG2)
    if mbjt is None or math.isnan(mbjt):
        mbjt = 1.0
    # Clamp mbjt to non-negative (linear interp could in principle dip)
    mbjt = max(mbjt, 1e-6)
    bjt.area = float(area) * float(mbjt)
    return bjt


# ============================================================ #
# V_SINT_PIN runner with PWL overrides
# ============================================================ #

def run_vsint_pin_pwl(name: str, surfaces, model_M1, model_M2, curves):
    """V_SINT_PIN over all curves, using 2D PWL surfaces (no row lookup)."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
    fails = 0
    t0 = time.time()
    for c in curves:
        VG1, VG2 = float(c["VG1"]), float(c["VG2"])
        P_M1, P_M2 = make_overrides_pwl(surfaces, VG1, VG2)
        bjt = make_bjt_pwl(surfaces, VG1, VG2)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        Id_pred_list = []
        Vb_list = []
        conv_list = []
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k, Vd_f in enumerate(Vd_arr):
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), VG1, VG2,
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred_list.append(abs(r["Id"]))
                    Vb_list.append(r["Vb"])
                    conv_list.append(bool(r["converged"]))
                    Vb_warm = r["Vb"] if r["converged"] else 0.0
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={VG1} VG2={VG2}: {e}")
            continue
        Id_pred = torch.tensor(Id_pred_list, dtype=torch.float64)
        conv = torch.tensor(conv_list)
        if not conv.any():
            fails += 1
            continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv].mean()))
        vb_max = float(max(Vb_list))
        vb_max_overall = max(vb_max_overall, vb_max)
        rec = {"VG1": VG1, "VG2": VG2,
               "log_rmse": rmse, "vb_max": vb_max,
               "n_conv": int(conv.sum()), "n_pts": len(Vd_arr),
               "Vd": Vd_arr.tolist(),
               "Id_meas": Id_meas.tolist(),
               "Id_pred": Id_pred.tolist(),
               "Vb": Vb_list,
               "Vsint": [0.0] * len(Vd_arr),
               "converged": conv_list,
               "params_used": {
                   "etab": float(P_M1["etab"]) if P_M1 and "etab" in P_M1 else None,
                   "k1": float(P_M1["k1"]) if P_M1 and "k1" in P_M1 else None,
                   "nfactor": float(P_M2["nfactor"]) if P_M2 and "nfactor" in P_M2 else None,
                   "mbjt_effective_area": bjt.area,
               }}
        per_bias.append(rec)
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
    log(f"  {name}: cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"Vb_max={vb_max_overall:.3f} conv_rate={conv_rate*100:.1f}% fails={fails} "
        f"wall={time.time()-t0:.0f}s")
    return {
        "name": name, "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "per_bias": per_bias,
    }


# ============================================================ #
# Plots
# ============================================================ #

def plot_param_surfaces(surfaces, fname):
    """4-subplot 2D PWL surfaces for etab, k1, nfactor, mbjt."""
    fields = ["ETAB", "K1", "NFACTOR", "mbjt"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    VG1_grid = np.linspace(0.15, 0.65, 60)
    VG2_grid = np.linspace(-0.25, 0.55, 80)
    G1, G2 = np.meshgrid(VG1_grid, VG2_grid, indexing="ij")
    for ax, f in zip(axes.flat, fields):
        s = surfaces.get(f)
        if s is None:
            ax.set_title(f"{f}: no data"); continue
        # Compute interpolated grid (lin w/ nearest fallback)
        Z = np.empty_like(G1)
        for i in range(G1.shape[0]):
            for j in range(G1.shape[1]):
                Z[i, j] = interp_param(surfaces, f, G1[i, j], G2[i, j])
        im = ax.pcolormesh(G1, G2, Z, shading="auto", cmap="viridis")
        ax.scatter(s["pts"][:, 0], s["pts"][:, 1], c=s["vals"],
                   edgecolors="white", s=40, linewidth=0.8, cmap="viridis")
        ax.set_xlabel("VG1 [V]"); ax.set_ylabel("VG2 [V]")
        ax.set_title(f"{f} surface (n_fit={len(s['pts'])})")
        plt.colorbar(im, ax=ax)
    fig.suptitle("z433: 2D PWL parameter surfaces (LinearND, Nearest fallback)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def overlay_plot(VG1_target, results, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    variants = [(n, r) for n, r in results.items() if r.get("per_bias")]
    rows_by_vg2 = {}
    for name, r in variants:
        for rec in r["per_bias"]:
            if abs(rec["VG1"] - VG1_target) < 1e-3:
                rows_by_vg2.setdefault(rec["VG2"], {})[name] = rec
    vg2_vals = sorted(rows_by_vg2.keys())
    if not vg2_vals:
        log(f"  no data at VG1={VG1_target}; skip {fname.name}")
        plt.close(fig); return
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    colors = {"Z430_V_SINT_PIN": "tab:green", "Z433_PWL_V_SINT_PIN": "tab:blue"}
    for ax, vg2 in zip(axes, chosen):
        sub = rows_by_vg2.get(vg2, {})
        meas = None
        for name in colors.keys():
            if name in sub:
                meas = sub[name]; break
        if meas is None:
            ax.set_title(f"VG2={vg2:.2f} (no data)"); continue
        ax.plot(meas["Vd"], meas["Id_meas"], "k-", lw=2.5, label="measured")
        for name, rec in sub.items():
            ax.plot(rec["Vd"], rec["Id_pred"], "--", lw=1.5,
                    color=colors.get(name, "gray"), label=name)
        ax.set_yscale("log"); ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z433: 2D PWL surfaces vs z430 row-lookup @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

def main():
    t_main = time.time()
    log("z433 starting — 2D PWL surfaces over Sebas BSIM fits")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    log("=== Building 2D PWL surfaces ===")
    surfaces = build_pwl_surfaces(sebas_rows)

    # Plot the surfaces
    plot_param_surfaces(surfaces, OUT / "per_param_surface_plots.png")

    # Quick sanity: report interpolated params at every measured-curve bias
    sanity = []
    for c in curves:
        VG1, VG2 = float(c["VG1"]), float(c["VG2"])
        d = {"VG1": VG1, "VG2": VG2}
        for f in ["ETAB", "K1", "NFACTOR", "mbjt"]:
            v = interp_param(surfaces, f, VG1, VG2)
            d[f] = v
        sanity.append(d)
    (OUT / "interpolated_params_at_curves.json").write_text(
        json.dumps(sanity, indent=2))
    log(f"wrote interpolated_params_at_curves.json ({len(sanity)} biases)")

    results = {}

    log("=== Z433_PWL_V_SINT_PIN (2D PWL overrides + V_SINT_PIN solver) ===")
    results["Z433_PWL_V_SINT_PIN"] = run_vsint_pin_pwl(
        "Z433_PWL_V_SINT_PIN", surfaces, model_M1, model_M2, curves)

    # Re-run z430 baseline pin for direct comparison (use cached if exists)
    z430_summary_path = ROOT / "results/z430_vsint_pin_cellwide/summary.json"
    if z430_summary_path.exists():
        z430_summary = json.loads(z430_summary_path.read_text())
        z430_pin = z430_summary.get("V_SINT_PIN", {})
        log(f"Z430 V_SINT_PIN reference: cell={z430_pin.get('cell_rmse_dec'):.3f} "
            f"per_branch={z430_pin.get('per_branch_rmse_dec')}")
    else:
        z430_pin = {}
        log("WARN: z430 summary not found")

    # Summary
    z433 = results["Z433_PWL_V_SINT_PIN"]
    summary = {
        "Z433_PWL_V_SINT_PIN": {
            "cell_rmse_dec": z433["cell_rmse_dec"],
            "per_branch_rmse_dec": z433["per_branch_rmse_dec"],
            "n_biases_evaluated": z433["n_biases_evaluated"],
            "vb_max_overall": z433["vb_max_overall"],
            "convergence_rate": z433["convergence_rate"],
            "fails": z433["fails"],
            "wall_sec": z433["wall_sec"],
        },
        "Z430_V_SINT_PIN_reference": {
            "cell_rmse_dec": z430_pin.get("cell_rmse_dec"),
            "per_branch_rmse_dec": z430_pin.get("per_branch_rmse_dec"),
            "n_biases_evaluated": z430_pin.get("n_biases_evaluated"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote summary.json")

    # Pre-registered gates
    z430_cell = z430_pin.get("cell_rmse_dec", 1.618)
    z430_vg1_02 = z430_pin.get("per_branch_rmse_dec", {}).get("VG1_0.2", 2.624)
    new_vg1_02 = z433["per_branch_rmse_dec"].get("VG1_0.2", float("nan"))
    new_cell = z433["cell_rmse_dec"]
    delta_vg1_02 = z430_vg1_02 - new_vg1_02 if not math.isnan(new_vg1_02) else float("nan")

    gates = {
        "INFRA_pass": z433["n_biases_evaluated"] > 0,
        "DISCOVERY_pass_vg1_0p2_le_2p13": (not math.isnan(new_vg1_02)) and new_vg1_02 <= 2.13,
        "AMBITIOUS_pass_cell_lt_1p0": new_cell < 1.0,
        "KILL_SHOT_pwl_no_effect": (not math.isnan(delta_vg1_02)) and abs(delta_vg1_02) < 0.1,
    }

    ablation = {
        "z430_reference": {
            "cell_rmse_dec": z430_cell,
            "per_branch_rmse_dec": z430_pin.get("per_branch_rmse_dec"),
        },
        "z433_pwl": {
            "cell_rmse_dec": new_cell,
            "per_branch_rmse_dec": z433["per_branch_rmse_dec"],
        },
        "delta_cell_dec": z430_cell - new_cell,
        "delta_per_branch_dec": {
            b: (z430_pin.get("per_branch_rmse_dec", {}).get(b, float("nan"))
                - z433["per_branch_rmse_dec"].get(b, float("nan")))
            for b in set(list(z433["per_branch_rmse_dec"].keys())
                         + list(z430_pin.get("per_branch_rmse_dec", {}).keys()))
        },
        "verdict_gates": gates,
    }
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2))
    log(f"wrote ablation.json")

    # Overlays — combine z433 PWL + z430 PIN (load per-bias from z430 if possible)
    # z430 stores per_bias only inside its summary.json? No, in ablation.json.
    z430_ablation_path = ROOT / "results/z430_vsint_pin_cellwide/ablation.json"
    z430_per_bias = []
    # z430 doesn't dump per_bias. Just plot z433 alone overlaid on measured.
    plot_results = {"Z433_PWL_V_SINT_PIN": results["Z433_PWL_V_SINT_PIN"]}
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, plot_results, OUT / f"overlay_VG1_{suffix}.png")

    # Honest analysis
    h = []
    h.append("# z433 — 2D PWL surfaces over Sebas BSIM per-bias fits\n\n")
    h.append("## Method\n")
    h.append("- LinearNDInterpolator over (V_G1, V_G2) for each of: ETAB, K1, NFACTOR, mbjt, ALPHA0, BETA0, IS, area.\n")
    h.append("- NearestNDInterpolator fallback when query is outside the convex hull (i.e. all extrapolation goes to nearest valid fit).\n")
    h.append("- Applied via V_SINT_PIN solver (z429's hard pin on V_Sint=0, 1D Newton on V_B) — best from z430.\n")
    h.append("- mbjt is clamped to >= 1e-6 to prevent negative-area from linear interp wobble.\n\n")
    h.append("## Surface coverage\n")
    h.append(f"- Sebas CSV: {len(sebas_rows)} rows total, {sum(1 for r in sebas_rows if not math.isnan(r.get('K1', float('nan'))))} with valid K1 (used for fit-param surfaces).\n")
    h.append(f"- Measured curves evaluated: {len(curves)}.\n")
    h.append(f"- z430 V_SINT_PIN evaluated {z430_pin.get('n_biases_evaluated', '-')} biases (skipped NaN rows).\n")
    h.append(f"- z433 PWL evaluated {z433['n_biases_evaluated']} biases (no row lookup — every bias gets PWL params).\n\n")
    h.append("## Results vs z430 V_SINT_PIN\n```\n")
    h.append(json.dumps(ablation, indent=2))
    h.append("\n```\n\n")
    h.append("## Per-bias residuals (Z433_PWL_V_SINT_PIN)\n```\n")
    for rec in z433["per_bias"]:
        p = rec.get("params_used", {})
        h.append(f"VG1={rec['VG1']:.2f} VG2={rec['VG2']:+.2f}  "
                 f"RMSE={rec['log_rmse']:.3f} dec  Vb_max={rec['vb_max']:+.3f}  "
                 f"conv={rec['n_conv']}/{rec['n_pts']}  "
                 f"etab={p.get('etab')} k1={p.get('k1')} nfac={p.get('nfactor')}\n")
    h.append("```\n\n")
    h.append("## Verdict\n")
    h.append(f"- z430 V_SINT_PIN cell-wide: {z430_cell:.3f} dec (VG1=0.2 branch: {z430_vg1_02:.3f})\n")
    h.append(f"- z433 PWL V_SINT_PIN cell-wide: {new_cell:.3f} dec (VG1=0.2 branch: {new_vg1_02:.3f})\n")
    h.append(f"- Δ cell: {ablation['delta_cell_dec']:+.3f} dec\n")
    h.append(f"- Δ VG1=0.2: {delta_vg1_02:+.3f} dec\n\n")
    h.append(f"- INFRA: {'PASS' if gates['INFRA_pass'] else 'FAIL'}\n")
    h.append(f"- DISCOVERY (VG1=0.2 ≤ 2.13 dec): {'PASS' if gates['DISCOVERY_pass_vg1_0p2_le_2p13'] else 'FAIL'}\n")
    h.append(f"- AMBITIOUS (cell < 1.0 dec): {'PASS' if gates['AMBITIOUS_pass_cell_lt_1p0'] else 'FAIL'}\n")
    h.append(f"- KILL_SHOT (PWL no effect on VG1=0.2, |Δ|<0.1 dec): {'TRIGGERED' if gates['KILL_SHOT_pwl_no_effect'] else 'no'}\n")
    (OUT / "honest_analysis.md").write_text("".join(h))
    log(f"wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
