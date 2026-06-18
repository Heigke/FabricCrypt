"""z446 — Combined VBIC (z443) + Pseudo-Transient backward sweep (z432).

Two orthogonal mechanisms expected to be additive:
  * VBIC level-4 NPN (use_vbic_for_q1=True, AVC1=0.5, AVC2=0.5) fixes Q1
    avalanche physics — helps sub-threshold VG1=0.2.
  * Pseudo-transient backward sweep fixes Newton attractor selection
    via V_B inheritance — helps high-VG1 latch.

Variants (all 33 biases, same loaders as z430/z432/z443):
  A) BASELINE_DC_GP        — DC Newton, GP Q1                (= z430 V_SINT_PIN)
  B) DC_VBIC               — DC Newton, VBIC AVL Q1          (= z443 VBIC_AVL)
  C) PT_BACKWARD_GP        — Pseudo-transient backward, GP   (= z432 backward)
  D) PT_BACKWARD_VBIC      — Pseudo-transient backward + VBIC AVL  (NEW)

Pre-registered gates:
  INFRA       : all 33 biases run for variant D
  DISCOVERY   : COMBINED cell-wide < 0.8 dec (better than best single)
  AMBITIOUS   : COMBINED cell-wide < 0.5 dec
  KILL_SHOT   : COMBINED worse than both PT-only and VBIC-only (destructive)
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
OUT = ROOT / "results/z446_vbic_pt"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)
_spec432 = _ilu.spec_from_file_location("z432", ROOT / "scripts/z432_pseudotransient.py")
z432 = _ilu.module_from_spec(_spec432); _spec432.loader.exec_module(z432)


# Reference numbers (from prior summaries)
Z430_DC_GP_CELL          = 1.6187161900853293   # z430 V_SINT_PIN
Z430_DC_GP_PER_BRANCH    = {"VG1_0.2": 2.6245587058145876,
                            "VG1_0.4": 0.7859912604242465,
                            "VG1_0.6": 1.0855839638811928}
Z443_DC_VBIC_CELL        = 1.3110292027686277   # z443 VBIC_AVL
Z443_DC_VBIC_PER_BRANCH  = {"VG1_0.2": 0.9106606186802844,
                            "VG1_0.4": 1.1351932963274642,
                            "VG1_0.6": 1.5995503114003287}
Z432_PT_BWD_CELL         = 1.026861976331113    # z432 backward
Z432_PT_BWD_PER_BRANCH   = {"VG1_0.2": 1.3534514215128182,
                            "VG1_0.4": 0.5213239064623348,
                            "VG1_0.6": 1.028497244436865}


# ============================================================ #
# Cellwide runner with optional extra_flags (e.g. VBIC) injected into cfg
# ============================================================ #

def run_cellwide_pt(name: str, extra_flags: dict, model_M1, model_M2, curves,
                    sebas_rows, direction: str = "backward"):
    """Pseudo-transient sweep with arbitrary cfg flags (VBIC, etc).

    Mirrors z432.run_cellwide but threads extra_flags into make_cfg.
    """
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(extra_flags))
    log_eps = 1e-15
    per_bias = []
    fails = 0
    t0 = time.time()
    vb_max_overall = -1e30
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Id_pred, Vb_list, conv_list, niter_list = z432.run_one_bias(
                    cfg, model_M1, model_M2, bjt, Vd_arr,
                    c["VG1"], c["VG2"],
                    backward=(direction == "backward"),
                    Vb_init_first=0.0)
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
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
        vb_max = float(max(Vb_list))
        vb_max_overall = max(vb_max_overall, vb_max)
        per_bias.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse,
            "vb_max": vb_max,
            "n_conv": int(conv_t.sum()),
            "n_pts": len(Vd_arr),
            "niter_mean": float(np.mean(niter_list)),
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_pred": list(Id_pred),
            "Vb": Vb_list,
            "converged": conv_list,
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
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"]) for b, v in per_branch.items()}
    total_pts = sum(r["n_pts"] for r in per_bias)
    total_conv = sum(r["n_conv"] for r in per_bias)
    conv_rate = total_conv / max(total_pts, 1)
    log(f"  {name}({direction}): cell={cell:.3f} per_branch="
        f"{ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"Vb_max={vb_max_overall:.3f} conv_rate={conv_rate*100:.1f}% "
        f"fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "name": name,
        "direction": direction,
        "extra_flags": dict(extra_flags),
        "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time() - t0, 1),
        "per_bias": per_bias,
    }


# ============================================================ #
# DC-Newton runners (re-use z443's path, with/without VBIC)
# ============================================================ #

def run_cellwide_dc(name: str, extra_flags: dict, model_M1, model_M2, curves,
                    sebas_rows):
    """DC Newton (V_SINT_PIN) — same as z443.run_vsint_pin_with_flags but
    minimal (no avalanche diagnostics).
    """
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(extra_flags))
    log_eps = 1e-15
    per_bias = []
    fails = 0
    t0 = time.time()
    vb_max_overall = -1e30
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        Id_pred_list = []
        Vb_list = []
        conv_list = []
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for Vd_f in Vd_arr:
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred_list.append(abs(r["Id"]))
                    Vb_list.append(r["Vb"])
                    conv_list.append(bool(r["converged"]))
                    if r["converged"]:
                        Vb_warm = r["Vb"]
                    else:
                        Vb_warm = 0.0
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
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
        per_bias.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse,
            "vb_max": vb_max,
            "n_conv": int(conv.sum()),
            "n_pts": len(Vd_arr),
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_pred": Id_pred.tolist(),
            "Vb": Vb_list,
            "converged": conv_list,
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
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"]) for b, v in per_branch.items()}
    total_pts = sum(r["n_pts"] for r in per_bias)
    total_conv = sum(r["n_conv"] for r in per_bias)
    conv_rate = total_conv / max(total_pts, 1)
    log(f"  {name}: cell={cell:.3f} per_branch="
        f"{ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"Vb_max={vb_max_overall:.3f} conv_rate={conv_rate*100:.1f}% "
        f"fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "name": name,
        "direction": "dc_newton",
        "extra_flags": dict(extra_flags),
        "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time() - t0, 1),
        "per_bias": per_bias,
    }


# ============================================================ #
# Overlay plot: measured + 4 variants
# ============================================================ #

def overlay_plot(VG1_target: float, variants: dict, fname: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    rows_by_vg2: dict[float, dict[str, dict]] = {}
    for name, r in variants.items():
        for rec in r["per_bias"]:
            if abs(rec["VG1"] - VG1_target) < 1e-3:
                rows_by_vg2.setdefault(rec["VG2"], {})[name] = rec
    vg2_vals = sorted(rows_by_vg2.keys())
    if not vg2_vals:
        plt.close(fig); return
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    colors = {
        "BASELINE_DC_GP":     "tab:red",
        "DC_VBIC":            "tab:orange",
        "PT_BACKWARD_GP":     "tab:cyan",
        "PT_BACKWARD_VBIC":   "tab:blue",
    }
    styles = {
        "BASELINE_DC_GP":     "--",
        "DC_VBIC":            "--",
        "PT_BACKWARD_GP":     ":",
        "PT_BACKWARD_VBIC":   "-",
    }
    for ax, vg2 in zip(axes, chosen):
        sub = rows_by_vg2.get(vg2, {})
        meas = next(iter(sub.values()), None)
        if meas is None:
            ax.set_title(f"VG2={vg2:.2f} (no data)"); continue
        ax.plot(meas["Vd"], meas["Id_meas"], "k-", lw=2.5, label="measured")
        for name in ("BASELINE_DC_GP", "DC_VBIC", "PT_BACKWARD_GP",
                     "PT_BACKWARD_VBIC"):
            if name in sub:
                r = sub[name]
                ax.plot(r["Vd"], r["Id_pred"], styles[name], lw=1.5,
                        color=colors[name], label=name)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z446 PT-backward × VBIC stacking @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

def main():
    t_main = time.time()
    log("z446 starting — combined PT-backward + VBIC")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    variants = {}

    log("=== A) BASELINE_DC_GP (DC Newton, GP) ===")
    variants["BASELINE_DC_GP"] = run_cellwide_dc(
        "BASELINE_DC_GP", {}, model_M1, model_M2, curves, sebas_rows)

    log("=== B) DC_VBIC (DC Newton, VBIC AVL Q1) ===")
    variants["DC_VBIC"] = run_cellwide_dc(
        "DC_VBIC",
        {"use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5},
        model_M1, model_M2, curves, sebas_rows)

    log("=== C) PT_BACKWARD_GP (pseudo-transient backward, GP) ===")
    variants["PT_BACKWARD_GP"] = run_cellwide_pt(
        "PT_BACKWARD_GP", {}, model_M1, model_M2, curves, sebas_rows,
        direction="backward")

    log("=== D) PT_BACKWARD_VBIC (pseudo-transient backward + VBIC AVL) ===")
    variants["PT_BACKWARD_VBIC"] = run_cellwide_pt(
        "PT_BACKWARD_VBIC",
        {"use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5},
        model_M1, model_M2, curves, sebas_rows,
        direction="backward")

    # ----- summary -----
    summary = {}
    for name, r in variants.items():
        summary[name] = {
            "cell_rmse_dec": r["cell_rmse_dec"],
            "per_branch_rmse_dec": r["per_branch_rmse_dec"],
            "n_biases_evaluated": r["n_biases_evaluated"],
            "vb_max_overall": r["vb_max_overall"],
            "convergence_rate": r["convergence_rate"],
            "fails": r["fails"],
            "wall_sec": r["wall_sec"],
            "extra_flags": r["extra_flags"],
        }

    cell_A = variants["BASELINE_DC_GP"]["cell_rmse_dec"]
    cell_B = variants["DC_VBIC"]["cell_rmse_dec"]
    cell_C = variants["PT_BACKWARD_GP"]["cell_rmse_dec"]
    cell_D = variants["PT_BACKWARD_VBIC"]["cell_rmse_dec"]
    best_single = min(cell_B, cell_C)

    # ----- additivity check -----
    delta_VBIC = cell_A - cell_B   # improvement from VBIC alone
    delta_PT   = cell_A - cell_C   # improvement from PT alone
    expected_additive = cell_A - (delta_VBIC + delta_PT)
    observed = cell_D
    additivity_gap = observed - expected_additive   # >0 = sub-additive

    summary["REFERENCE_HISTORICAL"] = {
        "z430_DC_GP_cell":   Z430_DC_GP_CELL,
        "z430_DC_GP_per_branch": Z430_DC_GP_PER_BRANCH,
        "z443_DC_VBIC_cell": Z443_DC_VBIC_CELL,
        "z443_DC_VBIC_per_branch": Z443_DC_VBIC_PER_BRANCH,
        "z432_PT_BWD_cell":  Z432_PT_BWD_CELL,
        "z432_PT_BWD_per_branch": Z432_PT_BWD_PER_BRANCH,
    }
    summary["STACKING_ANALYSIS"] = {
        "delta_from_VBIC_alone_dec":   delta_VBIC,
        "delta_from_PT_alone_dec":     delta_PT,
        "delta_from_COMBINED_dec":     cell_A - cell_D,
        "linear_additive_prediction_cell_dec": expected_additive,
        "observed_combined_cell_dec":  cell_D,
        "additivity_gap_dec":          additivity_gap,
        "best_single_cell_dec":        best_single,
        "combined_improves_over_best_single": (best_single - cell_D),
    }

    gates = {
        "INFRA_pass":                variants["PT_BACKWARD_VBIC"]["n_biases_evaluated"] > 0,
        "DISCOVERY_combined_lt_0p8": cell_D < 0.8,
        "AMBITIOUS_combined_lt_0p5": cell_D < 0.5,
        "BEATS_best_single":         cell_D < best_single - 0.05,
        "KILL_SHOT_destructive":     cell_D > best_single + 0.05,
    }
    summary["GATES"] = gates

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("wrote summary.json")

    # ----- overlays at the three target VG1s -----
    for vg1, suf in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, variants, OUT / f"overlay_VG1_{suf}.png")

    # ----- honest analysis -----
    pb = {n: variants[n]["per_branch_rmse_dec"] for n in variants}
    def fmt_branch(b):
        a = pb["BASELINE_DC_GP"].get(b, float("nan"))
        v = pb["DC_VBIC"].get(b, float("nan"))
        p = pb["PT_BACKWARD_GP"].get(b, float("nan"))
        d = pb["PT_BACKWARD_VBIC"].get(b, float("nan"))
        return (f"| {b} | {a:.3f} | {v:.3f} | {p:.3f} | **{d:.3f}** |")

    lines = []
    lines.append("# z446 — Combined PT-backward + VBIC honest analysis\n\n")
    lines.append("## Variants\n")
    lines.append("| name | description | mechanism |\n|---|---|---|\n")
    lines.append("| A) BASELINE_DC_GP | DC Newton + GP Q1 | reference (= z430) |\n")
    lines.append("| B) DC_VBIC | DC Newton + VBIC AVL Q1 | Q1 avalanche physics (= z443) |\n")
    lines.append("| C) PT_BACKWARD_GP | Pseudo-transient backward + GP Q1 | attractor selection (= z432) |\n")
    lines.append("| D) PT_BACKWARD_VBIC | PT backward + VBIC AVL Q1 | **both stacked (this study)** |\n\n")

    lines.append("## Cell-wide log-RMSE (dec)\n\n")
    lines.append("| variant | cell | Δ vs A | n_biases | conv_rate |\n")
    lines.append("|---|---|---|---|---|\n")
    for n in ("BASELINE_DC_GP", "DC_VBIC", "PT_BACKWARD_GP", "PT_BACKWARD_VBIC"):
        r = variants[n]
        d = cell_A - r["cell_rmse_dec"]
        lines.append(f"| {n} | {r['cell_rmse_dec']:.3f} | {d:+.3f} | "
                     f"{r['n_biases_evaluated']} | {r['convergence_rate']*100:.1f}% |\n")
    lines.append("\n")

    lines.append("## Per-VG1 branch RMSE (dec)\n\n")
    lines.append("| branch | A) DC_GP | B) DC_VBIC | C) PT_GP | D) PT_VBIC |\n")
    lines.append("|---|---|---|---|---|\n")
    for b in sorted(set(pb["BASELINE_DC_GP"].keys()) |
                    set(pb["PT_BACKWARD_VBIC"].keys())):
        lines.append(fmt_branch(b) + "\n")
    lines.append("\n")

    lines.append("## Additivity of mechanisms\n\n")
    lines.append(f"- ΔVBIC-alone   = {delta_VBIC:+.3f} dec (A→B)\n")
    lines.append(f"- ΔPT-alone     = {delta_PT:+.3f} dec (A→C)\n")
    lines.append(f"- Linear-additive prediction: cell ≈ {expected_additive:.3f} dec\n")
    lines.append(f"- Observed COMBINED:           cell = {cell_D:.3f} dec\n")
    lines.append(f"- Additivity gap (obs − pred): {additivity_gap:+.3f} dec  "
                 f"({'sub-additive' if additivity_gap > 0.05 else ('super-additive' if additivity_gap < -0.05 else 'roughly additive')})\n")
    lines.append(f"- Best single mechanism: "
                 f"{'B (VBIC)' if cell_B < cell_C else 'C (PT)'} = {best_single:.3f} dec\n")
    lines.append(f"- Combined improves over best single by: {best_single - cell_D:+.3f} dec\n\n")

    lines.append("## Gates\n\n")
    for k, v in gates.items():
        if "KILL" in k:
            lines.append(f"- {k}: {'TRIGGERED' if v else 'no'}\n")
        else:
            lines.append(f"- {k}: {'PASS' if v else 'FAIL'}\n")
    lines.append("\n")

    lines.append("## Honest verdict\n\n")
    if gates["AMBITIOUS_combined_lt_0p5"]:
        lines.append("- AMBITIOUS HIT: combined PT+VBIC closes the cell to < 0.5 dec.\n")
    elif gates["DISCOVERY_combined_lt_0p8"]:
        lines.append("- DISCOVERY: combined < 0.8 dec — mechanisms stack constructively.\n")
    elif gates["BEATS_best_single"]:
        lines.append("- Partial stacking: combined beats best single by >0.05 dec but "
                     "does not reach 0.8.\n")
    elif gates["KILL_SHOT_destructive"]:
        lines.append("- KILL SHOT: combined is WORSE than the best single mechanism by "
                     ">0.05 dec — mechanisms interact destructively.\n")
    else:
        lines.append("- Neutral: combined within ±0.05 dec of best single. Mechanisms "
                     "are not orthogonal at the cell-wide level.\n")

    (OUT / "honest_analysis.md").write_text("".join(lines))
    log("wrote honest_analysis.md")
    log(f"DONE wall={time.time()-t_main:.0f}s  cell_D={cell_D:.3f}")
    LOG.close()


if __name__ == "__main__":
    main()
