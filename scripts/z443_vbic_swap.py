"""z443 — Track A: swap Gummel-Poon Q1 NPN for VBIC level-4.

Hypothesis: VBIC's built-in avalanche multiplication M(V_BC) is the missing
physics that prevents pyport from following the measured snapback shape
(z430 cell-wide RMSE = 1.62 dec with V_SINT_PIN; GP w/o avalanche).

Per Zhou et al. recipe (BSIM3+VBIC for snapback in ngspice). VBIC adds:
  - Avalanche M(Vbc) Kloosterman-de Graaff
  - Parasitic substrate PNP (inert here)
  - Separated IBEI/IBCI/IBEN/IBCN base currents

Variants this script runs (33 biases per Sebas dataset, V_SINT_PIN solver):
  GP_CONTROL — current Gummel-Poon (== z430 V_SINT_PIN baseline = 1.619 dec)
  VBIC_NO_AVL — VBIC, avalanche disabled (sanity: must match GP_CONTROL)
  VBIC_AVL   — VBIC with literature Si avalanche (AVC1=0.5, AVC2=0.5)

Pre-registered gates:
  INFRA      : VBIC implemented + 33 biases run
  DISCOVERY  : cell-wide < 0.8 dec
  AMBITIOUS  : cell-wide < 0.5 dec
  KILL_SHOT  : VBIC_AVL ≈ GP_CONTROL → avalanche not the missing piece.

Run on ikaros, fp64 CPU, venv. PYTHONUNBUFFERED=1.
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
OUT = ROOT / "results/z443_vbic_swap"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# Reuse upstream modules
_s427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_s427); _s427.loader.exec_module(z427)
_s429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_s429); _s429.loader.exec_module(z429)

from nsram.bsim4_port.vbic import VBICNPN, compute_vbic, _VBIC_DEFAULTS_NOTE  # noqa


def run_vsint_pin_with_flags(name: str, extra_flags: dict, model_M1, model_M2,
                             curves, sebas_rows, collect_traces=True):
    """Adapted from z430.run_vsint_pin — adds extra_flags injection into cfg."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(extra_flags))
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
    vsint_max_overall = 0.0
    fails = 0
    t0 = time.time()
    # Diagnostic: avalanche multiplication traces (across all biases)
    avalanche_diag = []   # one row per bias: (VG1, VG2, Vd, Vbc, M_avc, Iavl)
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
        Vbc_list = []
        M_avc_list = []
        Iavl_list = []
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k, Vd_f in enumerate(Vd_arr):
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred_list.append(abs(r["Id"]))
                    Vb_list.append(r["Vb"])
                    conv_list.append(bool(r["converged"]))
                    # diagnostic: compute Vbc and M_avc post-hoc if VBIC active
                    Vbc_eff = r["Vb"] - float(Vd_f)
                    Vbc_list.append(Vbc_eff)
                    if bool(getattr(cfg, "use_vbic_for_q1", False)):
                        # cached VBIC was built on first dispatch call
                        v = getattr(bjt, "_vbic_cache", None)
                        if v is not None:
                            Vbe_t = torch.tensor([r["Vb"]], dtype=torch.float64)
                            Vbc_t = torch.tensor([Vbc_eff], dtype=torch.float64)
                            vo = compute_vbic(v, Vbe=Vbe_t, Vbc=Vbc_t,
                                              T_K=273.15 + cfg.T_C)
                            M_avc_list.append(float(vo["M_avc"][0]))
                            Iavl_list.append(float(vo["Iavl"][0]))
                        else:
                            M_avc_list.append(1.0); Iavl_list.append(0.0)
                    else:
                        M_avc_list.append(1.0); Iavl_list.append(0.0)
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
        rec = {"VG1": c["VG1"], "VG2": c["VG2"],
               "log_rmse": rmse, "vb_max": vb_max,
               "n_conv": int(conv.sum()), "n_pts": len(Vd_arr),
               "Vd": Vd_arr.tolist(),
               "Id_meas": Id_meas.tolist(),
               "Id_pred": Id_pred.tolist(),
               "Vb": Vb_list,
               "Vsint": [0.0] * len(Vd_arr),
               "Vbc": Vbc_list,
               "M_avc": M_avc_list,
               "Iavl": Iavl_list,
               "converged": conv_list}
        per_bias.append(rec)
        # Track maximum M_avc seen
        if M_avc_list:
            avalanche_diag.append({
                "VG1": c["VG1"], "VG2": c["VG2"],
                "M_avc_max": float(max(M_avc_list)),
                "Iavl_max":  float(max(Iavl_list)),
                "Vbc_min":   float(min(Vbc_list)),
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
    log(f"  {name}: cell={cell:.3f} per_branch={{{', '.join(f'{k}={v:.3f}' for k,v in per_branch_rmse.items())}}} "
        f"Vb_max={vb_max_overall:.3f} conv_rate={conv_rate*100:.1f}% fails={fails} "
        f"wall={time.time()-t0:.0f}s")
    return {
        "name": name, "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "vsint_max_overall": vsint_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "per_bias": per_bias if collect_traces else None,
        "avalanche_diag": avalanche_diag,
    }


# ============================================================ #
# Overlay plot (3-VG2 panels at given VG1) showing GP vs VBIC vs measured
# ============================================================ #

def overlay_plot(VG1_target: float, results: dict, fname: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    variants = [(n, r) for n, r in results.items() if r.get("per_bias")]
    rows_by_vg2: dict[float, dict[str, dict]] = {}
    for name, r in variants:
        for rec in r["per_bias"]:
            if abs(rec["VG1"] - VG1_target) < 1e-3:
                rows_by_vg2.setdefault(rec["VG2"], {})[name] = rec
    vg2_vals = sorted(rows_by_vg2.keys())
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    colors = {"GP_CONTROL": "tab:red", "VBIC_NO_AVL": "tab:orange",
              "VBIC_AVL": "tab:blue"}
    for ax, vg2 in zip(axes, chosen):
        sub = rows_by_vg2.get(vg2, {})
        meas = None
        for name in ("GP_CONTROL", "VBIC_NO_AVL", "VBIC_AVL"):
            if name in sub:
                meas = sub[name]
                break
        if meas is None:
            ax.set_title(f"VG2={vg2:.2f} (no data)")
            continue
        ax.plot(meas["Vd"], meas["Id_meas"], "k-", lw=2.5, label="measured")
        for name, rec in sub.items():
            ax.plot(rec["Vd"], rec["Id_pred"], "--", lw=1.5,
                    color=colors.get(name, "gray"), label=name)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z443 VBIC swap @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def ic_vbc_plot(results: dict, fname: Path):
    """Show |Ic|+Iavl vs Vbc per bias for VBIC_AVL — does avalanche kick in?"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    r = results.get("VBIC_AVL")
    if r is None or not r.get("per_bias"):
        log("  ic_vbc_plot: no VBIC_AVL data")
        return
    cmap = plt.cm.viridis
    pb = r["per_bias"]
    # color by index
    for i, rec in enumerate(pb):
        color = cmap(i / max(len(pb)-1, 1))
        ax.plot(rec["Vbc"], rec["M_avc"], "-",
                color=color, lw=1.0, alpha=0.7,
                label=f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f}"
                      if i % 6 == 0 else None)
    ax.set_xlabel("V_BC = V_B − V_D  [V]  (forward-active = negative)")
    ax.set_ylabel("M_avc (= 1 + Iavl/(It+Ibci))")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title("z443 VBIC avalanche multiplication kicking in (VBIC_AVL)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

def main():
    t_main = time.time()
    log("z443 starting — VBIC swap (Track A)")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results: dict[str, dict] = {}

    log("=== GP_CONTROL (Gummel-Poon — current pyport, == z430 V_SINT_PIN) ===")
    results["GP_CONTROL"] = run_vsint_pin_with_flags(
        "GP_CONTROL", {}, model_M1, model_M2, curves, sebas_rows)

    log("=== VBIC_NO_AVL (VBIC with avalanche disabled — sanity check) ===")
    results["VBIC_NO_AVL"] = run_vsint_pin_with_flags(
        "VBIC_NO_AVL",
        {"use_vbic_for_q1": True, "vbic_AVC1": 0.0, "vbic_AVC2": 0.0},
        model_M1, model_M2, curves, sebas_rows)

    log("=== VBIC_AVL (VBIC with Si default avalanche AVC1=0.5 AVC2=0.5) ===")
    results["VBIC_AVL"] = run_vsint_pin_with_flags(
        "VBIC_AVL",
        {"use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5},
        model_M1, model_M2, curves, sebas_rows)

    # Summary
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "cell_rmse_dec": r["cell_rmse_dec"],
            "per_branch_rmse_dec": r["per_branch_rmse_dec"],
            "n_biases_evaluated": r["n_biases_evaluated"],
            "vb_max_overall": r["vb_max_overall"],
            "convergence_rate": r["convergence_rate"],
            "fails": r["fails"],
            "wall_sec": r["wall_sec"],
        }

    Z430_V_SINT_PIN = 1.6187161900853293
    summary["_reference"] = {
        "z430_V_SINT_PIN_cell_rmse_dec": Z430_V_SINT_PIN,
        "Note": ("GP_CONTROL here uses identical solver/cfg to z430's "
                 "V_SINT_PIN variant; expect ≈ 1.619 dec."),
        "VBIC_defaults_not_from_card": _VBIC_DEFAULTS_NOTE,
    }

    # Pre-registered verdict gates
    gp_cell = summary["GP_CONTROL"]["cell_rmse_dec"]
    sanity = summary["VBIC_NO_AVL"]["cell_rmse_dec"]
    vbic_cell = summary["VBIC_AVL"]["cell_rmse_dec"]
    summary["_verdict"] = {
        "INFRA_pass": all(summary[n]["n_biases_evaluated"] > 0
                          for n in ("GP_CONTROL", "VBIC_NO_AVL", "VBIC_AVL")),
        "DISCOVERY_lt_0p8": vbic_cell < 0.8,
        "AMBITIOUS_lt_0p5": vbic_cell < 0.5,
        "KILL_SHOT_vbic_eq_gp": abs(vbic_cell - gp_cell) < 0.05,
        "SANITY_no_avl_eq_gp": abs(sanity - gp_cell) < 0.10,
        "delta_vbic_vs_gp": gp_cell - vbic_cell,
        "delta_sanity_vs_gp": gp_cell - sanity,
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("wrote summary.json")

    # Overlay plots
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, results, OUT / f"overlay_VG1_{suffix}.png")

    # Avalanche multiplication trace plot
    ic_vbc_plot(results, OUT / "ic_vbc_traces.png")

    # Honest analysis
    lines = ["# z443 — Track A: VBIC swap honest analysis\n\n",
             "## Variants\n",
             "- **GP_CONTROL**: Gummel-Poon (current pyport, == z430 V_SINT_PIN).\n",
             "- **VBIC_NO_AVL**: VBIC w/ avalanche disabled (sanity check).\n",
             "- **VBIC_AVL**: VBIC w/ Si default avalanche (AVC1=0.5, AVC2=0.5).\n",
             "\n## Parameter mapping GP → VBIC\n",
             "```\n",
             "GP.Is  → VBIC.IS         (transport sat)\n",
             "GP.Bf  → VBIC.IBEI = IS/Bf  (ideal B-E)\n",
             "GP.Br  → VBIC.IBCI = IS/Br  (ideal B-C)\n",
             "GP.Nf  → VBIC.NF, NEI    (forward emission)\n",
             "GP.Nr  → VBIC.NR, NCI    (reverse emission)\n",
             "GP.Ne  → VBIC.NEN        (non-ideal B-E emission)\n",
             "GP.Nc  → VBIC.NCN        (non-ideal B-C emission)\n",
             "GP.Ise → VBIC.IBEN       (non-ideal B-E sat)\n",
             "GP.Isc → VBIC.IBCN       (non-ideal B-C sat)\n",
             "GP.Ikf → VBIC.IKF        (forward knee)\n",
             "GP.Ikr → VBIC.IKR        (reverse knee)\n",
             "GP.Va  → VBIC.VEF        (forward Early)\n",
             "GP.Vb  → VBIC.VER        (reverse Early)\n",
             "GP.area→ VBIC.area\n",
             "```\n",
             "\n## VBIC params NOT in Sebas card (using defaults)\n",
             "```\n", json.dumps(_VBIC_DEFAULTS_NOTE, indent=2), "\n```\n",
             "\n## Results\n",
             f"- GP_CONTROL    cell-wide RMSE: **{gp_cell:.3f} dec**\n",
             f"- VBIC_NO_AVL   cell-wide RMSE: **{sanity:.3f} dec**  (sanity)\n",
             f"- VBIC_AVL      cell-wide RMSE: **{vbic_cell:.3f} dec**\n",
             f"- z430 V_SINT_PIN reference: 1.619 dec\n",
             f"- Δ(VBIC_AVL − GP_CONTROL) = {vbic_cell - gp_cell:+.3f} dec\n",
             "\n## Verdict (pre-registered gates)\n",
             f"- INFRA      : {'PASS' if summary['_verdict']['INFRA_pass'] else 'FAIL'}\n",
             f"- DISCOVERY  : {'PASS' if summary['_verdict']['DISCOVERY_lt_0p8'] else 'FAIL'}  (cell < 0.8 dec)\n",
             f"- AMBITIOUS  : {'PASS' if summary['_verdict']['AMBITIOUS_lt_0p5'] else 'FAIL'}  (cell < 0.5 dec)\n",
             f"- KILL_SHOT  : {'TRIGGERED' if summary['_verdict']['KILL_SHOT_vbic_eq_gp'] else 'no'}  "
             f"(|VBIC−GP| < 0.05 dec)\n",
             f"- SANITY     : {'PASS' if summary['_verdict']['SANITY_no_avl_eq_gp'] else 'FAIL'}  "
             f"(VBIC_NO_AVL ≈ GP_CONTROL)\n",
             "\n## Avalanche diagnostic (VBIC_AVL)\n",
             "Per-bias maximum M_avc and Iavl over the V_D sweep:\n",
             "```\n",
    ]
    for d in results["VBIC_AVL"].get("avalanche_diag", []):
        lines.append(
            f"VG1={d['VG1']:.1f} VG2={d['VG2']:+.2f} "
            f"M_avc_max={d['M_avc_max']:.3f} Iavl_max={d['Iavl_max']:.2e} "
            f"Vbc_min={d['Vbc_min']:+.3f}\n"
        )
    lines.append("```\n\n## Interpretation\n")
    # Auto-narrative
    if summary['_verdict']['KILL_SHOT_vbic_eq_gp']:
        lines.append(
            "**KILL_SHOT triggered**: VBIC's avalanche multiplication at "
            "literature Si defaults does NOT close the gap. The residual "
            "0.6 dec (research subagent prediction) is NOT explained by "
            "B-C avalanche. Either (a) AVC1/AVC2 need fitting to 130 nm "
            "process (not allowed by NO-CHEAT), or (b) avalanche is not "
            "the missing physics — look elsewhere (e.g., bulk impact "
            "ionization Iii is already in the model, or self-heating, or "
            "distributed Rb).\n")
    elif summary['_verdict']['DISCOVERY_lt_0p8']:
        lines.append(
            "**DISCOVERY**: VBIC closes the cell-wide gap below 0.8 dec. "
            "The avalanche M(V_BC) was the missing piece. Recommend "
            "promoting `use_vbic_for_q1=True` to default.\n")
    elif summary['_verdict']['delta_vbic_vs_gp'] > 0.1:
        lines.append(
            "**PARTIAL improvement**: VBIC reduces cell-wide RMSE by "
            f"{summary['_verdict']['delta_vbic_vs_gp']:.2f} dec but does "
            "not reach DISCOVERY threshold. Avalanche helps but is not "
            "sufficient.\n")
    else:
        lines.append(
            "**No improvement**: VBIC ≈ GP_CONTROL. Avalanche multiplication "
            "with literature defaults does not move the needle. Same "
            "conclusion as KILL_SHOT in interpretation.\n")
    (OUT / "honest_analysis.md").write_text("".join(lines))
    log("wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
