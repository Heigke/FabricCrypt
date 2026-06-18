"""z430 — V_Sint=0 PIN tested cell-wide.

S19 (z429) found that pinning V_Sint=0 with 1D Newton on V_B reduced
the gap at VG1=0.6/VG2=0.0/Vd=2.0 from 4.25 → 1.26 dec. z430 tests
that pin across the full Sebas dataset.

Variants:
  BASELINE        — z427 ALL_FLAGS_ON (no pin)
  M2_RS_100       — soft pin via cfg.m2_source_Rs=100 Ω (physical
                    substrate tap)
  V_SINT_PIN      — hard pin: 1D Newton on V_B with Vsint forced to 0
                    (reuses z429's run_vsint_pinned)

Reuses z427_vsint_fix loaders/build/cfg.
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
OUT = ROOT / "results/z430_vsint_pin_cellwide"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# --- reuse z427 + z429 modules
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port.nsram_cell_2T import forward_2t, _residuals  # noqa


# ============================================================ #
# Variant runners
# ============================================================ #

def run_forward(name: str, extra_flags: dict, model_M1, model_M2, curves,
                sebas_rows):
    """Use forward_2t — works for BASELINE and M2_RS_100."""
    return z427.cell_rmse(name, extra_flags, model_M1, model_M2, curves,
                          sebas_rows, collect_traces=True)


def run_vsint_pin(name: str, model_M1, model_M2, curves, sebas_rows):
    """Hard pin: V_Sint=0, 1D Newton on V_B per Vd point (reuses z429)."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
    vsint_max_overall = 0.0  # by construction = 0
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
        Id_pred_list = []
        Vb_list = []
        conv_list = []
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k, Vd_f in enumerate(Vd_arr):
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    # Id reported by run_vsint_pinned uses Ids_M1 magnitude; same
                    # convention as forward_2t's Id_pred.abs() that z427 uses.
                    Id_pred_list.append(abs(r["Id"]))
                    Vb_list.append(r["Vb"])
                    conv_list.append(bool(r["converged"]))
                    # warm-start next Vd with this Vb if converged, else reset
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
               "converged": conv_list}
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
        f"Vb_max={vb_max_overall:.3f} Vsint_max={vsint_max_overall:.3f} "
        f"conv_rate={conv_rate*100:.1f}% fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "name": name, "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "vsint_max_overall": vsint_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "per_bias": per_bias,
    }


def vsint_max_from_traces(per_bias):
    if not per_bias:
        return None
    mx = -1e30
    for r in per_bias:
        if r.get("Vsint") is not None:
            for v in r["Vsint"]:
                if v > mx:
                    mx = v
    return mx if mx > -1e29 else None


# ============================================================ #
# Overlay plots
# ============================================================ #

def overlay_plot(VG1_target: float, results: dict, fname: Path):
    """Overlay all variants at given VG1 across all VG2 sub-curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    # find variants that have per_bias traces
    variants = [(n, r) for n, r in results.items() if r.get("per_bias")]
    # pick three VG2 values at this VG1
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
    colors = {"BASELINE": "tab:red", "M2_RS_100": "tab:orange",
              "V_SINT_PIN": "tab:green"}
    for ax, vg2 in zip(axes, chosen):
        sub = rows_by_vg2.get(vg2, {})
        # measured (from first variant available)
        meas = None
        for name in ("BASELINE", "M2_RS_100", "V_SINT_PIN"):
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
    fig.suptitle(f"z430: V_Sint pin variants vs measured @ VG1={VG1_target:.1f}",
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
    log("z430 starting — V_Sint=0 PIN cell-wide test")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results: dict[str, dict] = {}

    log("=== BASELINE (z427 ALL_FLAGS_ON) ===")
    results["BASELINE"] = run_forward("BASELINE", {}, model_M1, model_M2,
                                       curves, sebas_rows)

    log("=== M2_RS_100 (soft pin via cfg.m2_source_Rs=100 Ω) ===")
    results["M2_RS_100"] = run_forward("M2_RS_100",
                                        {"m2_source_Rs": 100.0},
                                        model_M1, model_M2, curves, sebas_rows)

    log("=== V_SINT_PIN (hard pin, 1D Newton on V_B) ===")
    results["V_SINT_PIN"] = run_vsint_pin("V_SINT_PIN", model_M1, model_M2,
                                           curves, sebas_rows)

    # Summary
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "cell_rmse_dec": r["cell_rmse_dec"],
            "per_branch_rmse_dec": r["per_branch_rmse_dec"],
            "n_biases_evaluated": r["n_biases_evaluated"],
            "vb_max_overall": r.get("vb_max_overall"),
            "vsint_max_overall": r.get("vsint_max_overall",
                                       vsint_max_from_traces(r.get("per_bias"))),
            "convergence_rate": r.get("convergence_rate"),
            "fails": r["fails"],
            "wall_sec": r["wall_sec"],
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote summary.json")

    # Ablation vs z427 baseline (already 3.899)
    Z427_BASELINE = 3.8986888982883516
    Z429_PIN_GAP = 1.26  # single bias VG1=0.6, VG2=0.0, Vd=2.0
    ablation = {
        "z427_baseline_cell_rmse_dec": Z427_BASELINE,
        "z429_pin_gap_dec_single_bias_VG1_0p6_VG2_0p0": Z429_PIN_GAP,
        "z430_results": summary,
        "deltas_vs_z427_baseline": {
            n: Z427_BASELINE - summary[n]["cell_rmse_dec"]
            for n in summary
        },
        "verdict_gates": {
            "INFRA_pass": all(summary[n]["n_biases_evaluated"] > 0 for n in summary),
            "DISCOVERY_pass_lt_2p0": any(summary[n]["cell_rmse_dec"] < 2.0
                                           for n in summary
                                           if n != "BASELINE"),
            "AMBITIOUS_pass_lt_1p0": any(summary[n]["cell_rmse_dec"] < 1.0
                                          for n in summary
                                          if n != "BASELINE"),
            "KILL_SHOT": (summary.get("V_SINT_PIN", {}).get("cell_rmse_dec", 1e9)
                          > 2.5),
        },
    }
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2))
    log(f"wrote ablation.json")

    # Overlays
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, results,
                     OUT / f"overlay_VG1_{suffix}.png")

    # Honest analysis
    honest = ["# z430 — V_Sint=0 PIN cell-wide test\n",
              "## Variants\n",
              "- **BASELINE**: z427 ALL_FLAGS_ON (suppress_bulk_diode_forward + q1_be_oneway + use_mario_ipos), no pin.\n",
              "- **M2_RS_100**: soft pin via `cfg.m2_source_Rs = 100 Ω` — adds substrate shunt I_shunt = V_Sint/100 to R_Sint.\n",
              "- **V_SINT_PIN**: hard pin via z429's `run_vsint_pinned` — 1D Newton on V_B with V_Sint forced to 0, per Vd point, with warm-start across Vd sweep.\n",
              "\n## Results\n",
              "```\n", json.dumps(summary, indent=2), "\n```\n",
              "\n## Ablation\n",
              "```\n", json.dumps(ablation["deltas_vs_z427_baseline"], indent=2), "\n```\n",
              "\n## Per-bias residuals\n"]
    for name, r in results.items():
        honest.append(f"\n### {name} per-bias log-RMSE\n```\n")
        if r.get("per_bias"):
            for rec in r["per_bias"]:
                honest.append(
                    f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f} "
                    f"RMSE={rec['log_rmse']:.3f} dec  Vb_max={rec['vb_max']:.3f}"
                    + (f" conv={rec.get('n_conv','-')}/{rec.get('n_pts','-')}\n"
                       if 'n_conv' in rec else "\n")
                )
        honest.append("```\n")
    # Verdict
    pin_cell = summary.get("V_SINT_PIN", {}).get("cell_rmse_dec", float("nan"))
    rs_cell = summary.get("M2_RS_100", {}).get("cell_rmse_dec", float("nan"))
    base_cell = summary.get("BASELINE", {}).get("cell_rmse_dec", float("nan"))
    honest.append("\n## Verdict\n")
    honest.append(f"- Baseline cell-wide: {base_cell:.3f} dec\n")
    honest.append(f"- M2_RS_100 cell-wide: {rs_cell:.3f} dec\n")
    honest.append(f"- V_SINT_PIN cell-wide: {pin_cell:.3f} dec\n")
    honest.append(f"- z429 single-bias pin gap was 1.26 dec; cell-wide will be different (averages over all biases).\n")
    gates = ablation["verdict_gates"]
    honest.append(f"- INFRA: {'PASS' if gates['INFRA_pass'] else 'FAIL'}\n")
    honest.append(f"- DISCOVERY (< 2.0 dec on PIN or RS): {'PASS' if gates['DISCOVERY_pass_lt_2p0'] else 'FAIL'}\n")
    honest.append(f"- AMBITIOUS (< 1.0 dec on PIN or RS): {'PASS' if gates['AMBITIOUS_pass_lt_1p0'] else 'FAIL'}\n")
    honest.append(f"- KILL_SHOT (PIN > 2.5 dec → physics not just V_Sint runaway): {'TRIGGERED' if gates['KILL_SHOT'] else 'no'}\n")
    # Equivalence
    if not (math.isnan(pin_cell) or math.isnan(rs_cell)):
        diff = abs(pin_cell - rs_cell)
        honest.append(f"\n- Soft Rs vs hard PIN cell-wide gap: |{rs_cell:.3f} - {pin_cell:.3f}| = {diff:.3f} dec\n")
        honest.append(f"  - Equivalent (<0.2 dec): {'yes' if diff < 0.2 else 'no'}\n")
    (OUT / "honest_analysis.md").write_text("".join(honest))
    log(f"wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
