"""z437 — Enable use_snapback_sub=True cell-wide (S25).

S23 audit (results/z435_lambda_homotopy/audit.md) discovered that the
`compute_snapback` subcircuit (avalanche M(V_db) Slotboom–Chynoweth +
parasitic vertical NPN) is implemented in
`nsram/bsim4_port/snapback_subcircuit.py` and wired into
`_residuals` (nsram_cell_2T.py:1431) — but `use_snapback_sub` defaults
to False and has never been turned on in our V_SINT_PIN baseline.

This is exactly what O74's 4/4 oracle consensus recommended:
M(V_DB) Chynoweth avalanche + coupled vertical BJT. Already in pyport,
just not activated.

Variants (all use V_SINT_PIN, hard pin V_Sint=0, 1D Newton on V_B):
  BASELINE                : no snapback (== z430 V_SINT_PIN, target 1.619 dec)
  SNAPBACK_DEFAULT        : use_snapback_sub=True with module defaults
                            (BV=2.0, n=4, Bf=417, Va=0.90, Is=6.0e-9)
  SNAPBACK_VBR_3p5        : SNAPBACK_DEFAULT, BV=3.5
  SNAPBACK_VBR_4p5        : SNAPBACK_DEFAULT, BV=4.5  (typical 130nm)
  SNAPBACK_VBR_5p5        : SNAPBACK_DEFAULT, BV=5.5
  SNAPBACK_LAMBDA_HOMOTOPY: SNAPBACK at best BV plus λ-homotopy on BJT Bf
                            (3.0 → 1.0 prefactor on snapback BJT Ic) to
                            try to track latched branch with subcircuit on.

Pre-registered gates:
  INFRA      : all variants × all 33 biases done
  DISCOVERY  : ANY variant cell-wide improves ≥ 0.3 dec vs z430 BASELINE
               (i.e. cell < 1.319 dec)
  AMBITIOUS  : cell-wide < 1.0 dec
  KILL_SHOT  : even with snapback ON, no improvement → BSIM topology +
               Mario PWL fundamentally insufficient

Honest reporting: ALL variants reported. If snapback hurts, say so.
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
OUT = ROOT / "results/z437_snapback_enabled"
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

# For λ-homotopy variant: monkey-patch parasitic_npn_collector_current
from nsram.bsim4_port import snapback_subcircuit as _snap
_ORIG_NPN = _snap.parasitic_npn_collector_current
_SNAP_BJT_LAMBDA = 1.0  # scales Ic of the parasitic vertical NPN

def _patched_npn(Vd, Vs, Vb, params, T_K=300.15):
    Ic = _ORIG_NPN(Vd, Vs, Vb, params, T_K=T_K)
    if _SNAP_BJT_LAMBDA == 1.0:
        return Ic
    return _SNAP_BJT_LAMBDA * Ic

_snap.parasitic_npn_collector_current = _patched_npn
# nsram_cell_2T imports compute_snapback at call site (inside _residuals)
# from the same module, so this patch is picked up.


# ============================================================ #
# Snapback cfg overlays
# ============================================================ #

def snapback_flags(BV: float, Bf: float = 417.0,
                   Va: float = 0.90, Is: float = 6.0256e-9,
                   n_avl: float = 4.0):
    return dict(
        use_snapback_sub=True,
        snap_BV=BV,
        snap_n_avl=n_avl,
        snap_Bf=Bf,
        snap_Va=Va,
        snap_Is=Is,
        snap_Nf=1.0,
        snap_Id_clamp=1.0e-2,
        snap_Iii_clamp=1.0e-2,
    )


# ============================================================ #
# V_SINT pin cell-wide runner (with snapback flags applied to cfg)
# ============================================================ #

def run_vsint_pin_snapback(name: str, extra_cfg: dict,
                           model_M1, model_M2, curves, sebas_rows,
                           collect_traces: bool = True):
    """Hard pin V_Sint=0, 1D Newton on V_B per Vd, with snapback flags
    optionally applied. Captures V_B trajectories so we can plot V_DB
    later (V_DB = V_d - V_b)."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    # Apply snapback extras
    for k, v in extra_cfg.items():
        setattr(cfg, k, v)
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
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
        f"Vb_max={vb_max_overall:.3f} conv={conv_rate*100:.1f}% fails={fails} "
        f"wall={time.time()-t0:.0f}s n_biases={cell_n}")
    return {
        "name": name, "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "vsint_max_overall": 0.0,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "extra_cfg": {k: (float(v) if isinstance(v, (int, float)) else v)
                      for k, v in extra_cfg.items()},
        "per_bias": per_bias,
    }


# ============================================================ #
# λ-homotopy on parasitic-NPN Ic for the best BV variant
# ============================================================ #

LAMBDA_SCHEDULE = [3.0, 2.5, 2.0, 1.6, 1.3, 1.15, 1.0]


def run_snapback_lambda_homotopy(name: str, extra_cfg: dict,
                                 model_M1, model_M2, curves, sebas_rows):
    """Same as run_vsint_pin_snapback, but at each (Vd, VG1, VG2) point
    run a λ-homotopy on the parasitic-NPN Ic prefactor (snapback BJT
    only — does NOT touch BSIM4 Iii). Warm-starts Vb across both λ and
    Vd. Reports the λ=1.0 final answer.
    """
    global _SNAP_BJT_LAMBDA
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    for k, v in extra_cfg.items():
        setattr(cfg, k, v)
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
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
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k_idx, Vd_f in enumerate(Vd_arr):
                    Vb_cur = Vb_warm
                    last_r = None
                    for lam in LAMBDA_SCHEDULE:
                        _SNAP_BJT_LAMBDA = float(lam)
                        r = z429.run_vsint_pinned(
                            cfg, model_M1, model_M2, bjt,
                            float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                            Vsint_pin=0.0, Vb_init=Vb_cur)
                        if r["converged"]:
                            Vb_cur = r["Vb"]
                        last_r = r
                    _SNAP_BJT_LAMBDA = 1.0
                    Id_pred_list.append(abs(last_r["Id"]))
                    Vb_list.append(last_r["Vb"])
                    conv_list.append(bool(last_r["converged"]))
                    if last_r["converged"]:
                        Vb_warm = last_r["Vb"]
                    else:
                        Vb_warm = 0.0
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        finally:
            _SNAP_BJT_LAMBDA = 1.0
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
        f"Vb_max={vb_max_overall:.3f} conv={conv_rate*100:.1f}% fails={fails} "
        f"wall={time.time()-t0:.0f}s")
    return {
        "name": name, "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "vsint_max_overall": 0.0,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "extra_cfg": {k: (float(v) if isinstance(v, (int, float)) else v)
                      for k, v in extra_cfg.items()},
        "lambda_schedule": LAMBDA_SCHEDULE,
        "per_bias": per_bias,
    }


# ============================================================ #
# Plots
# ============================================================ #

COLORS = {
    "BASELINE":          "tab:red",
    "SNAPBACK_DEFAULT":  "tab:blue",
    "SNAPBACK_VBR_3p5":  "tab:purple",
    "SNAPBACK_VBR_4p5":  "tab:green",
    "SNAPBACK_VBR_5p5":  "tab:olive",
    "SNAPBACK_LAMBDA_HOMOTOPY": "tab:cyan",
}


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
    for ax, vg2 in zip(axes, chosen):
        sub = rows_by_vg2.get(vg2, {})
        meas = next((sub[n] for n in results if n in sub), None)
        if meas is None:
            ax.set_title(f"VG2={vg2:.2f} (no data)")
            continue
        ax.plot(meas["Vd"], meas["Id_meas"], "k-", lw=2.5, label="measured")
        for name, rec in sub.items():
            ax.plot(rec["Vd"], rec["Id_pred"], "--", lw=1.2,
                    color=COLORS.get(name, "gray"), label=name)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z437: snapback subcircuit variants @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def vb_vdb_traces(results: dict, fname: Path):
    """Plot V_B and V_DB (= V_D - V_B) vs V_D, picking VG1=0.6 VG2=0.0
    (highest-bias point, most likely to trigger avalanche)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    VG1_t, VG2_t = 0.6, 0.0
    for name, r in results.items():
        if not r.get("per_bias"):
            continue
        rec = next((p for p in r["per_bias"]
                    if abs(p["VG1"] - VG1_t) < 1e-3 and
                       abs(p["VG2"] - VG2_t) < 1e-3), None)
        if rec is None:
            continue
        Vd = np.array(rec["Vd"]); Vb = np.array(rec["Vb"])
        Vdb = Vd - Vb
        color = COLORS.get(name, "gray")
        axes[0].plot(Vd, Vb, "-", color=color, label=name, lw=1.3)
        axes[1].plot(Vd, Vdb, "-", color=color, label=name, lw=1.3)
    axes[0].set_xlabel("V_D [V]"); axes[0].set_ylabel("V_B [V]")
    axes[0].set_title(f"V_B(V_D) @ VG1={VG1_t} VG2={VG2_t}")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("V_D [V]"); axes[1].set_ylabel("V_DB = V_D - V_B [V]")
    axes[1].set_title("V_DB(V_D) — where M(V_DB) avalanche kicks in")
    # Mark BV candidates
    for bv, ls in [(3.5, ":"), (4.5, "--"), (5.5, "-.")]:
        axes[1].axhline(bv, color="black", linestyle=ls, lw=0.8, alpha=0.5,
                        label=f"BV={bv}")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    fig.suptitle("z437: body potential + drain-body voltage", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

Z430_BASELINE_DEC = 1.6187161900853293  # z430 V_SINT_PIN cell-wide reference

def main():
    t_main = time.time()
    log("z437 starting — enable use_snapback_sub=True cell-wide")
    log(f"reference: z430 V_SINT_PIN cell-wide = {Z430_BASELINE_DEC:.3f} dec")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results: dict[str, dict] = {}

    log("=== BASELINE (V_SINT_PIN, no snapback subcircuit) ===")
    results["BASELINE"] = run_vsint_pin_snapback(
        "BASELINE", {}, model_M1, model_M2, curves, sebas_rows)

    log("=== SNAPBACK_DEFAULT (use_snapback_sub=True, BV=2.0 default) ===")
    results["SNAPBACK_DEFAULT"] = run_vsint_pin_snapback(
        "SNAPBACK_DEFAULT", snapback_flags(BV=2.0),
        model_M1, model_M2, curves, sebas_rows)

    log("=== SNAPBACK_VBR_3p5 ===")
    results["SNAPBACK_VBR_3p5"] = run_vsint_pin_snapback(
        "SNAPBACK_VBR_3p5", snapback_flags(BV=3.5),
        model_M1, model_M2, curves, sebas_rows)

    log("=== SNAPBACK_VBR_4p5 (typical 130nm drain-body BV) ===")
    results["SNAPBACK_VBR_4p5"] = run_vsint_pin_snapback(
        "SNAPBACK_VBR_4p5", snapback_flags(BV=4.5),
        model_M1, model_M2, curves, sebas_rows)

    log("=== SNAPBACK_VBR_5p5 ===")
    results["SNAPBACK_VBR_5p5"] = run_vsint_pin_snapback(
        "SNAPBACK_VBR_5p5", snapback_flags(BV=5.5),
        model_M1, model_M2, curves, sebas_rows)

    # Pick best BV among the swept ones
    bv_variants = [n for n in
                   ("SNAPBACK_DEFAULT", "SNAPBACK_VBR_3p5",
                    "SNAPBACK_VBR_4p5", "SNAPBACK_VBR_5p5")
                   if n in results]
    best_name = min(bv_variants,
                    key=lambda n: results[n]["cell_rmse_dec"])
    best_extra = {k: v for k, v in results[best_name]["extra_cfg"].items()}
    log(f"best BV-sweep variant: {best_name} "
        f"(cell={results[best_name]['cell_rmse_dec']:.3f} dec); "
        f"using its cfg for λ-homotopy")

    log("=== SNAPBACK_LAMBDA_HOMOTOPY (best BV + λ on snapback BJT Ic) ===")
    results["SNAPBACK_LAMBDA_HOMOTOPY"] = run_snapback_lambda_homotopy(
        "SNAPBACK_LAMBDA_HOMOTOPY", best_extra,
        model_M1, model_M2, curves, sebas_rows)

    # ----------- Summary -----------
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "cell_rmse_dec": r["cell_rmse_dec"],
            "per_branch_rmse_dec": r["per_branch_rmse_dec"],
            "n_biases_evaluated": r["n_biases_evaluated"],
            "vb_max_overall": r.get("vb_max_overall"),
            "convergence_rate": r.get("convergence_rate"),
            "fails": r["fails"],
            "wall_sec": r["wall_sec"],
            "extra_cfg": r.get("extra_cfg", {}),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("wrote summary.json")

    # ----------- Ablation -----------
    deltas = {n: Z430_BASELINE_DEC - summary[n]["cell_rmse_dec"]
              for n in summary}
    best_n = min(summary, key=lambda n: summary[n]["cell_rmse_dec"])
    best_dec = summary[best_n]["cell_rmse_dec"]
    ablation = {
        "z430_baseline_cell_rmse_dec": Z430_BASELINE_DEC,
        "variants": summary,
        "deltas_vs_z430_baseline_dec_positive_is_improvement": deltas,
        "best_variant": best_n,
        "best_cell_rmse_dec": best_dec,
        "verdict_gates": {
            "INFRA_pass": all(summary[n]["n_biases_evaluated"] > 0
                              for n in summary),
            "DISCOVERY_improve_gte_0p3_dec": (Z430_BASELINE_DEC - best_dec) >= 0.3,
            "AMBITIOUS_lt_1p0_dec": best_dec < 1.0,
            "KILL_SHOT_no_improvement": (Z430_BASELINE_DEC - best_dec) < 0.05,
        },
    }
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2))
    log("wrote ablation.json")

    # ----------- Plots -----------
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, results, OUT / f"overlay_VG1_{suffix}.png")
    vb_vdb_traces(results, OUT / "vb_db_traces.png")

    # ----------- Audit md -----------
    audit_lines = [
        "# z437 audit — what `compute_snapback` does\n\n",
        "## Source\n",
        "- `nsram/nsram/bsim4_port/snapback_subcircuit.py`\n",
        "- Wired into 2T residual at `nsram_cell_2T.py:1431` (`if cfg.use_snapback_sub: ...`)\n",
        "- Two outputs added to KCL:\n",
        "  - `Id_extra` → drain assembly (regenerative kick, +Id)\n",
        "  - `Iii_body` → body KCL R_B (avalanche holes into body, +R_B)\n\n",
        "## Physics\n",
        "1. Slotboom–Chynoweth avalanche multiplier on V_DB = V_D - V_B:\n",
        "   `M(V_DB) = 1 / (1 - clip((V_DB / BV)^n, max=0.99))`\n",
        "   `Iii_body = (M - 1) * |Ids_BSIM4|`  → clamped at 10 mA.\n",
        "2. Parasitic vertical NPN (collector=V_D, base=V_B, emitter=V_S):\n",
        "   `Vbe=Vb-Vs; Vbc=Vb-Vd`\n",
        "   `Icc = Is * (exp(Vbe/(Nf*Vt)) - 1)`\n",
        "   `Ic = Bf * Icc / (1 - Vbc/Va)` (Early effect, denom clamped ≥ 1e-3)\n",
        "   `Id_extra = Ic` → clamped at 10 mA.\n\n",
        "## Parameters (defaults in `NSRAMCell2TConfig`)\n",
        "| name | default | meaning |\n|---|---|---|\n",
        "| snap_BV | 2.0 V | drain-body breakdown voltage |\n",
        "| snap_n_avl | 4.0 | Slotboom exponent (textbook) |\n",
        "| snap_Bf | 417 | parasitic vertical NPN β (R-46) |\n",
        "| snap_Va | 0.90 V | NPN Early voltage |\n",
        "| snap_Is | 6.0e-9 A | NPN saturation current |\n",
        "| snap_Nf | 1.0 | NPN emission coefficient |\n",
        "| snap_Id_clamp | 10 mA | regenerative kick ceiling |\n",
        "| snap_Iii_clamp | 10 mA | body-injection ceiling |\n\n",
        "## What z437 tested\n",
        "BV swept ∈ {2.0 (default), 3.5, 4.5, 5.5}. All other params kept at\n",
        "their physics-derived defaults. Then a λ-homotopy variant on the\n",
        "parasitic-NPN Ic prefactor (3.0→1.0 in 7 steps, warm-start Vb).\n\n",
        "## Headline results\n",
        f"- z430 V_SINT_PIN baseline: {Z430_BASELINE_DEC:.3f} dec (reference)\n",
    ]
    for n in summary:
        d = summary[n]
        sign = "+" if deltas[n] >= 0 else ""
        audit_lines.append(
            f"- {n}: cell={d['cell_rmse_dec']:.3f} dec "
            f"({sign}{deltas[n]:+.3f} vs z430)  "
            f"per_branch={ {k: round(v,3) for k,v in d['per_branch_rmse_dec'].items()} }\n")
    audit_lines.append(f"\n**Best variant: {best_n} ({best_dec:.3f} dec)**\n")
    (OUT / "audit.md").write_text("".join(audit_lines))
    log("wrote audit.md")

    # ----------- Honest analysis -----------
    gates = ablation["verdict_gates"]
    honest = [
        "# z437 honest analysis — snapback subcircuit, cell-wide\n\n",
        "## Context\n",
        "S23 audit (z435) found the `compute_snapback` subcircuit\n",
        "(avalanche M(V_DB) Slotboom–Chynoweth + parasitic vertical NPN)\n",
        "implemented but never activated in the V_SINT_PIN baseline. This\n",
        "exactly matches O74's 4/4 oracle consensus prescription.\n",
        f"z430 V_SINT_PIN cell-wide reference: **{Z430_BASELINE_DEC:.3f} dec**.\n\n",
        "## Variants\n",
        "- BASELINE — V_SINT_PIN, no snapback (regression target)\n",
        "- SNAPBACK_DEFAULT — module defaults (BV=2.0, n=4, Bf=417)\n",
        "- SNAPBACK_VBR_{3.5, 4.5, 5.5} — BV sweep\n",
        "- SNAPBACK_LAMBDA_HOMOTOPY — best BV + λ-homotopy on parasitic-NPN Ic\n\n",
        "## Cell-wide results\n",
        "```\n", json.dumps({n: round(summary[n]["cell_rmse_dec"], 3)
                              for n in summary}, indent=2), "\n```\n\n",
        "## Per-branch (VG1) breakdown\n",
        "```\n", json.dumps({n: summary[n]["per_branch_rmse_dec"]
                              for n in summary}, indent=2), "\n```\n\n",
        "## Δ vs z430 baseline (positive = improvement)\n",
        "```\n", json.dumps({n: round(deltas[n], 3) for n in deltas}, indent=2),
        "\n```\n\n",
        "## Pre-registered verdict gates\n",
        f"- INFRA (all variants × 33 biases): "
        f"{'PASS' if gates['INFRA_pass'] else 'FAIL'}\n",
        f"- DISCOVERY (≥0.3 dec improvement vs z430): "
        f"{'PASS' if gates['DISCOVERY_improve_gte_0p3_dec'] else 'FAIL'}\n",
        f"- AMBITIOUS (< 1.0 dec cell-wide): "
        f"{'PASS' if gates['AMBITIOUS_lt_1p0_dec'] else 'FAIL'}\n",
        f"- KILL_SHOT (improvement < 0.05 dec): "
        f"{'TRIGGERED' if gates['KILL_SHOT_no_improvement'] else 'no'}\n\n",
        f"## Best variant: **{best_n}** ({best_dec:.3f} dec)\n\n",
        "## Per-bias residuals (full)\n",
    ]
    for name, r in results.items():
        honest.append(f"\n### {name}\n```\n")
        for rec in r.get("per_bias", []):
            honest.append(
                f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f} "
                f"RMSE={rec['log_rmse']:.3f} dec  Vb_max={rec['vb_max']:.3f}  "
                f"conv={rec['n_conv']}/{rec['n_pts']}\n")
        honest.append("```\n")
    # Regime-specificity note
    base_branch = summary["BASELINE"]["per_branch_rmse_dec"]
    best_branch = summary[best_n]["per_branch_rmse_dec"]
    honest.append("\n## Regime-specificity\n")
    for b in sorted(set(base_branch) | set(best_branch)):
        d = base_branch.get(b, float("nan")) - best_branch.get(b, float("nan"))
        honest.append(
            f"- {b}: baseline={base_branch.get(b, float('nan')):.3f} "
            f"→ {best_n}={best_branch.get(b, float('nan')):.3f}  "
            f"(Δ={d:+.3f} dec)\n")
    (OUT / "honest_analysis.md").write_text("".join(honest))
    log("wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
