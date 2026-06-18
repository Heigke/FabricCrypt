"""z431 — BSIM4 v4.8.3 §6.2 GIDL ablation, cell-wide on V_SINT_PIN.

Audit conclusion (see results/z431_bsim4_gidl/audit.md):
  - BSIM4 v4.8.3 GIDL/GISL is ALREADY implemented (leak.py + nsram_cell_2T.py).
  - PTM130 card already has agidl=1.99e-8.
  - `forward_2t`/`solve_2t_steady_state` already adds Igidl_M1 to the drain
    current sum (line 1935).
  - BUT z430's V_SINT_PIN path reports `Id = Ids_M1` only, dropping Igidl,
    Ic_Q1, Ibd_M1, lateral/avalanche/snap contributions. This explains the
    2.63 dec VG1=0.2 stall.

z431 fixes the reported Id and runs the GIDL_ON vs GIDL_OFF ablation cell-wide.

Variants on the V_SINT_PIN solver (z429.run_vsint_pinned, modified locally):
  - GIDL_OFF (sd.scaled.agidl = sd.scaled.agisl = 0)
  - GIDL_ON  (Sebas card values: M1/M2 agidl=1.99e-8 bgidl=1.624e9 cgidl=6.3 egidl=0.91)
Both use the FULL drain-pin sum:
    Id = Ids_M1 + Ic_Q1 + Ic_lat + Ic_avalanche + Igidl_M1 - Ibd_M1 - Ie_vert + I_snap_d

Gates:
  INFRA      : 33 biases × 2 variants run, plots written.
  DISCOVERY  : VG1=0.2 (GIDL_ON) < 1.5 dec  AND  VG1=0.4 stays < 1.0  AND  VG1=0.6 stays < 1.5.
  AMBITIOUS  : cell-wide (GIDL_ON) < 1.0 dec without breaking snapback.
  KILL_SHOT  : GIDL fails to help VG1=0.2  OR  breaks VG1=0.4/0.6 by > 0.3 dec.
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
OUT = ROOT / "results/z431_bsim4_gidl"
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

from nsram.bsim4_port.nsram_cell_2T import _residuals  # noqa


# ============================================================ #
# Patched V_SINT_PIN solver with FULL drain-pin sum
# ============================================================ #

def resid_and_comp(cfg, model_M1, model_M2, bjt, Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f):
    """Like z429.resid_pair but returns full comp dict for drain-pin reconstruction."""
    Vd = torch.tensor([Vd_f], dtype=torch.float64)
    VG1 = torch.tensor([VG1_f], dtype=torch.float64)
    VG2 = torch.tensor([VG2_f], dtype=torch.float64)
    Vsint = torch.tensor([Vsint_f], dtype=torch.float64)
    Vb = torch.tensor([Vb_f], dtype=torch.float64)
    with torch.no_grad():
        R_S, R_B, comp = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                    Vsint, Vb, None, None, model_M2=model_M2)
    return float(R_S.item()), float(R_B.item()), comp


def drain_pin_current(comp, *, gidl_on: bool):
    """Mirror solve_2t_steady_state's drain-pin sum (line 1929-1943)."""
    def g(k, default=0.0):
        v = comp.get(k, default)
        if isinstance(v, torch.Tensor):
            return float(v.item())
        return float(v)
    Id = (
        g("Ids_M1")
        + g("Ic_Q1")
        + g("Ic_Q2")
        + g("Ic_lat")
        + g("Ic_avalanche")
        + (g("Igidl_M1") if gidl_on else 0.0)
        - g("Ibd_M1")
        - g("Ie_vert")
        + g("I_snap_d")
    )
    igidl = g("Igidl_M1") if gidl_on else 0.0
    return Id, igidl


def run_vsint_pinned_drainsum(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                              *, gidl_on: bool, Vsint_pin=0.0, Vb_init=0.0):
    """Same Newton as z429.run_vsint_pinned, but returns Id from the full drain-pin
    sum and exposes the Igidl_M1 contribution for isolation plotting."""
    Vb = Vb_init
    for it in range(80):
        R_S, R_B, _ = resid_and_comp(cfg, model_M1, model_M2, bjt,
                                      Vsint_pin, Vb, Vd_f, VG1_f, VG2_f)
        eps = 1e-5
        _, R_Bp, _ = resid_and_comp(cfg, model_M1, model_M2, bjt,
                                      Vsint_pin, Vb + eps, Vd_f, VG1_f, VG2_f)
        dRdV = (R_Bp - R_B) / eps
        if abs(dRdV) < 1e-30:
            break
        dV = -R_B / dRdV
        if abs(dV) > 0.2:
            dV = math.copysign(0.2, dV)
        Vb_new = Vb + dV
        Vb_new = max(-0.2, min(1.0, Vb_new))
        if abs(Vb_new - Vb) < 1e-10:
            Vb = Vb_new
            break
        Vb = Vb_new
    R_S, R_B, comp = resid_and_comp(cfg, model_M1, model_M2, bjt,
                                     Vsint_pin, Vb, Vd_f, VG1_f, VG2_f)
    Id, Igidl_part = drain_pin_current(comp, gidl_on=gidl_on)
    return dict(Vb=Vb, Vsint=Vsint_pin, Id=Id, Igidl=Igidl_part,
                resid_RB=abs(R_B), resid_RS=abs(R_S),
                converged=(abs(R_B) < 1e-8))


# ============================================================ #
# GIDL ablation override
# ============================================================ #

def make_gidl_off_overrides():
    """Return (P_M1_off, P_M2_off) that, when MERGED into the Sebas overrides,
    zero out agidl/agisl/bgidl/bgisl/cgidl/cgisl on both transistors."""
    zero = torch.tensor(0.0, dtype=torch.float64)
    big = torch.tensor(1e30, dtype=torch.float64)  # bgidl→∞ makes exp(-b/T1)→0
    # We set agidl=agisl=0; the _gidl_one_side kernel hard-zeros on a<=0.
    return ({"agidl": zero, "agisl": zero},
            {"agidl": zero, "agisl": zero})


def merge_overrides(P_main, P_extra):
    out = dict(P_main) if P_main else {}
    if P_extra:
        out.update(P_extra)
    return out


# ============================================================ #
# Run one variant
# ============================================================ #

def run_variant(name, *, gidl_on: bool, model_M1, model_M2, curves, sebas_rows,
                cfg, sd_M1, sd_M2):
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
    fails = 0
    t0 = time.time()
    P_off_M1, P_off_M2 = make_gidl_off_overrides()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        if not gidl_on:
            P_M1 = merge_overrides(P_M1, P_off_M1)
            P_M2 = merge_overrides(P_M2, P_off_M2)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        Id_pred_list = []
        Igidl_list = []
        Vb_list = []
        conv_list = []
        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k, Vd_f in enumerate(Vd_arr):
                    r = run_vsint_pinned_drainsum(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        gidl_on=gidl_on, Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred_list.append(abs(r["Id"]))
                    Igidl_list.append(r["Igidl"])
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
               "Igidl": Igidl_list,
               "Vb": Vb_list,
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
        f"Vb_max={vb_max_overall:.3f}  conv_rate={conv_rate*100:.1f}% fails={fails} "
        f"wall={time.time()-t0:.0f}s n_biases={cell_n}")
    return {
        "name": name,
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
# Plots
# ============================================================ #

def overlay_plot(VG1_target: float, results: dict, fname: Path):
    """Overlay GIDL_ON vs GIDL_OFF on the same VG1 cut, across up to 3 VG2 values."""
    # Get all VG2 values present at this VG1 across variants
    vg2_set = set()
    for r in results.values():
        for rec in r.get("per_bias", []):
            if abs(rec["VG1"] - VG1_target) < 1e-3:
                vg2_set.add(round(rec["VG2"], 3))
    vg2_list = sorted(vg2_set)[:3]
    n = len(vg2_list)
    if n == 0:
        log(f"  overlay VG1={VG1_target}: no biases found")
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True, squeeze=False)
    axes = axes[0]
    colors = {"GIDL_ON": "tab:red", "GIDL_OFF": "tab:blue"}
    for ax, vg2 in zip(axes, vg2_list):
        plotted_meas = False
        for name, r in results.items():
            for rec in r.get("per_bias", []):
                if abs(rec["VG1"] - VG1_target) < 1e-3 and abs(rec["VG2"] - vg2) < 1e-3:
                    if not plotted_meas:
                        ax.semilogy(rec["Vd"],
                                    np.maximum(np.array(rec["Id_meas"]), 1e-15),
                                    "k.-", lw=1.4, ms=4, label="meas")
                        plotted_meas = True
                    ax.semilogy(rec["Vd"],
                                np.maximum(np.array(rec["Id_pred"]), 1e-15),
                                color=colors.get(name, "gray"), lw=1.2,
                                label=f"{name} (rmse={rec['log_rmse']:.2f})")
                    break
        ax.set_xlabel("V_D (V)")
        ax.set_title(f"VG1={VG1_target}, VG2={vg2:+.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|Id| (A)")
    fig.suptitle(f"z431 GIDL overlay at VG1={VG1_target}", y=1.02)
    fig.tight_layout()
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {fname.name}")


def gidl_isolation_plot(results: dict, fname: Path):
    """Log-plot of Igidl_M1 contribution alone vs V_D at VG1=0.2 for GIDL_ON variant."""
    r_on = results.get("GIDL_ON")
    if r_on is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for rec in r_on["per_bias"]:
        if abs(rec["VG1"] - 0.2) < 1e-3:
            igidl = np.array(rec["Igidl"])
            ids_only = np.array(rec["Id_pred"]) - igidl  # rest of drain-pin sum
            vd = np.array(rec["Vd"])
            mask = igidl > 0
            if mask.any():
                ax.semilogy(vd[mask], igidl[mask], "-",
                            label=f"VG2={rec['VG2']:+.2f} Igidl")
                plotted = True
            ax.semilogy(vd, np.maximum(np.abs(ids_only), 1e-18), "--", alpha=0.4,
                        label=f"VG2={rec['VG2']:+.2f} non-GIDL")
    if not plotted:
        ax.text(0.5, 0.5, "Igidl_M1 ≤ 0 across VG1=0.2", ha="center", transform=ax.transAxes)
    ax.set_xlabel("V_D (V)")
    ax.set_ylabel("|I| (A)")
    ax.set_title("z431 GIDL isolation: Igidl_M1 vs non-GIDL drain-pin sum, VG1=0.2")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

def main():
    t_main = time.time()
    log("z431 starting — BSIM4 v4.8.3 §6.2 GIDL ablation, cell-wide on V_SINT_PIN")
    log("Audit: GIDL is already implemented (leak.py compute_igidl_gisl, b4ld.c §2274-2370).")
    log("       PTM130 has agidl=1.99e-8; Sebas M1/M2 cards have full agidl/bgidl/cgidl/egidl.")
    log("       z430 V_SINT_PIN reported Id=Ids_M1 only, dropping Igidl_M1 from drain-pin sum.")
    log("       z431 reports full drain-pin sum + ablates agidl=agisl=0 in GIDL_OFF.")

    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})

    results: dict[str, dict] = {}

    log("=== GIDL_OFF (agidl=agisl=0 on M1 & M2) ===")
    results["GIDL_OFF"] = run_variant("GIDL_OFF", gidl_on=False,
                                       model_M1=model_M1, model_M2=model_M2,
                                       curves=curves, sebas_rows=sebas_rows,
                                       cfg=cfg, sd_M1=sd_M1, sd_M2=sd_M2)

    log("=== GIDL_ON (Sebas card values) ===")
    results["GIDL_ON"] = run_variant("GIDL_ON", gidl_on=True,
                                      model_M1=model_M1, model_M2=model_M2,
                                      curves=curves, sebas_rows=sebas_rows,
                                      cfg=cfg, sd_M1=sd_M1, sd_M2=sd_M2)

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
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote summary.json")

    # Ablation
    Z430_VSINT_PIN = {
        "cell": 1.619,
        "VG1_0.2": 2.625,
        "VG1_0.4": 0.786,
        "VG1_0.6": 1.086,
    }
    on  = summary["GIDL_ON"]
    off = summary["GIDL_OFF"]
    pb_on  = on["per_branch_rmse_dec"]
    pb_off = off["per_branch_rmse_dec"]

    def get(d, k): return d.get(k, float("nan"))

    snapback_check = {
        "VG1_0.4_on":  get(pb_on, "VG1_0.4"),
        "VG1_0.6_on":  get(pb_on, "VG1_0.6"),
        "VG1_0.4_off": get(pb_off, "VG1_0.4"),
        "VG1_0.6_off": get(pb_off, "VG1_0.6"),
        "VG1_0.4_delta_on_vs_off": get(pb_on, "VG1_0.4") - get(pb_off, "VG1_0.4"),
        "VG1_0.6_delta_on_vs_off": get(pb_on, "VG1_0.6") - get(pb_off, "VG1_0.6"),
        "snapback_preserved_VG1_0p4": (get(pb_on, "VG1_0.4") < 1.0
                                       and (get(pb_on, "VG1_0.4") - get(pb_off, "VG1_0.4")) < 0.3),
        "snapback_preserved_VG1_0p6": (get(pb_on, "VG1_0.6") < 1.5
                                       and (get(pb_on, "VG1_0.6") - get(pb_off, "VG1_0.6")) < 0.3),
    }
    gates = {
        "INFRA_pass": (on["n_biases_evaluated"] > 0
                       and off["n_biases_evaluated"] > 0),
        "DISCOVERY_pass":
            (get(pb_on, "VG1_0.2") < 1.5
             and snapback_check["snapback_preserved_VG1_0p4"]
             and snapback_check["snapback_preserved_VG1_0p6"]),
        "AMBITIOUS_pass":
            (on["cell_rmse_dec"] < 1.0
             and snapback_check["snapback_preserved_VG1_0p4"]
             and snapback_check["snapback_preserved_VG1_0p6"]),
        "KILL_SHOT":
            (get(pb_on, "VG1_0.2") >= get(pb_off, "VG1_0.2") - 0.1
             or (not snapback_check["snapback_preserved_VG1_0p4"])
             or (not snapback_check["snapback_preserved_VG1_0p6"])),
    }
    ablation = {
        "z430_vsint_pin_reference": Z430_VSINT_PIN,
        "z431_GIDL_OFF": {
            "cell_rmse_dec": off["cell_rmse_dec"],
            "per_branch_rmse_dec": pb_off,
        },
        "z431_GIDL_ON": {
            "cell_rmse_dec": on["cell_rmse_dec"],
            "per_branch_rmse_dec": pb_on,
        },
        "deltas_on_minus_off": {
            "cell": on["cell_rmse_dec"] - off["cell_rmse_dec"],
            "VG1_0.2": get(pb_on, "VG1_0.2") - get(pb_off, "VG1_0.2"),
            "VG1_0.4": get(pb_on, "VG1_0.4") - get(pb_off, "VG1_0.4"),
            "VG1_0.6": get(pb_on, "VG1_0.6") - get(pb_off, "VG1_0.6"),
        },
        "snapback_check": snapback_check,
        "gates": gates,
    }
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2))
    log(f"wrote ablation.json")

    # Plots
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, results, OUT / f"overlay_VG1_{suffix}.png")
    gidl_isolation_plot(results, OUT / "gidl_isolation.png")

    # Honest analysis
    lines = ["# z431 — BSIM4 v4.8.3 §6.2 GIDL ablation (V_SINT_PIN, cell-wide)\n",
             "## Audit\n",
             "GIDL/GISL was already implemented in `nsram/bsim4_port/leak.py` via\n",
             "`compute_igidl_gisl` (b4ld.c §2274-2370, manual §6.2, gidlMod=0).\n",
             "PTM130 already has `agidl=1.99e-8`. The only thing actually missing was\n",
             "that `z429.run_vsint_pinned` reported `Id = Ids_M1` and dropped Igidl_M1\n",
             "(and Ic_Q1, Ibd_M1, lateral/avalanche/snap) from the reported drain pin\n",
             "current — even though those terms participated in the body residual.\n",
             "z431 reports the same drain-pin sum as `solve_2t_steady_state` (line 1929-1943)\n",
             "and ablates by zeroing `agidl=agisl=0` in `sd.scaled` for the OFF variant.\n",
             "\n## Per-VG1 RMSE (log-decades)\n",
             "```\n",
             f"{'VG1':<10}{'GIDL_OFF':>12}{'GIDL_ON':>12}{'delta':>10}{'z430_PIN':>12}\n",
             f"{'-'*56}\n"]
    for vg1 in ["VG1_0.2", "VG1_0.4", "VG1_0.6"]:
        d_on  = get(pb_on, vg1)
        d_off = get(pb_off, vg1)
        d430  = Z430_VSINT_PIN.get(vg1.replace("VG1_", "VG1_"), float("nan"))
        lines.append(f"{vg1:<10}{d_off:>12.3f}{d_on:>12.3f}{d_on - d_off:>10.3f}{d430:>12.3f}\n")
    lines.append(f"{'cell':<10}{off['cell_rmse_dec']:>12.3f}{on['cell_rmse_dec']:>12.3f}"
                 f"{on['cell_rmse_dec'] - off['cell_rmse_dec']:>10.3f}"
                 f"{Z430_VSINT_PIN['cell']:>12.3f}\n")
    lines.append("```\n")
    lines.append("\n## Snapback preservation check\n")
    lines.append("```\n")
    lines.append(json.dumps(snapback_check, indent=2))
    lines.append("\n```\n")
    lines.append("\n## Gates\n")
    for k, v in gates.items():
        lines.append(f"- **{k}**: {'PASS' if v else 'FAIL'}\n")
    lines.append("\n## Verdict\n")
    if gates["KILL_SHOT"]:
        lines.append("- **KILL SHOT TRIGGERED**: GIDL either failed to improve VG1=0.2 "
                     "or degraded VG1=0.4/0.6 by >0.3 dec. The 2.63 dec VG1=0.2 stall "
                     "is NOT explained by missing GIDL.\n")
    if gates["AMBITIOUS_pass"]:
        lines.append("- **AMBITIOUS PASS**: cell-wide < 1.0 dec with snapback intact.\n")
    elif gates["DISCOVERY_pass"]:
        lines.append("- **DISCOVERY PASS**: VG1=0.2 < 1.5 dec and snapback preserved.\n")
    else:
        lines.append("- Neither DISCOVERY nor AMBITIOUS gate satisfied.\n")
    lines.append("\n## Per-bias detail (GIDL_ON)\n```\n")
    for rec in results["GIDL_ON"]["per_bias"]:
        lines.append(f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f} "
                     f"RMSE={rec['log_rmse']:.3f} dec  Vb_max={rec['vb_max']:.3f} "
                     f"conv={rec['n_conv']}/{rec['n_pts']}\n")
    lines.append("```\n\n## Per-bias detail (GIDL_OFF)\n```\n")
    for rec in results["GIDL_OFF"]["per_bias"]:
        lines.append(f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f} "
                     f"RMSE={rec['log_rmse']:.3f} dec  Vb_max={rec['vb_max']:.3f} "
                     f"conv={rec['n_conv']}/{rec['n_pts']}\n")
    lines.append("```\n")
    (OUT / "honest_analysis.md").write_text("".join(lines))
    log(f"wrote honest_analysis.md")

    log(f"DONE wall={time.time() - t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
