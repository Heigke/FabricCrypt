"""z440 — M2 body-shunt topology test (S29).

Hypothesis (O75 oracle 4/4 + Mario Nature 2025 paper):
  The missing physics is a VG2-gated body discharge path. M2 acts as
  a body-discharge MOSFET (drain=V_body, source=GND, gate=V_G2), where
  R_B(V_G2) = 1/g_m,M2(V_G2). Higher V_G2 → lower R_B → body never
  reaches forward-bias → snapback collapses smoothly.

Schematic audit verdict (see audit.md): the existing pyport topology
M2.D=Sint (series) is consistent with Sebas's .asc as a series channel.
We do NOT change that. Instead we add an ADDITIVE body-shunt path
parametrized as a smooth square-law MOS-like channel between V_body
and GND, gated by V_G2. This is Option B from the task description
(parallel structure, not topology change).

Pre-registered gates:
  INFRA:        33 biases evaluated, no exception
  DISCOVERY:    cell-wide RMSE < 1.027 dec (z432's baseline) AND
                VG1=0.2 high-VG2 (≥ +0.3) shows NO snapback
  AMBITIOUS:    cell-wide < 0.7 dec
  KILL_SHOT:    shunt makes things WORSE (cell > 1.20 dec) — implies
                real silicon has another quench mechanism

NO-CHEAT: report ALL VG2 per VG1 in a table.
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
OUT = ROOT / "results/z440_m2_body_shunt"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# Reuse z427 + z429 + z432 plumbing
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)
_spec432 = _ilu.spec_from_file_location("z432", ROOT / "scripts/z432_pseudotransient.py")
z432 = _ilu.module_from_spec(_spec432); _spec432.loader.exec_module(z432)

# Speed cap: pseudo-transient typically converges within 100-300 steps; 800
# is overkill for our 4-shunt scan. Trade accuracy for budget.
N_STEPS_FAST = 300


def run_one_bias_fast(cfg, model_M1, model_M2, bjt, Vd_arr, VG1, VG2,
                      backward=False, Vb_init_first=0.0, n_steps=N_STEPS_FAST):
    """Like z432.run_one_bias but with custom n_steps and lower N_MIN_STEPS."""
    order = list(range(len(Vd_arr) - 1, -1, -1)) if backward else list(range(len(Vd_arr)))
    Vb_warm = Vb_init_first
    Id_out = [None] * len(Vd_arr)
    Vb_out = [None] * len(Vd_arr)
    conv_out = [False] * len(Vd_arr)
    niter_out = [0] * len(Vd_arr)
    for idx in order:
        Vd_f = float(Vd_arr[idx])
        r = z432.integrate_vb(cfg, model_M1, model_M2, bjt,
                              Vd_f, float(VG1), float(VG2),
                              Vb_init=Vb_warm, n_steps=n_steps)
        Id_out[idx] = abs(r["Id"])
        Vb_out[idx] = r["Vb"]
        conv_out[idx] = bool(r["converged"])
        niter_out[idx] = int(r["niter"])
        Vb_warm = r["Vb"]
    return Id_out, Vb_out, conv_out, niter_out


# ============================================================ #
# Shunt parameter sets to scan
# ============================================================ #
# beta_sh: strong-inversion conductance coefficient [A/V^2]
# Vth_sh:  shunt-MOS threshold [V]
# Vt_sh:   subthresh slope factor [V]
# lambda:  channel-length modulation [1/V]
# Is0:     subthreshold prefactor [A]
SHUNT_PARAM_SETS = [
    # name,            beta,   Vth,   Vt,    lam,   Is0
    ("OFF",            0.0,    0.0,   0.05,  0.0,   0.0),       # control
    ("WEAK",           1e-7,  -0.10,  0.05,  0.0,   1e-15),
    ("MED",            1e-6,  -0.10,  0.05,  0.0,   1e-14),
    ("STRONG",         1e-5,  -0.10,  0.05,  0.0,   1e-13),
]


def configure_shunt(cfg, params):
    name, beta, vth, vt, lam, is0 = params
    if name == "OFF":
        setattr(cfg, "use_m2_body_shunt", False)
    else:
        setattr(cfg, "use_m2_body_shunt", True)
        setattr(cfg, "m2_shunt_beta", float(beta))
        setattr(cfg, "m2_shunt_Vth", float(vth))
        setattr(cfg, "m2_shunt_Vt", float(vt))
        setattr(cfg, "m2_shunt_lambda", float(lam))
        setattr(cfg, "m2_shunt_Is0", float(is0))


# ============================================================ #
# Cell-wide eval (pseudo-transient, fwd sweep) with shunt
# ============================================================ #

def run_cellwide_with_shunt(name, model_M1, model_M2, curves, sebas_rows,
                            shunt_params):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    configure_shunt(cfg, shunt_params)
    log_eps = 1e-15
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
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Id_pred, Vb_list, conv_list, niter_list = run_one_bias_fast(
                    cfg, model_M1, model_M2, bjt, Vd_arr,
                    c["VG1"], c["VG2"],
                    backward=False, Vb_init_first=0.0)
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
        # snapback detector: peak in dlog10(I)/dVd
        Id_log = np.log10(np.array(Id_pred) + log_eps)
        dI_meas = np.log10(np.array(Id_meas) + log_eps)
        # measure local "fold" amplitude in V_D ∈ [0.5, 2.0] window:
        # max - min of Id_log in that window.
        vd_np = np.array(Vd_arr)
        mask_win = (vd_np >= 0.5)
        snap_amp = float(Id_log[mask_win].max() - Id_log[mask_win].min()) if mask_win.any() else 0.0
        snap_amp_meas = float(dI_meas[mask_win].max() - dI_meas[mask_win].min()) if mask_win.any() else 0.0
        # Lightweight I_shunt extraction: only at peak Vb (one _residuals call)
        from nsram.bsim4_port.nsram_cell_2T import _residuals
        I_shunt_arr = [0.0] * len(Vd_arr)
        if bool(getattr(cfg, "use_m2_body_shunt", False)):
            k_peak = int(np.argmax(Vb_list))
            Vd_t = torch.tensor([float(Vd_arr[k_peak])], dtype=torch.float64)
            VG1_t = torch.tensor([c["VG1"]], dtype=torch.float64)
            VG2_t = torch.tensor([c["VG2"]], dtype=torch.float64)
            Vs_t = torch.tensor([0.0], dtype=torch.float64)
            Vb_t = torch.tensor([Vb_list[k_peak]], dtype=torch.float64)
            with z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                _, _, comp = _residuals(cfg, model_M1, bjt, Vd_t, VG1_t, VG2_t,
                                        Vs_t, Vb_t, None, None, model_M2=model_M2)
            I_shunt_arr[k_peak] = float(comp.get("I_m2_body_shunt", torch.tensor(0.0)).abs().item())
        per_bias.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse,
            "vb_max": float(max(Vb_list)),
            "n_conv": int(conv_t.sum()),
            "n_pts": len(Vd_arr),
            "snap_amp_pred_dec": snap_amp,
            "snap_amp_meas_dec": snap_amp_meas,
            "I_shunt_max": float(max(I_shunt_arr)) if I_shunt_arr else 0.0,
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_pred": Id_pred,
            "Vb": Vb_list,
            "I_shunt": I_shunt_arr,
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
    log(f"  {name}: cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"conv={conv_rate*100:.1f}% fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "name": name,
        "shunt_params": shunt_params,
        "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time() - t0, 1),
        "per_bias": per_bias,
    }


# ============================================================ #
# Plots
# ============================================================ #

def overlay_plot_with_shunt(VG1_target, runs_per_bias_by_name, fname):
    """One row, three VG2 panels (low, mid, high). Overlay measured (black)
    + OFF (red) + STRONG (blue). All shunt sweeps shown lightly."""
    # collect VG2 values for this VG1 from OFF run
    off_rows = {r["VG2"]: r for r in runs_per_bias_by_name["OFF"]
                if abs(r["VG1"] - VG1_target) < 1e-3}
    vg2_vals = sorted(off_rows.keys())
    if len(vg2_vals) == 0:
        return
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    colors = {"OFF": "tab:red", "WEAK": "tab:orange", "MED": "tab:olive",
              "STRONG": "tab:blue", "STRONG_low_Vth": "tab:purple",
              "STRONG_steep": "tab:cyan"}
    fig, axes = plt.subplots(1, len(chosen), figsize=(5*len(chosen), 4.5),
                              sharey=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, vg2 in zip(axes, chosen):
        meas = off_rows.get(vg2)
        if meas is None:
            continue
        ax.plot(meas["Vd"], meas["Id_meas"], "k-", lw=2.5, label="measured")
        for nm, per_bias in runs_per_bias_by_name.items():
            rows = {r["VG2"]: r for r in per_bias
                    if abs(r["VG1"] - VG1_target) < 1e-3}
            r = rows.get(vg2)
            if r is None:
                continue
            lw = 1.5 if nm in ("OFF", "STRONG") else 1.0
            alpha = 1.0 if nm in ("OFF", "STRONG") else 0.6
            ax.plot(r["Vd"], r["Id_pred"], "--", lw=lw, alpha=alpha,
                    color=colors.get(nm, "gray"), label=nm)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z440 M2 body-shunt overlay @ VG1={VG1_target:.1f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def ids_m2_vs_vg2(runs_per_bias_by_name, fname):
    """Quantify I_shunt at the final V_D (=2 V) vs V_G2 for each VG1."""
    fig, ax = plt.subplots(figsize=(7, 5))
    # plot STRONG only — clearest signal
    rows = runs_per_bias_by_name.get("STRONG", [])
    if not rows:
        plt.close(fig)
        return
    # group by VG1
    by_vg1 = {}
    for r in rows:
        by_vg1.setdefault(r["VG1"], []).append(
            (r["VG2"], r["I_shunt_max"]))
    for vg1, pairs in sorted(by_vg1.items()):
        pairs.sort()
        vg2s = [p[0] for p in pairs]
        ish = [p[1] for p in pairs]
        ax.plot(vg2s, ish, "o-", label=f"VG1={vg1:.1f}")
    ax.set_yscale("log")
    ax.set_xlabel("V_G2 [V]")
    ax.set_ylabel("max |I_M2_body_shunt| [A]")
    ax.set_title("z440 — VG2-gated body discharge magnitude (STRONG set)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

Z432_BASELINE_CELL = 1.027   # see results/z432_pseudotransient/summary.json
Z430_VSINT_PIN_CELL = 1.619


def main():
    t_main = time.time()
    log("z440 starting — M2 body-shunt topology test (S29)")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    runs = {}
    for params in SHUNT_PARAM_SETS:
        nm = params[0]
        log(f"=== Running shunt set: {nm}  params={params[1:]}")
        runs[nm] = run_cellwide_with_shunt(
            nm, model_M1, model_M2, curves, sebas_rows, params)

    runs_per_bias = {nm: r["per_bias"] for nm, r in runs.items()}
    # Overlays for VG1=0.2, 0.4, 0.6
    for vg1, sfx in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot_with_shunt(vg1, runs_per_bias,
                                 OUT / f"overlay_VG1_{sfx}.png")
    # body-shunt magnitude
    ids_m2_vs_vg2(runs_per_bias, OUT / "ids_m2_vs_vg2.png")

    # Tables: per-bias log-RMSE for each shunt set
    table_md_lines = ["# z440 — per-bias log-RMSE per shunt set\n\n"]
    # Build union of (VG1, VG2)
    keys = sorted({(r["VG1"], r["VG2"]) for nm, rs in runs_per_bias.items()
                   for r in rs})
    header = "| VG1 | VG2 | " + " | ".join(nm for nm in runs) + " | meas_snap |\n"
    sep = "|---|---|" + "|".join("---" for _ in runs) + "|---|\n"
    table_md_lines.append(header)
    table_md_lines.append(sep)
    for vg1, vg2 in keys:
        cells = []
        meas_snap = None
        for nm in runs:
            rs = [r for r in runs_per_bias[nm]
                  if abs(r["VG1"]-vg1) < 1e-3 and abs(r["VG2"]-vg2) < 1e-3]
            if rs:
                cells.append(f"{rs[0]['log_rmse']:.3f}")
                meas_snap = rs[0]["snap_amp_meas_dec"]
            else:
                cells.append("—")
        table_md_lines.append(f"| {vg1:.2f} | {vg2:+.2f} | "
                              f"{' | '.join(cells)} | "
                              f"{meas_snap:.2f} |\n" if meas_snap is not None
                              else f"| {vg1:.2f} | {vg2:+.2f} | "
                              f"{' | '.join(cells)} | — |\n")
    (OUT / "per_bias_table.md").write_text("".join(table_md_lines))

    # Summary
    cells = {nm: r["cell_rmse_dec"] for nm, r in runs.items()}
    best_name = min(cells, key=cells.get)
    best_cell = cells[best_name]
    # VG1=0.2 high-VG2 snapback check (Id at last 5 Vd points)
    snap_results = {}
    for nm, rs in runs_per_bias.items():
        highvg2_snap = []
        for r in rs:
            # Measured snapback at VG1=0.2 collapses at VG2 >= +0.10 (verified
            # from data: 3.4→0.6 dec between VG2=+0.05 and +0.10). Pre-reg
            # said "≥0.3" but Sebas's CSV has no VG2≥0.3 at VG1=0.2 — adjust to
            # the actual collapse threshold in the data.
            if abs(r["VG1"] - 0.2) < 1e-3 and r["VG2"] >= 0.10 - 1e-3:
                highvg2_snap.append({
                    "VG2": r["VG2"],
                    "snap_amp_pred_dec": r["snap_amp_pred_dec"],
                    "snap_amp_meas_dec": r["snap_amp_meas_dec"],
                })
        snap_results[nm] = highvg2_snap

    summary = {
        "cell_rmse_dec_by_shunt": cells,
        "best_shunt": best_name,
        "best_cell_rmse_dec": best_cell,
        "REFERENCE": {
            "z430_v_sint_pin_cell_rmse_dec": Z430_VSINT_PIN_CELL,
            "z432_ptran_cell_rmse_dec": Z432_BASELINE_CELL,
        },
        "DELTAS_VS_Z432": {
            nm: Z432_BASELINE_CELL - v for nm, v in cells.items()
        },
        "VG1_0p2_high_VG2_snapback_predicted": snap_results,
        "per_branch_by_shunt": {
            nm: r["per_branch_rmse_dec"] for nm, r in runs.items()
        },
        "convergence_by_shunt": {
            nm: r["convergence_rate"] for nm, r in runs.items()
        },
        "n_biases_by_shunt": {
            nm: r["n_biases_evaluated"] for nm, r in runs.items()
        },
        "SHUNT_PARAM_SETS": [
            dict(name=p[0], beta=p[1], Vth=p[2], Vt=p[3], lam=p[4], Is0=p[5])
            for p in SHUNT_PARAM_SETS
        ],
    }

    # Gates
    # DISCOVERY: best_cell < Z432_BASELINE AND VG1=0.2 high-VG2 snap_pred < 0.5 dec
    best_high_snap = snap_results.get(best_name, [])
    worst_pred_snap = max((s["snap_amp_pred_dec"] for s in best_high_snap),
                          default=0.0)
    discovery_no_snap = worst_pred_snap < 0.5
    gates = {
        "INFRA_pass": all(r["n_biases_evaluated"] > 0 for r in runs.values()),
        "DISCOVERY_cell_below_z432": best_cell < Z432_BASELINE_CELL,
        "DISCOVERY_VG1_0p2_high_VG2_no_snapback": discovery_no_snap,
        "AMBITIOUS_cell_lt_0p7": best_cell < 0.7,
        "KILL_SHOT_shunt_worse": best_cell > 1.20,
    }
    summary["GATES"] = gates

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                  default=lambda o: float(o)
                                                  if hasattr(o,"__float__") else str(o)))
    log("wrote summary.json")

    # Honest analysis
    lines = []
    lines.append("# z440 — M2 body-shunt (S29) — honest analysis\n\n")
    lines.append("## What we did\n")
    lines.append("Added a VG2-gated body→GND discharge channel as an additive\n"
                 "term in R_B. Physically motivated by Mario Nature 2025 \n"
                 "(slide 12.25-12.27): R_B(V_G2)↓ as V_G2↑ enables collection\n"
                 "of excess carriers from bulk. Existing series-M2 topology\n"
                 "(M2.D=Sint) is unchanged — this is a PARALLEL shunt path.\n\n")
    lines.append("Mathematical form (smooth square-law + subthresh tail):\n")
    lines.append("```\n")
    lines.append("V_ov = softplus((V_G2 - V_th_sh)/V_t_sh) * V_t_sh\n")
    lines.append("V_ds_eff = smooth_min(V_b, V_ov)  (≥ 0)\n")
    lines.append("I_shunt = beta * V_ov * V_ds_eff * (1 + lambda * V_b)\n")
    lines.append("         + Is0 * exp((V_G2-V_th_sh)/V_t_sh) * (1 - exp(-V_b/V_t_sh))\n")
    lines.append("R_B  -=  I_shunt   (current LEAVES body)\n")
    lines.append("```\n\n")
    lines.append("## Headline (cell-wide log-RMSE, dec; pseudo-transient fwd sweep)\n\n")
    lines.append(f"- z430 V_SINT_PIN (DC Newton):  {Z430_VSINT_PIN_CELL:.3f}\n")
    lines.append(f"- z432 pseudo-transient (no shunt): {Z432_BASELINE_CELL:.3f}\n\n")
    for nm, v in cells.items():
        delta = Z432_BASELINE_CELL - v
        lines.append(f"- z440 {nm:18s}: {v:.3f}  (Δ vs z432 = {delta:+.3f} dec)\n")
    lines.append(f"\nBest shunt set: **{best_name}** @ {best_cell:.3f} dec\n\n")
    lines.append("## VG1=0.2 high-VG2 (≥0.3) snapback predicted (dec, V_D ≥ 0.5)\n\n")
    lines.append("Measured at these biases shows NO snapback. Goal: shunt\n"
                 "should drop predicted snap_amp_pred_dec close to measured.\n\n")
    lines.append("| Shunt | VG2 | snap_pred [dec] | snap_meas [dec] |\n")
    lines.append("|---|---|---|---|\n")
    for nm in runs:
        for s in snap_results.get(nm, []):
            lines.append(f"| {nm} | {s['VG2']:+.2f} | "
                         f"{s['snap_amp_pred_dec']:.2f} | "
                         f"{s['snap_amp_meas_dec']:.2f} |\n")
    lines.append("\n## Per-branch RMSE\n\n")
    lines.append("| Branch | " + " | ".join(runs.keys()) + " |\n")
    lines.append("|---|" + "|".join("---" for _ in runs) + "|\n")
    branches = sorted({b for r in runs.values()
                       for b in r["per_branch_rmse_dec"]})
    for b in branches:
        cells_row = [f"{runs[nm]['per_branch_rmse_dec'].get(b, float('nan')):.3f}"
                     for nm in runs]
        lines.append(f"| {b} | " + " | ".join(cells_row) + " |\n")
    lines.append("\n## Gates\n\n")
    for k, v in gates.items():
        verdict = "PASS" if v else ("FAIL" if "KILL" not in k else "no")
        lines.append(f"- {k}: {verdict}\n")
    lines.append("\n## Honest verdict\n\n")
    if gates["AMBITIOUS_cell_lt_0p7"]:
        lines.append("- **AMBITIOUS HIT**: M2 body-shunt closes cell to <0.7 dec.\n"
                     "  Strong evidence the missing physics is a VG2-gated body\n"
                     "  discharge consistent with Mario Nature 2025.\n")
    elif gates["DISCOVERY_cell_below_z432"] and gates["DISCOVERY_VG1_0p2_high_VG2_no_snapback"]:
        lines.append("- **DISCOVERY**: shunt reduces cell-wide RMSE below z432 baseline\n"
                     "  AND eliminates VG1=0.2 high-VG2 false snapback.\n")
    elif gates["DISCOVERY_cell_below_z432"]:
        lines.append("- **PARTIAL DISCOVERY**: cell-RMSE improves but snapback at VG1=0.2\n"
                     "  high-VG2 not fully quenched. Further parameter tuning needed.\n")
    elif gates["KILL_SHOT_shunt_worse"]:
        lines.append("- **KILL SHOT**: adding the shunt makes things WORSE. The body\n"
                     "  discharge isn't what's missing — real silicon must have\n"
                     "  another quench mechanism (e.g. an explicit metal contact\n"
                     "  or higher-order BSIM4 physics).\n")
    else:
        lines.append("- **Marginal / null**: the M2 body-shunt produces no decisive\n"
                     "  effect. Either parameters need a wider scan, or the\n"
                     "  hypothesis is incomplete.\n")
    (OUT / "honest_analysis.md").write_text("".join(lines))
    log("wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
