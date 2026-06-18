"""z435 — λ-homotopy on avalanche gain (OpenAI proposal).

Hypothesis: cold Newton in z430 V_SINT_PIN lands on a low-current branch
(V_B near 0.3 V) and misses a hidden high-current latched branch
(V_B near 0.7 V firing the parasitic BJT). By inflating the avalanche
gain prefactor λ from 3.0 → 1.0 in a continuation over 10 steps at each
V_D, with Vb warm-started across the homotopy AND across V_D steps, we
might track that branch down to the physical λ=1.0.

Implementation: monkey-patch `compute_iimpact` in
`nsram.bsim4_port.nsram_cell_2T` to multiply Iii by a module-level
`_LAMBDA`. (Snapback subcircuit is disabled in the z430 baseline, so the
only avalanche-prefactor in play is alpha0 inside compute_iimpact.
Multiplying the returned Iii by λ is mathematically equivalent to
scaling alpha0 by λ.)

Reuses z427/z429/z430 infrastructure for loading, cfg, 1D Newton on V_B
with V_Sint pinned to 0.
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
OUT = ROOT / "results/z435_lambda_homotopy"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# --- reuse z427 + z429 modules ----------------------------------------- #
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port import nsram_cell_2T as _nc2t
from nsram.bsim4_port import nsram_cell as _nc

_ORIG_IIMPACT = _nc2t.compute_iimpact  # the BOUND import inside nsram_cell_2T
_ORIG_IIMPACT_NC = _nc.compute_iimpact  # also patch single-T module for safety

# Module-level λ knob. Multiplies Iii returned by compute_iimpact.
_LAMBDA = 1.0


def _patched_iimpact(model, sd, dc_result, Vds):
    Iii = _ORIG_IIMPACT(model, sd, dc_result, Vds=Vds)
    if _LAMBDA == 1.0:
        return Iii
    return _LAMBDA * Iii


# install patch (idempotent)
_nc2t.compute_iimpact = _patched_iimpact
_nc.compute_iimpact = _patched_iimpact


# ====================================================================== #
# Homotopy V_SINT pin solver                                              #
# ====================================================================== #

LAMBDA_SCHEDULE = [3.0, 2.7, 2.4, 2.1, 1.85, 1.6, 1.4, 1.25, 1.12, 1.0]


def homotopy_step(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f, Vb_warm):
    """Run λ-homotopy: λ=3 → 1 with 10 steps, warm-starting Vb.

    At each λ, 1D Newton on V_B with V_Sint pinned to 0. Final Vb at
    λ=1.0 is the homotopy answer for this (Vd, VG1, VG2) point.
    """
    global _LAMBDA
    Vb_traj = []
    Id_traj = []
    conv_traj = []
    Vb_cur = Vb_warm
    for lam in LAMBDA_SCHEDULE:
        _LAMBDA = float(lam)
        r = z429.run_vsint_pinned(
            cfg, model_M1, model_M2, bjt,
            float(Vd_f), float(VG1_f), float(VG2_f),
            Vsint_pin=0.0, Vb_init=Vb_cur)
        Vb_traj.append(r["Vb"])
        Id_traj.append(abs(r["Id"]))
        conv_traj.append(bool(r["converged"]))
        if r["converged"]:
            Vb_cur = r["Vb"]
        # else keep Vb_cur (don't reset)
    _LAMBDA = 1.0
    final = dict(
        Vb=Vb_traj[-1], Id=Id_traj[-1], converged=conv_traj[-1],
        Vb_traj=Vb_traj, Id_traj=Id_traj, conv_traj=conv_traj,
    )
    return final


def run_homotopy_cellwide(model_M1, model_M2, curves, sebas_rows):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
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
        homotopy_traces = []  # full λ trajectories
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k, Vd_f in enumerate(Vd_arr):
                    out = homotopy_step(cfg, model_M1, model_M2, bjt,
                                        float(Vd_f), float(c["VG1"]),
                                        float(c["VG2"]), Vb_warm)
                    Id_pred_list.append(out["Id"])
                    Vb_list.append(out["Vb"])
                    conv_list.append(out["converged"])
                    homotopy_traces.append({
                        "Vd": float(Vd_f),
                        "lambdas": LAMBDA_SCHEDULE,
                        "Vb_traj": out["Vb_traj"],
                        "Id_traj": out["Id_traj"],
                        "conv_traj": out["conv_traj"],
                    })
                    if out["converged"]:
                        Vb_warm = out["Vb"]
                    else:
                        Vb_warm = 0.0
        except Exception as e:
            fails += 1
            log(f"  fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
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
        rec = {
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse, "vb_max": vb_max,
            "n_conv": int(conv.sum()), "n_pts": len(Vd_arr),
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_pred": Id_pred.tolist(),
            "Vb": Vb_list,
            "Vsint": [0.0] * len(Vd_arr),
            "converged": conv_list,
            "homotopy_traces": homotopy_traces,
        }
        per_bias.append(rec)
        log(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}  rmse={rmse:.3f}  "
            f"Vb_max={vb_max:.3f}  conv={int(conv.sum())}/{len(Vd_arr)}")
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
    log(f"=== HOMOTOPY result: cell={cell:.3f} per_branch="
        f"{ {k:round(v,3) for k,v in per_branch_rmse.items()} }  "
        f"Vb_max={vb_max_overall:.3f}  conv={conv_rate*100:.1f}%  "
        f"fails={fails}  wall={time.time()-t0:.0f}s")
    return {
        "name": "LAMBDA_HOMOTOPY",
        "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "vsint_max_overall": 0.0,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "lambda_schedule": LAMBDA_SCHEDULE,
        "per_bias": per_bias,
    }


# ====================================================================== #
# Plots                                                                   #
# ====================================================================== #

def overlay_plot(VG1_target: float, hom_result: dict, baseline_per_bias,
                 fname: Path):
    """Overlay homotopy vs z430 V_SINT_PIN baseline + measured."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    rows_by_vg2 = {}
    for rec in hom_result["per_bias"]:
        if abs(rec["VG1"] - VG1_target) < 1e-3:
            rows_by_vg2[rec["VG2"]] = {"HOM": rec}
    if baseline_per_bias:
        for rec in baseline_per_bias:
            if abs(rec["VG1"] - VG1_target) < 1e-3:
                rows_by_vg2.setdefault(rec["VG2"], {})["BASE"] = rec
    vg2_vals = sorted(rows_by_vg2.keys())
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    for ax, vg2 in zip(axes, chosen):
        sub = rows_by_vg2.get(vg2, {})
        if "HOM" in sub:
            meas_Vd = sub["HOM"]["Vd"]; meas_Id = sub["HOM"]["Id_meas"]
            ax.plot(meas_Vd, meas_Id, "k-", lw=2.5, label="measured")
            ax.plot(sub["HOM"]["Vd"], sub["HOM"]["Id_pred"], "g--", lw=1.5,
                    label=f"λ-HOM  rmse={sub['HOM']['log_rmse']:.2f}")
        if "BASE" in sub:
            ax.plot(sub["BASE"]["Vd"], sub["BASE"]["Id_pred"], "r:", lw=1.5,
                    label=f"V_SINT_PIN  rmse={sub['BASE']['log_rmse']:.2f}")
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z435 λ-homotopy vs z430 V_SINT_PIN @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def vb_trajectory_plot(hom_result: dict, fname: Path):
    """Plot V_B(λ) for a few representative biases — shows whether the
    homotopy tracks a high-V_B (latched) branch or stays low-V_B."""
    # Pick 3 biases at high Vd (where snapback would happen)
    picks = []
    for VG1_target in [0.2, 0.4, 0.6]:
        matches = [r for r in hom_result["per_bias"]
                   if abs(r["VG1"] - VG1_target) < 1e-3]
        if matches:
            # pick the one with widest VG2 spread (mid VG2)
            mid = matches[len(matches)//2]
            picks.append(mid)
    if not picks:
        log("  no biases for vb_trajectory_plot")
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(5*len(picks), 4),
                             sharey=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, rec in zip(axes, picks):
        traces = rec["homotopy_traces"]
        # Find Vd close to 2.0 (avalanche regime) and 1.0 (sub-fold)
        Vds = [t["Vd"] for t in traces]
        if not Vds:
            continue
        Vd_high = max(Vds)
        Vd_mid = min(Vds, key=lambda v: abs(v - 0.5*Vd_high))
        Vd_low = min(Vds, key=lambda v: abs(v - 0.2*Vd_high))
        for Vd_pick, color, label in [
            (Vd_low, "tab:blue", f"Vd={Vd_low:.2f}"),
            (Vd_mid, "tab:orange", f"Vd={Vd_mid:.2f}"),
            (Vd_high, "tab:red", f"Vd={Vd_high:.2f}"),
        ]:
            tr = next(t for t in traces if t["Vd"] == Vd_pick)
            ax.plot(tr["lambdas"], tr["Vb_traj"], "o-", color=color, label=label)
        ax.set_xlabel("λ (avalanche gain prefactor)")
        ax.set_title(f"VG1={rec['VG1']:.1f}  VG2={rec['VG2']:+.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.invert_xaxis()  # λ goes 3 → 1
    axes[0].set_ylabel("V_B [V]")
    fig.suptitle("z435 λ-homotopy trajectory of V_B (λ: 3.0 → 1.0)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ====================================================================== #
# Main                                                                    #
# ====================================================================== #

def main():
    t_main = time.time()
    log(f"z435 starting — λ-homotopy on avalanche gain.  "
        f"λ schedule = {LAMBDA_SCHEDULE}")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    # Load z430 baseline for comparison
    base_path = ROOT / "results/z430_vsint_pin_cellwide/summary.json"
    baseline = json.loads(base_path.read_text())
    base_vsint = baseline["V_SINT_PIN"]
    log(f"z430 V_SINT_PIN baseline: cell_rmse={base_vsint['cell_rmse_dec']:.3f} dec, "
        f"per_branch={base_vsint['per_branch_rmse_dec']}, "
        f"Vb_max={base_vsint['vb_max_overall']:.3f}")
    # baseline per_bias traces aren't in summary; we'll plot homotopy with
    # measured + just the homotopy line (baseline summary doesn't carry per_bias)
    baseline_per_bias = base_vsint.get("per_bias", [])
    if not baseline_per_bias:
        # try to find a cached per_bias dump
        per_bias_dump = ROOT / "results/z430_vsint_pin_cellwide/per_bias_vsint_pin.json"
        if per_bias_dump.exists():
            baseline_per_bias = json.loads(per_bias_dump.read_text())

    # Run homotopy
    log("=== HOMOTOPY RUN ===")
    hom = run_homotopy_cellwide(model_M1, model_M2, curves, sebas_rows)

    # Comparison
    delta = base_vsint["cell_rmse_dec"] - hom["cell_rmse_dec"]
    log(f"DELTA: z430 V_SINT_PIN={base_vsint['cell_rmse_dec']:.3f} dec, "
        f"z435 HOM={hom['cell_rmse_dec']:.3f} dec, "
        f"improvement={delta:+.3f} dec")

    # Per-bias RMSE comparison (where possible)
    rmse_compare = []
    for rec in hom["per_bias"]:
        entry = {"VG1": rec["VG1"], "VG2": rec["VG2"],
                 "hom_rmse": rec["log_rmse"],
                 "hom_vb_max": rec["vb_max"]}
        if baseline_per_bias:
            for b in baseline_per_bias:
                if abs(b["VG1"] - rec["VG1"]) < 1e-3 and \
                   abs(b["VG2"] - rec["VG2"]) < 1e-3:
                    entry["base_rmse"] = b["log_rmse"]
                    entry["base_vb_max"] = b.get("vb_max")
                    entry["delta_dec"] = b["log_rmse"] - rec["log_rmse"]
                    break
        rmse_compare.append(entry)

    summary = {
        "z430_v_sint_pin_baseline": {
            "cell_rmse_dec": base_vsint["cell_rmse_dec"],
            "per_branch_rmse_dec": base_vsint["per_branch_rmse_dec"],
            "vb_max_overall": base_vsint["vb_max_overall"],
        },
        "z435_lambda_homotopy": {k: v for k, v in hom.items()
                                  if k != "per_bias"},
        "improvement_dec": delta,
        "rmse_compare": rmse_compare,
        "preflight": {
            "INFRA":      "PASS" if hom["n_biases_evaluated"] >= 20 else "FAIL",
            "DISCOVERY":  "PASS" if delta >= 0.3 else "FAIL",
            "AMBITIOUS":  "PASS" if hom["cell_rmse_dec"] < 1.0 else "FAIL",
            "KILL_SHOT":  ("CONFIRMED — no hidden latched branch"
                            if abs(delta) < 0.05
                            else "REFUTED — homotopy moved branch"),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote summary.json")

    # also dump the full per_bias homotopy traces (separate file, big)
    (OUT / "per_bias_homotopy.json").write_text(json.dumps(hom["per_bias"]))
    log(f"wrote per_bias_homotopy.json")

    # Plots
    log("=== PLOTS ===")
    for VG1 in [0.2, 0.4, 0.6]:
        overlay_plot(VG1, hom, baseline_per_bias,
                     OUT / f"overlay_VG1_{str(VG1).replace('.','p')}.png")
    vb_trajectory_plot(hom, OUT / "vb_trajectory.png")

    # Honest analysis
    pf = summary["preflight"]
    lines = [
        "# z435 — λ-homotopy on avalanche gain: honest analysis",
        "",
        f"- z430 V_SINT_PIN baseline: **{base_vsint['cell_rmse_dec']:.3f} dec** "
            f"(per-branch {base_vsint['per_branch_rmse_dec']}, "
            f"Vb_max={base_vsint['vb_max_overall']:.3f}).",
        f"- z435 λ-homotopy:           **{hom['cell_rmse_dec']:.3f} dec** "
            f"(per-branch {hom['per_branch_rmse_dec']}, "
            f"Vb_max={hom['vb_max_overall']:.3f}, "
            f"conv={hom['convergence_rate']*100:.1f}%).",
        f"- Δ = {delta:+.3f} dec (positive = homotopy improves).",
        "",
        "## Pre-registered gates",
        f"- INFRA      : {pf['INFRA']}  ({hom['n_biases_evaluated']} biases evaluated)",
        f"- DISCOVERY  : {pf['DISCOVERY']}  (gate ≥ 0.3 dec; got {delta:+.3f})",
        f"- AMBITIOUS  : {pf['AMBITIOUS']}  (gate < 1.0 dec; got {hom['cell_rmse_dec']:.3f})",
        f"- KILL_SHOT  : {pf['KILL_SHOT']}",
        "",
        "## Audit recap (see audit.md)",
        "- BSIM4 IIT1/IIT2 do not exist in this pyport — the substrate model uses",
        "  alpha0/alpha1/beta0 per BSIM4 manual §6.1. Those params ARE wired,",
        "  Sebas's CSV overrides reach `sd.scaled` via `patch_sd_scaled`, and Iii",
        "  is summed into R_B with correct sign. DeepSeek's silent-zero hypothesis",
        "  is falsified by inspection AND by `tests/test_leak.py` coverage.",
        "",
        "## Interpretation",
    ]
    if abs(delta) < 0.05:
        lines += [
            "- KILL_SHOT confirmed. Homotopy lands on the same branch as cold Newton.",
            "  There is no hidden latched (V_B≈0.7 V) branch in the BSIM4-only",
            "  body-KCL topology at these biases. The bias-dependent gap that",
            "  z430 exposes is NOT solver-discovery-limited; it is model-physics-",
            "  limited (BSIM4 with alpha0-only avalanche cannot reach the latched",
            "  branch — that requires the explicit parasitic-BJT subcircuit).",
        ]
    elif delta >= 0.3:
        lines += [
            "- DISCOVERY: λ-homotopy uncovered a higher-current branch missed by",
            "  cold Newton. Vb_max grew from",
            f"  {base_vsint['vb_max_overall']:.3f} → {hom['vb_max_overall']:.3f} V,",
            "  consistent with the latched-branch hypothesis.",
        ]
    else:
        lines += [
            f"- λ-homotopy moved the answer by {delta:+.3f} dec — below the",
            "  pre-registered 0.3-dec DISCOVERY gate. Either the latched branch",
            "  is only marginally distinct from cold-Newton's basin, or our",
            "  schedule/clamps prevented it from being tracked into λ=1.0.",
        ]
    lines += [
        "",
        "## Files",
        "- audit.md             — Part 1 pre-flight audit",
        "- summary.json         — per-bias RMSE comparison + preflight verdicts",
        "- per_bias_homotopy.json — full λ trajectories per bias point",
        "- overlay_VG1_{0p2,0p4,0p6}.png — measured vs HOM vs V_SINT_PIN",
        "- vb_trajectory.png    — V_B(λ) for representative biases",
        "",
        f"_Wall time: {time.time()-t_main:.0f} s_",
    ]
    (OUT / "honest_analysis.md").write_text("\n".join(lines))
    log(f"wrote honest_analysis.md")
    log(f"z435 done. wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
