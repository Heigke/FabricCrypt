"""z444 — Roychowdhury BESD PNPN port: single coupled snapback device.

Track B from the research plan. Tests the hypothesis that the missing
physics in our parasitic-BJT chain is not parameters but TOPOLOGY: the
whole regenerative PNPN stack must be modelled as ONE coupled device with
internal positive feedback, not as a decomposed pair of BJTs.

Source: github.com/jaijeet/BESD (Verilog-A `BESD_1_0_0.va` and ModSpec
`BESD_1_0_0_ModSpec.m`). Ported to torch in
`nsram/bsim4_port/besd_pnpn.py`.

Pre-registered gates (against z440/z432 cell-wide baselines):
    INFRA:     BESD ported + 33 biases run (DONE if script completes)
    DISCOVERY: cell-wide log-RMSE < 0.7 dec
    AMBITIOUS: cell-wide log-RMSE < 0.4 dec
    KILL_SHOT: BESD ≈ legacy PNPN attempts (cell-wide ≥ 1.0 dec)
               → topology was right, parameter problem unsolved

Comparison conditions:
    OFF      : legacy z427/z432 baseline (parasitic BJT chain intact)
    BESD_DEF : BESD with default Roychowdhury parameters
    BESD_LOW : BESD with weaker on-branch (Gon=0.02), tighter K
    BESD_HOT : BESD strong (Gon=0.5), early trigger (body_VBH=0.4)
    BESD_NO  : BESD enabled but replace_parasitics=False (additive overlay)

NO if-else inside the model. All branching is via tanh / softplus /
sigmoid. The internal latch state `s` is solved by Picard iteration
(24 iters) on the DC fixed point `tanh(K*(Vstar+sstar)) = sstar`.
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z444_besd_pnpn"
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

# Import the BESD port
from nsram.bsim4_port.besd_pnpn import (
    BESDParams, enable_besd_pnpn, disable_besd_pnpn,
    install_besd_pnpn_patch,
)

# Speed cap (per-bias pseudo-transient steps)
N_STEPS_FAST = 200


def run_one_bias_fast(cfg, model_M1, model_M2, bjt, Vd_arr, VG1, VG2,
                      backward=False, Vb_init_first=0.0, n_steps=N_STEPS_FAST):
    """Like z432.run_one_bias but with custom n_steps."""
    order = list(range(len(Vd_arr) - 1, -1, -1)) if backward else list(
        range(len(Vd_arr)))
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
        Vb_warm = r["Vb"]
        Id_out[idx] = float(r["Id"])
        Vb_out[idx] = float(r["Vb"])
        conv_out[idx] = bool(r["converged"])
        niter_out[idx] = int(r["niter"])
    return Id_out, Vb_out, conv_out, niter_out


# ============================================================ #
# BESD condition set
# ============================================================ #
# (name, params_kwargs, replace_parasitics, sint_frac)
BESD_CONDITIONS = [
    ("OFF",      None,                                            False, 0.0),
    ("BESD_DEF", dict(),                                          True,  0.0),
    ("BESD_LOW", dict(Gon=0.02, K=15.0, Alpha=4.0, I_scale=0.3),  True,  0.0),
    ("BESD_HOT", dict(Gon=0.5,  body_VBH=0.4, VT1=1.0,
                       body_VT1_shift=0.7, I_scale=1.0),          True,  0.0),
]


def apply_condition(cfg, name, params_kw, replace, sint_frac):
    if name == "OFF":
        disable_besd_pnpn(cfg)
        return
    p = BESDParams(**params_kw) if params_kw else BESDParams()
    enable_besd_pnpn(cfg, p, replace_parasitics=replace,
                     sint_frac=sint_frac, add_to_id=True)


# ============================================================ #
# Id assembly: add I_BESD to MOSFET Id when device is on the drain
# ============================================================ #

def id_with_besd(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f, Vb_f,
                 Vsint_pin=0.0):
    """Evaluate residuals once at converged (Vsint_pin, Vb) and return
    Id = base_Id + (besd_add_to_id ? I_BESD : 0)."""
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    Vd_t = torch.tensor([Vd_f], dtype=torch.float64)
    VG1_t = torch.tensor([VG1_f], dtype=torch.float64)
    VG2_t = torch.tensor([VG2_f], dtype=torch.float64)
    Vs_t = torch.tensor([Vsint_pin], dtype=torch.float64)
    Vb_t = torch.tensor([Vb_f], dtype=torch.float64)
    R_S, R_B, comps = _residuals(cfg, model_M1, bjt, Vd_t, VG1_t, VG2_t,
                                 Vs_t, Vb_t, None, None, model_M2=model_M2)
    base_Id = comps["Ids_M1"].item()
    if bool(getattr(cfg, "use_besd_pnpn", False)) and bool(
            getattr(cfg, "besd_add_to_id", False)):
        base_Id = base_Id + comps["I_BESD"].item()
    return base_Id, comps


# ============================================================ #
# Cell-wide eval
# ============================================================ #

def run_cellwide(name, params_kw, replace, sint_frac,
                 model_M1, model_M2, curves, sebas_rows):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    apply_condition(cfg, name, params_kw, replace, sint_frac)
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
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                    z427.patch_sd_scaled(sd_M2, P_M2):
                Id_pred, Vb_list, conv_list, niter_list = run_one_bias_fast(
                    cfg, model_M1, model_M2, bjt, Vd_arr,
                    c["VG1"], c["VG2"],
                    backward=False, Vb_init_first=0.0)
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue

        # BESD I/s diagnostics: extract at a few representative Vd points only.
        # Id is already correct because the residual patch folds I_BESD into
        # comps["Ids_M1"] when besd_add_to_id=True, so the pseudo-transient
        # integrator returned Id-with-BESD natively. We just sample s/I_BESD
        # for plotting at: final Vd (max bias) and the last 5 Vd points.
        I_BESD_arr = [0.0] * len(Vd_arr)
        s_arr = [0.0] * len(Vd_arr)
        if bool(getattr(cfg, "use_besd_pnpn", False)):
            try:
                # Sample BESD diagnostics every Nth Vd to keep cost low
                step = max(1, len(Vd_arr) // 6)
                idxs = list(range(0, len(Vd_arr), step))
                if (len(Vd_arr) - 1) not in idxs:
                    idxs.append(len(Vd_arr) - 1)
                with z427.patch_sd_scaled(sd_M1, P_M1), \
                        z427.patch_sd_scaled(sd_M2, P_M2):
                    for k in idxs:
                        _, comps_k = id_with_besd(
                            cfg, model_M1, model_M2, bjt,
                            float(Vd_arr[k]), c["VG1"], c["VG2"],
                            Vb_list[k], Vsint_pin=0.0)
                        I_BESD_arr[k] = float(
                            comps_k["I_BESD"].abs().item())
                        s_arr[k] = float(comps_k["s_BESD"].item())
            except Exception as e:
                log(f"  {name} BESD diag-extract fail "
                    f"VG1={c['VG1']} VG2={c['VG2']}: {e}")
                # Non-fatal — keep going with Id_pred from integrator

        Id_pred_t = torch.tensor(Id_pred, dtype=torch.float64)
        conv_t = torch.tensor(conv_list)
        if not conv_t.any():
            fails += 1
            continue
        # Take abs() — BESD can pull I_BESD negative briefly while s is small
        Id_pred_abs = Id_pred_t.abs() + log_eps
        log_p = torch.log10(Id_pred_abs)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv_t].mean()))
        # snapback amplitude (max-min of log10 over Vd≥0.5)
        vd_np = np.array(Vd_arr)
        Id_log = np.log10(np.abs(np.array(Id_pred)) + log_eps)
        Im_log = np.log10(np.array(Id_meas) + log_eps)
        mask_win = (vd_np >= 0.5)
        snap_pred = float(Id_log[mask_win].max() - Id_log[mask_win].min()) \
            if mask_win.any() else 0.0
        snap_meas = float(Im_log[mask_win].max() - Im_log[mask_win].min()) \
            if mask_win.any() else 0.0

        per_bias.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse,
            "vb_max": float(max(Vb_list)),
            "n_conv": int(conv_t.sum()),
            "n_pts": len(Vd_arr),
            "snap_amp_pred_dec": snap_pred,
            "snap_amp_meas_dec": snap_meas,
            "I_BESD_max": float(max(I_BESD_arr)) if I_BESD_arr else 0.0,
            "s_max": float(max(s_arr)) if s_arr else 0.0,
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_pred": Id_pred,
            "Vb": Vb_list,
            "s_BESD": s_arr,
            "I_BESD": I_BESD_arr,
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
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"])
                       for b, v in per_branch.items()}
    total_pts = sum(r["n_pts"] for r in per_bias)
    total_conv = sum(r["n_conv"] for r in per_bias)
    conv_rate = total_conv / max(total_pts, 1)
    log(f"  {name}: cell={cell:.3f} "
        f"per_branch={ {k: round(v,3) for k,v in per_branch_rmse.items()} } "
        f"conv={conv_rate*100:.1f}% fails={fails} "
        f"wall={time.time()-t0:.0f}s")
    return dict(
        name=name,
        params=params_kw or {},
        replace_parasitics=replace,
        sint_frac=sint_frac,
        cell_rmse_dec=cell,
        per_branch_rmse_dec=per_branch_rmse,
        n_biases_evaluated=cell_n,
        convergence_rate=conv_rate,
        fails=fails,
        wall_sec=round(time.time() - t0, 1),
        per_bias=per_bias,
    )


# ============================================================ #
# Plots
# ============================================================ #

def overlay_plot(VG1_target, runs_by_name, fname):
    off_rows = {r["VG2"]: r for r in runs_by_name["OFF"]
                if abs(r["VG1"] - VG1_target) < 1e-3}
    vg2_vals = sorted(off_rows.keys())
    if not vg2_vals:
        return
    chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]] \
        if len(vg2_vals) >= 3 else vg2_vals
    colors = {"OFF": "tab:red",
              "BESD_DEF": "tab:blue",
              "BESD_LOW": "tab:green",
              "BESD_HOT": "tab:purple",
              "BESD_NO":  "tab:olive",
              "BESD_SINT":"tab:cyan"}
    fig, axes = plt.subplots(1, len(chosen), figsize=(5*len(chosen), 4.5),
                              sharey=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, vg2 in zip(axes, chosen):
        meas = off_rows.get(vg2)
        if meas is None:
            continue
        ax.plot(meas["Vd"], meas["Id_meas"], "k-", lw=2.5, label="measured")
        for nm, per_bias in runs_by_name.items():
            rows = {r["VG2"]: r for r in per_bias
                    if abs(r["VG1"] - VG1_target) < 1e-3}
            r = rows.get(vg2)
            if r is None:
                continue
            Idp = np.abs(np.array(r["Id_pred"]))
            ax.plot(r["Vd"], Idp, "--", lw=1.3, alpha=0.85,
                    color=colors.get(nm, "gray"), label=nm)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f} VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z444 BESD PNPN overlay @ VG1={VG1_target:.1f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def plot_s_vs_vd(runs_by_name, fname):
    """Sanity plot: BESD latch state s vs Vd for a few biases."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for nm in ("BESD_DEF", "BESD_LOW", "BESD_HOT"):
        per_bias = runs_by_name.get(nm, [])
        for r in per_bias:
            if abs(r["VG1"] - 0.4) < 1e-3 and abs(r["VG2"]) < 1e-3:
                ax.plot(r["Vd"], r["s_BESD"], "-",
                        label=f"{nm} VG1=0.4 VG2=0.0")
                break
    ax.set_xlabel("V_D [V]")
    ax.set_ylabel("s (BESD latch state)")
    ax.set_title("z444 BESD internal latch state s vs V_D")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main
# ============================================================ #

Z432_BASELINE_CELL = 1.027   # results/z432_pseudotransient/summary.json


def main():
    t_main = time.time()
    log("z444 starting — BESD PNPN single-coupled-device snapback test")
    install_besd_pnpn_patch()   # patch _residuals once; conditions toggle cfg
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    runs = {}
    for (nm, params_kw, replace, sf) in BESD_CONDITIONS:
        log(f"=== Running BESD condition: {nm}  "
            f"params={params_kw} replace={replace} sint_frac={sf}")
        try:
            runs[nm] = run_cellwide(nm, params_kw, replace, sf,
                                     model_M1, model_M2, curves, sebas_rows)
        except Exception as e:
            log(f"  {nm} FATAL: {e}")
            log(traceback.format_exc())
            runs[nm] = dict(name=nm, error=str(e), cell_rmse_dec=float("inf"),
                            per_bias=[], n_biases_evaluated=0,
                            convergence_rate=0.0, fails=0, wall_sec=0,
                            params=params_kw or {},
                            replace_parasitics=replace, sint_frac=sf)

    runs_per_bias = {nm: r.get("per_bias", []) for nm, r in runs.items()}

    # Overlays for VG1=0.2, 0.4, 0.6
    for vg1, sfx in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        try:
            overlay_plot(vg1, runs_per_bias, OUT / f"overlay_VG1_{sfx}.png")
        except Exception as e:
            log(f"  overlay {sfx} fail: {e}")
    try:
        plot_s_vs_vd(runs_per_bias, OUT / "besd_s_state.png")
    except Exception as e:
        log(f"  s-state plot fail: {e}")

    # Per-bias table
    table_lines = ["# z444 — per-bias log-RMSE per BESD condition\n\n"]
    keys = sorted({(r["VG1"], r["VG2"]) for rs in runs_per_bias.values()
                   for r in rs})
    header = "| VG1 | VG2 | " + " | ".join(nm for nm in runs) + " | meas_snap |\n"
    sep = "|---|---|" + "|".join("---" for _ in runs) + "|---|\n"
    table_lines += [header, sep]
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
        snap_s = f"{meas_snap:.2f}" if meas_snap is not None else "—"
        table_lines.append(
            f"| {vg1:.2f} | {vg2:+.2f} | {' | '.join(cells)} | {snap_s} |\n")
    (OUT / "per_bias_table.md").write_text("".join(table_lines))

    # Cell-wide summary + gate evaluation
    cells = {nm: r.get("cell_rmse_dec", float("inf")) for nm, r in runs.items()}
    best_name = min(cells, key=cells.get)
    best_cell = cells[best_name]
    log(f"=== Cell-wide summary ===")
    for nm, v in cells.items():
        log(f"   {nm:>10s}: {v:.3f}  (n={runs[nm].get('n_biases_evaluated',0)},"
            f" conv={runs[nm].get('convergence_rate', 0.0)*100:.1f}%)")
    log(f"BEST condition: {best_name}  cell={best_cell:.3f}")
    log(f"z432 baseline (OFF reference): {Z432_BASELINE_CELL:.3f}")

    # Pre-registered gates
    besd_best = min({k: v for k, v in cells.items() if k != "OFF"}.items(),
                    key=lambda kv: kv[1], default=("none", float("inf")))
    besd_best_name, besd_best_cell = besd_best
    gate_eval = dict(
        INFRA_BESD_ported_and_ran=any(
            r.get("n_biases_evaluated", 0) > 0
            for nm, r in runs.items() if nm != "OFF"),
        DISCOVERY_cellwide_lt_0p7=(besd_best_cell < 0.7),
        AMBITIOUS_cellwide_lt_0p4=(besd_best_cell < 0.4),
        KILL_SHOT_no_improvement=(
            besd_best_cell >= 1.0 and best_name == "OFF"),
        OFF_cell=cells.get("OFF", None),
        BESD_best_name=besd_best_name,
        BESD_best_cell=besd_best_cell,
    )

    summary = dict(
        script="z444_besd_pnpn.py",
        date="2026-05-16",
        runs={nm: {k: v for k, v in r.items() if k != "per_bias"}
              for nm, r in runs.items()},
        cells=cells,
        best=dict(name=best_name, cell=best_cell),
        gates=gate_eval,
        z432_baseline=Z432_BASELINE_CELL,
        wall_total_sec=round(time.time() - t_main, 1),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote summary.json  total wall={time.time()-t_main:.0f}s")
    log(f"gates: {gate_eval}")

    # Drop per-bias detail to a separate file (don't pollute summary.json)
    (OUT / "per_bias_detail.json").write_text(
        json.dumps({nm: r.get("per_bias", []) for nm, r in runs.items()},
                   indent=1))

    LOG.close()


if __name__ == "__main__":
    main()
