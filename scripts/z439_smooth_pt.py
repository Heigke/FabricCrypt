"""z439 — Smooth pseudo-transient body integration (S28).

Hypothesis: the "shake" (zig-zag) in z432 overlays at VG1=0.4/0.6 is solver
noise from EXPLICIT EULER, not physics. Implicit Euler / BDF2 should remove
or attenuate it.

Three variants integrated:
  - BASELINE   : explicit Euler                         (same as z432)
  - IMPLICIT   : implicit Euler (1-2 Newton iterations) with FD Jacobian
  - BDF2       : 2-step BDF2 (uses V_B[t-1] and V_B[t] to advance to t+1)

We also run an extra `HIRES` variant of all 3 with n_steps=2000 and dt /10
restricted to the 3 target VG1 values (0.2/0.4/0.6) for smoothness comparison.

Backward sweep only (it had lower cell-rmse 1.027 dec in z432 and is the
sweep direction the user saw "shake" in).
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
OUT = ROOT / "results/z439_smooth_pt"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# reuse z427 (loaders, cfg) + z429 (resid_pair)
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)


# === Default integration parameters (matches z432) ===
C_B_DEFAULT = 1.0e-18
DT_DEFAULT = 1.0e-9
N_STEPS_DEFAULT = 800
TOL_DV_DEFAULT = 1.0e-5
N_MIN_STEPS = 100
RESID_REL_TOL = 1e-4
DVB_CAP = 0.05  # mV per step cap
VB_MIN, VB_MAX = -0.2, 1.0

# FD step for Jacobian
FD_H = 1e-4


def _RB(cfg, model_M1, model_M2, bjt, Vb, Vd_f, VG1_f, VG2_f,
        Vsint_pin=0.0):
    _, R_B, Id = z429.resid_pair(cfg, model_M1, model_M2, bjt,
                                 Vsint_pin, float(Vb), Vd_f, VG1_f, VG2_f)
    return R_B, Id


def _clamp(v):
    if v > VB_MAX: return VB_MAX
    if v < VB_MIN: return VB_MIN
    return v


def integrate_explicit(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                       Vb_init, C_B=C_B_DEFAULT, dt=DT_DEFAULT,
                       n_steps=N_STEPS_DEFAULT, tol_dv=TOL_DV_DEFAULT,
                       Vsint_pin=0.0):
    Vb = float(Vb_init)
    converged = False
    n_used = n_steps
    for k in range(n_steps):
        R_B, Id = _RB(cfg, model_M1, model_M2, bjt, Vb, Vd_f, VG1_f, VG2_f, Vsint_pin)
        dVb = (R_B / C_B) * dt
        if abs(dVb) > DVB_CAP:
            dVb = math.copysign(DVB_CAP, dVb)
        Vb_new = _clamp(Vb + dVb)
        rel_tol = RESID_REL_TOL * max(abs(Id), 1e-12)
        if k >= N_MIN_STEPS and abs(Vb_new - Vb) < tol_dv and abs(R_B) < rel_tol:
            Vb = Vb_new; n_used = k + 1; converged = True; break
        Vb = Vb_new
    R_B, Id = _RB(cfg, model_M1, model_M2, bjt, Vb, Vd_f, VG1_f, VG2_f, Vsint_pin)
    return dict(Vb=Vb, Id=Id, resid_RB=abs(R_B), niter=n_used, converged=converged)


def integrate_implicit(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                       Vb_init, C_B=C_B_DEFAULT, dt=DT_DEFAULT,
                       n_steps=N_STEPS_DEFAULT, tol_dv=TOL_DV_DEFAULT,
                       Vsint_pin=0.0, n_newton=2):
    """Implicit Euler:
        f(Vb_new) = (Vb_new - Vb)/dt - R_B(Vb_new)/C_B = 0
    Solve with `n_newton` Newton steps; J = 1/dt - (dR_B/dVb)/C_B (FD).
    """
    Vb = float(Vb_init)
    converged = False
    n_used = n_steps
    inv_dt = 1.0 / dt
    for k in range(n_steps):
        # Newton iterate with initial guess = Vb (i.e., explicit predictor)
        Vb_guess = Vb
        for _it in range(n_newton):
            R_B_g, Id_g = _RB(cfg, model_M1, model_M2, bjt, Vb_guess,
                              Vd_f, VG1_f, VG2_f, Vsint_pin)
            f_val = (Vb_guess - Vb) * inv_dt - R_B_g / C_B
            # FD Jacobian
            R_B_p, _ = _RB(cfg, model_M1, model_M2, bjt, _clamp(Vb_guess + FD_H),
                           Vd_f, VG1_f, VG2_f, Vsint_pin)
            dRB_dV = (R_B_p - R_B_g) / FD_H
            J = inv_dt - dRB_dV / C_B
            if abs(J) < 1e-30:
                break
            dV = -f_val / J
            # damp to keep within DVB_CAP
            if abs(dV) > DVB_CAP:
                dV = math.copysign(DVB_CAP, dV)
            Vb_guess = _clamp(Vb_guess + dV)
            if abs(dV) < 1e-9:
                break
        Vb_new = Vb_guess
        # evaluate residual at the (implicit) solution
        R_B, Id = _RB(cfg, model_M1, model_M2, bjt, Vb_new,
                      Vd_f, VG1_f, VG2_f, Vsint_pin)
        rel_tol = RESID_REL_TOL * max(abs(Id), 1e-12)
        if k >= N_MIN_STEPS and abs(Vb_new - Vb) < tol_dv and abs(R_B) < rel_tol:
            Vb = Vb_new; n_used = k + 1; converged = True; break
        Vb = Vb_new
    R_B, Id = _RB(cfg, model_M1, model_M2, bjt, Vb, Vd_f, VG1_f, VG2_f, Vsint_pin)
    return dict(Vb=Vb, Id=Id, resid_RB=abs(R_B), niter=n_used, converged=converged)


def integrate_bdf2(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                   Vb_init, C_B=C_B_DEFAULT, dt=DT_DEFAULT,
                   n_steps=N_STEPS_DEFAULT, tol_dv=TOL_DV_DEFAULT,
                   Vsint_pin=0.0, n_newton=2):
    """BDF2:  (3 V_new - 4 V_n + V_{n-1}) / (2 dt) = R_B(V_new)/C_B.
    First step bootstrapped with implicit Euler.
    """
    Vb_nm1 = float(Vb_init)  # V_{n-1}
    # first step: implicit Euler
    r0 = integrate_implicit(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                            Vb_init=Vb_nm1, C_B=C_B, dt=dt, n_steps=1,
                            tol_dv=0.0, Vsint_pin=Vsint_pin, n_newton=n_newton)
    Vb_n = r0["Vb"]
    converged = False
    n_used = n_steps
    coeff = 1.5 / dt
    for k in range(1, n_steps):
        # Newton on g(V) = (3V - 4 Vb_n + Vb_nm1)/(2 dt) - R_B(V)/C_B
        Vb_guess = Vb_n  # predictor: previous value
        for _it in range(n_newton):
            R_B_g, Id_g = _RB(cfg, model_M1, model_M2, bjt, Vb_guess,
                              Vd_f, VG1_f, VG2_f, Vsint_pin)
            g_val = (3.0 * Vb_guess - 4.0 * Vb_n + Vb_nm1) / (2.0 * dt) - R_B_g / C_B
            R_B_p, _ = _RB(cfg, model_M1, model_M2, bjt, _clamp(Vb_guess + FD_H),
                           Vd_f, VG1_f, VG2_f, Vsint_pin)
            dRB_dV = (R_B_p - R_B_g) / FD_H
            J = coeff - dRB_dV / C_B
            if abs(J) < 1e-30:
                break
            dV = -g_val / J
            if abs(dV) > DVB_CAP:
                dV = math.copysign(DVB_CAP, dV)
            Vb_guess = _clamp(Vb_guess + dV)
            if abs(dV) < 1e-9:
                break
        Vb_new = Vb_guess
        R_B, Id = _RB(cfg, model_M1, model_M2, bjt, Vb_new,
                      Vd_f, VG1_f, VG2_f, Vsint_pin)
        rel_tol = RESID_REL_TOL * max(abs(Id), 1e-12)
        if k >= N_MIN_STEPS and abs(Vb_new - Vb_n) < tol_dv and abs(R_B) < rel_tol:
            Vb_nm1, Vb_n = Vb_n, Vb_new
            n_used = k + 1; converged = True; break
        Vb_nm1, Vb_n = Vb_n, Vb_new
    R_B, Id = _RB(cfg, model_M1, model_M2, bjt, Vb_n, Vd_f, VG1_f, VG2_f, Vsint_pin)
    return dict(Vb=Vb_n, Id=Id, resid_RB=abs(R_B), niter=n_used, converged=converged)


INTEGRATORS = {
    "BASELINE": integrate_explicit,
    "IMPLICIT": integrate_implicit,
    "BDF2": integrate_bdf2,
}


# ============================================================ #
# Cell-wide runner
# ============================================================ #

def run_one_bias(integ_fn, cfg, model_M1, model_M2, bjt, Vd_arr, VG1, VG2,
                 backward=True, Vb_init_first=0.0,
                 n_steps=N_STEPS_DEFAULT, dt=DT_DEFAULT):
    if backward:
        order = list(range(len(Vd_arr) - 1, -1, -1))
    else:
        order = list(range(len(Vd_arr)))
    Vb_warm = Vb_init_first
    Id_out = [None] * len(Vd_arr)
    Vb_out = [None] * len(Vd_arr)
    conv_out = [False] * len(Vd_arr)
    niter_out = [0] * len(Vd_arr)
    for idx in order:
        Vd_f = float(Vd_arr[idx])
        r = integ_fn(cfg, model_M1, model_M2, bjt, Vd_f,
                     float(VG1), float(VG2), Vb_init=Vb_warm,
                     n_steps=n_steps, dt=dt)
        Id_out[idx] = abs(r["Id"])
        Vb_out[idx] = r["Vb"]
        conv_out[idx] = bool(r["converged"])
        niter_out[idx] = int(r["niter"])
        Vb_warm = r["Vb"]
    return Id_out, Vb_out, conv_out, niter_out


def run_cellwide(variant_name, integ_fn, model_M1, model_M2, curves, sebas_rows,
                 n_steps=N_STEPS_DEFAULT, dt=DT_DEFAULT,
                 limit_vg1=None):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    log_eps = 1e-15
    per_bias = []
    fails = 0
    t0 = time.time()
    vb_max_overall = -1e30
    for c in curves:
        if limit_vg1 is not None:
            if not any(abs(c["VG1"] - v) < 1e-3 for v in limit_vg1):
                continue
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Id_pred, Vb_list, conv_list, niter_list = run_one_bias(
                    integ_fn, cfg, model_M1, model_M2, bjt, Vd_arr,
                    c["VG1"], c["VG2"], backward=True, Vb_init_first=0.0,
                    n_steps=n_steps, dt=dt)
        except Exception as e:
            fails += 1
            log(f"  {variant_name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
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
            "log_rmse": rmse, "vb_max": vb_max,
            "n_conv": int(conv_t.sum()), "n_pts": len(Vd_arr),
            "niter_mean": float(np.mean(niter_list)),
            "Vd": Vd_arr.tolist(),
            "Id_meas": Id_meas.tolist(),
            "Id_pred": Id_pred, "Vb": Vb_list,
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
    log(f"  {variant_name}: cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"Vb_max={vb_max_overall:.3f} conv_rate={conv_rate*100:.1f}% fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "name": variant_name,
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
# Smoothness metric
# ============================================================ #

def smoothness_per_bias(per_bias):
    """std of d log10(Id) / d V_D — lower = smoother."""
    stds = []
    for r in per_bias:
        Vd = np.array(r["Vd"])
        Id = np.array(r["Id_pred"])
        if len(Vd) < 3:
            continue
        # log10 derivative w.r.t. V_D
        log_id = np.log10(np.maximum(Id, 1e-30))
        # finite differences
        dlog = np.diff(log_id) / np.maximum(np.diff(Vd), 1e-9)
        # measure of zig-zag: std of second derivative (curvature noise)
        d2 = np.diff(dlog)
        stds.append(float(np.std(d2)))
    return stds


def smoothness_summary(per_bias_by_variant):
    out = {}
    for v, pb in per_bias_by_variant.items():
        s = smoothness_per_bias(pb)
        # per-VG1 breakdown
        per_vg1 = {}
        for r, s_i in zip(pb, s if len(s)==len(pb) else s):
            pass
        # explicit per-VG1
        per_vg1_acc = {}
        for r in pb:
            Vd = np.array(r["Vd"])
            Id = np.array(r["Id_pred"])
            if len(Vd) < 3: continue
            log_id = np.log10(np.maximum(Id, 1e-30))
            dlog = np.diff(log_id) / np.maximum(np.diff(Vd), 1e-9)
            d2 = np.diff(dlog)
            key = f"VG1_{r['VG1']:.1f}"
            per_vg1_acc.setdefault(key, []).append(float(np.std(d2)))
        per_vg1_mean = {k: float(np.mean(v)) for k, v in per_vg1_acc.items()}
        out[v] = {
            "mean_std_d2logId_dVd2": float(np.mean(s)) if s else None,
            "median_std": float(np.median(s)) if s else None,
            "n": len(s),
            "per_VG1_mean": per_vg1_mean,
        }
    return out


# ============================================================ #
# Plotting
# ============================================================ #

def overlay_variant_plot(VG1_target, results_by_variant, fname):
    """Overlay 3 variants per VG1 — choose 3 representative VG2 values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    def rows_at(per_bias, vg1):
        return {r["VG2"]: r for r in per_bias if abs(r["VG1"] - vg1) < 1e-3}

    baseline_rows = rows_at(results_by_variant["BASELINE"]["per_bias"], VG1_target)
    implicit_rows = rows_at(results_by_variant["IMPLICIT"]["per_bias"], VG1_target)
    bdf2_rows = rows_at(results_by_variant["BDF2"]["per_bias"], VG1_target)
    vg2_vals = sorted(set(baseline_rows) | set(implicit_rows) | set(bdf2_rows))
    if not vg2_vals:
        plt.close(fig); return
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    for ax, vg2 in zip(axes, chosen):
        meas_rec = (baseline_rows.get(vg2) or implicit_rows.get(vg2)
                    or bdf2_rows.get(vg2))
        if meas_rec is None:
            ax.set_title(f"VG2={vg2:.2f} (no data)")
            continue
        ax.plot(meas_rec["Vd"], meas_rec["Id_meas"], "k-", lw=2.5, label="measured")
        if vg2 in baseline_rows:
            r = baseline_rows[vg2]
            ax.plot(r["Vd"], r["Id_pred"], "--", lw=1.5, color="tab:red",
                    label="BASELINE (expl Euler)")
        if vg2 in implicit_rows:
            r = implicit_rows[vg2]
            ax.plot(r["Vd"], r["Id_pred"], "-", lw=1.5, color="tab:blue",
                    label="IMPLICIT (Newton)")
        if vg2 in bdf2_rows:
            r = bdf2_rows[vg2]
            ax.plot(r["Vd"], r["Id_pred"], ":", lw=2.0, color="tab:green",
                    label="BDF2")
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z439 smooth-PT solver comparison (backward sweep) @ VG1={VG1_target:.1f}",
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
    log("z439 starting — smooth pseudo-transient (explicit / implicit / BDF2)")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    target_vg1s = [0.2, 0.4, 0.6]

    # Cell-wide for all 3 variants @ default dt/n_steps
    log("=== Cell-wide backward sweeps (3 variants, default dt/n_steps) ===")
    results = {}
    for name, fn in INTEGRATORS.items():
        log(f"--- variant {name} ---")
        results[name] = run_cellwide(name, fn, model_M1, model_M2,
                                     curves, sebas_rows,
                                     n_steps=N_STEPS_DEFAULT, dt=DT_DEFAULT)

    # HIRES (n_steps=2000, dt/10) restricted to target VG1s for smoothness only
    log("=== HIRES (n_steps=2000, dt/10) limited to VG1 in {0.2,0.4,0.6} ===")
    results_hires = {}
    for name, fn in INTEGRATORS.items():
        log(f"--- HIRES {name} ---")
        results_hires[name] = run_cellwide(f"HIRES_{name}", fn,
                                           model_M1, model_M2,
                                           curves, sebas_rows,
                                           n_steps=2000, dt=DT_DEFAULT/10.0,
                                           limit_vg1=target_vg1s)

    # Smoothness metric: default and HIRES
    smooth_default = smoothness_summary({k: v["per_bias"] for k, v in results.items()})
    smooth_hires = smoothness_summary({k: v["per_bias"] for k, v in results_hires.items()})

    # Overlays per target VG1 (default-resolution)
    for vg1 in target_vg1s:
        fname = OUT / f"overlay_VG1_{str(vg1).replace('.','p')}.png"
        overlay_variant_plot(vg1, results, fname)

    # Summary
    summary = {
        "VARIANTS": {
            name: {
                "cell_rmse_dec": r["cell_rmse_dec"],
                "per_branch_rmse_dec": r["per_branch_rmse_dec"],
                "n_biases_evaluated": r["n_biases_evaluated"],
                "vb_max_overall": r["vb_max_overall"],
                "convergence_rate": r["convergence_rate"],
                "fails": r["fails"],
                "wall_sec": r["wall_sec"],
            } for name, r in results.items()
        },
        "HIRES": {
            name: {
                "cell_rmse_dec": r["cell_rmse_dec"],
                "per_branch_rmse_dec": r["per_branch_rmse_dec"],
                "n_biases_evaluated": r["n_biases_evaluated"],
                "convergence_rate": r["convergence_rate"],
                "wall_sec": r["wall_sec"],
            } for name, r in results_hires.items()
        },
        "REFERENCE": {
            "z432_backward_cell_rmse_dec": 1.026861976331113,
            "z432_forward_cell_rmse_dec": 1.34906163370139,
        },
        "CONFIG": {
            "C_B_F": C_B_DEFAULT, "dt_s": DT_DEFAULT,
            "n_steps_max": N_STEPS_DEFAULT, "tol_dv_V": TOL_DV_DEFAULT,
            "Vsint_pin_V": 0.0, "Vb_clamp": [VB_MIN, VB_MAX],
            "FD_h_for_Jacobian": FD_H, "n_newton_per_step": 2,
        },
        "wall_sec_total": round(time.time() - t_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "smoothness_metric.json").write_text(json.dumps({
        "default_resolution": smooth_default,
        "hires_resolution": smooth_hires,
        "notes": "metric = std of finite-difference second derivative of "
                 "log10(Id) w.r.t. V_D, averaged over biases. "
                 "Lower = smoother (less zig-zag).",
    }, indent=2))

    # Honest analysis
    base_s = smooth_default["BASELINE"]["mean_std_d2logId_dVd2"] or float("inf")
    imp_s = smooth_default["IMPLICIT"]["mean_std_d2logId_dVd2"] or float("inf")
    bdf_s = smooth_default["BDF2"]["mean_std_d2logId_dVd2"] or float("inf")
    rel_imp = (base_s - imp_s) / base_s if base_s > 0 else 0.0
    rel_bdf = (base_s - bdf_s) / base_s if base_s > 0 else 0.0

    base_c = results["BASELINE"]["cell_rmse_dec"]
    imp_c  = results["IMPLICIT"]["cell_rmse_dec"]
    bdf_c  = results["BDF2"]["cell_rmse_dec"]

    discovery = (rel_imp >= 0.5) or (rel_bdf >= 0.5)
    ambitious = ((imp_c < 1.027 and rel_imp >= 0.5) or
                 (bdf_c < 1.027 and rel_bdf >= 0.5))
    kill_shot = (not discovery) and (abs(rel_imp) < 0.1) and (abs(rel_bdf) < 0.1)

    md = []
    md.append(f"# z439 honest analysis — smooth pseudo-transient (S28)\n")
    md.append(f"Wall time: {summary['wall_sec_total']:.0f}s\n\n")
    md.append("## Cell-wide log-RMSE (backward sweep)\n")
    md.append(f"| Variant   | cell_rmse_dec | wall_s | conv_rate | fails |\n")
    md.append(f"|-----------|---------------|--------|-----------|-------|\n")
    for name, r in results.items():
        md.append(f"| {name:9s} | {r['cell_rmse_dec']:.3f}         | "
                  f"{r['wall_sec']:.0f}   | {r['convergence_rate']*100:.1f}%   | "
                  f"{r['fails']}   |\n")
    md.append(f"\nReference: z432 backward cell_rmse = 1.027 dec.\n\n")

    md.append("## Smoothness metric (std of d²log10(Id)/dVd², lower = smoother)\n")
    md.append(f"| Variant   | mean | rel. vs BASELINE |\n")
    md.append(f"|-----------|------|------------------|\n")
    md.append(f"| BASELINE  | {base_s:.3e} | — |\n")
    md.append(f"| IMPLICIT  | {imp_s:.3e} | {rel_imp*100:+.1f}% |\n")
    md.append(f"| BDF2      | {bdf_s:.3e} | {rel_bdf*100:+.1f}% |\n\n")

    md.append("## Pre-registered outcomes\n")
    md.append(f"- DISCOVERY (smoothness improves ≥50% on at least one variant): "
              f"**{'YES' if discovery else 'NO'}**\n")
    md.append(f"- AMBITIOUS (cell < 1.027 AND smoothness ≥50% better): "
              f"**{'YES' if ambitious else 'NO'}**\n")
    md.append(f"- KILL_SHOT (shake is intrinsic, no solver helps — all <10% diff): "
              f"**{'YES' if kill_shot else 'NO'}**\n\n")

    md.append("## Per-VG1 smoothness (default resolution)\n")
    for v in ["BASELINE", "IMPLICIT", "BDF2"]:
        md.append(f"- **{v}**: {smooth_default[v]['per_VG1_mean']}\n")
    md.append("\n## HIRES (n_steps=2000, dt/10) smoothness\n")
    for v in ["BASELINE", "IMPLICIT", "BDF2"]:
        md.append(f"- **{v}**: {smooth_hires[v]['per_VG1_mean']}\n")

    md.append("\n## HIRES cell-rmse (limited VG1)\n")
    for name, r in results_hires.items():
        md.append(f"- {name}: {r['cell_rmse_dec']:.3f} dec  "
                  f"(n_biases={r['n_biases_evaluated']}, wall={r['wall_sec']:.0f}s)\n")

    md.append("\n## Interpretation\n")
    if kill_shot:
        md.append("- All three solvers produce statistically equivalent traces. "
                  "The zig-zag is therefore **not solver noise** — it is a "
                  "real feature of the bistability landscape (warm-started V_B "
                  "lands on different basins of attraction as V_D crosses the "
                  "snapback threshold). Need a different fix (e.g. arc-length "
                  "continuation, finer V_D sweep, basin tracking).\n")
    elif discovery:
        md.append("- Implicit/BDF2 reduced the zig-zag metric by ≥50%, "
                  "confirming the user's intuition that the shake was solver "
                  "noise (explicit-Euler oscillating across the basin "
                  "boundary). Recommend adopting the smoother integrator as "
                  "the default for downstream NS-RAM transient runs.\n")
    else:
        md.append("- Smoothness improved but by <50%. Implicit/BDF2 helps "
                  "marginally, suggesting the shake is a mix of solver "
                  "artifact and genuine basin-of-attraction flips. Need "
                  "additional dampening or finer V_D resolution.\n")

    (OUT / "honest_analysis.md").write_text("".join(md))

    log(f"DONE in {time.time()-t_main:.0f}s. "
        f"BASELINE={base_c:.3f}  IMPL={imp_c:.3f}  BDF2={bdf_c:.3f}  "
        f"smooth_rel: IMPL={rel_imp*100:+.1f}%  BDF2={rel_bdf*100:+.1f}%  "
        f"discovery={discovery} ambitious={ambitious} kill_shot={kill_shot}")


if __name__ == "__main__":
    main()
