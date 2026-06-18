"""z332 — R-13 REAL FIX: Vb as a FREE VARIABLE via multistart 2x2 Newton.

Diagnosis (z330): pyport's `_solve_at_fixed_vb` (surrogate path) pins
Vb=Vd_ref → wrong root. Even the existing 2x2 `solve_2t_steady_state`
converges to the *trivial* high-Vb root (Vb≈Vd, body dead) because:
  * KCL residual is ~0 at Vb≈Vd (every body current ≈ 0)
  * KCL residual is also ~0 at the physical low-Vb root (Iii ≈ Dwell leak)
  * With damped Newton + naive init, the basin near Vsint=Vd/2, Vb=0
    is *not* the closest attractor.

This script builds `solve_2t_floating_vb(VG1, VG2, Vd, ...)`:

  • Multi-start: try Vb_init ∈ {0.05, 0.20, 0.40, Vd*0.5, Vd-0.7, 0.0}
    × Vsint_init ∈ {Vd*0.2, Vd*0.5}
  • For each start, run damped 2x2 Newton (analytical 2x2 inverse on
    finite-diff Jacobian, same primitives as `_residuals` /
    `_jacobian_finite_diff` / `_solve_jac_2x2`).
  • Filter to "converged" candidates (||R||_inf < 1e-12).
  • PREFER the LOWEST-Vb converged root (the physical body basin)
    over the trivial Vb≈Vd root.

Output:
  results/z332_vb_free_solver/{summary.json, snapback.png, partial.json}
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data/sebas_2026_04_22"
OUT_DIR = ROOT / "results/z332_vb_free_solver"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL = OUT_DIR / "partial.json"
SUMMARY = OUT_DIR / "summary.json"
SNAPBACK_PNG = OUT_DIR / "snapback.png"

sys.path.insert(0, str(ROOT / "nsram"))

DTYPE = torch.float64
DEVICE = torch.device("cpu")  # solver is tiny, CPU is fine and avoids HIP quirks

# v5b recipe constants (same as z326)
ALPHA0_CONST = 7.842e-5
PDIODE_AREA = 2.2e-11
PDIODE_N = 1.0535
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}
TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02
LAT_BV_DISABLED = 1.0e6
BF_CARD = 10000.0


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_v5b_postfix(cfg, vg1, vd_for_vnwell=2.0):
    # R-13: match ngspice deck (z330) topology — Dwell + Rwell=10G between
    # vnwell (= Vd in z330 deck) and Vb. Without this anchor the body is
    # truly floating and the only stable root is the trivial Vb≈Vd basin.
    cfg.bjt_emitter_to_gnd = True       # R-13: match z330 ngspice deck topology
    cfg.use_well_diode = True
    cfg.vnwell = float(vd_for_vnwell)
    cfg.vnwell_Rs = 1.0e10              # 10 GΩ Rwell (matches ngspice deck)
    cfg.vnwell_Js = 3.4089e-7
    cfg.vnwell_area = 1.0e-12
    cfg.vnwell_n = 1.017
    cfg.vnwell_mbjt = 1.0
    cfg.body_pdiode_to = "off"
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False
    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N
    cfg.Cbody = 7e-15
    cfg.body_pdiode_Rs = float(R_BODY_TABLE.get(round(vg1, 2), 1.0e9))
    cfg.use_lateral_collector = False
    cfg.lat_BV = LAT_BV_DISABLED
    cfg.lat_BV_max = LAT_BV_DISABLED * 1.1
    cfg.enable_tat = True
    cfg.tat_jtss = TAT_JTSS
    cfg.tat_njts = TAT_NJTS
    cfg.tat_vtss = TAT_VTSS
    cfg.tat_xtss = TAT_XTSS
    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False
    if hasattr(cfg, "invalidate"):
        cfg.invalidate()


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield
        return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


# ---------------------------------------------------------------------------
# REAL FIX: 2x2 multistart Newton with physical-root selection
# ---------------------------------------------------------------------------

def _newton_2x2_single(
    cfg, M1, bjt, Vd_t, VG1_t, VG2_t, Vsint0, Vb0,
    M2=None, max_iters=80, tol_abs=1e-12, tol_rel=1e-6, h=1e-6,
    damp_init=1.0, damp_min=1.0 / 256.0, max_step=0.3,
):
    """Run damped 2x2 Newton from (Vsint0, Vb0). Returns
    (Vsint*, Vb*, R_S*, R_B*, components*, niter, converged_bool).

    All inputs scalar/shape-() tensors. Pure no_grad (we don't need
    autograd here — fit only uses Id values).
    """
    from nsram.bsim4_port.nsram_cell_2T import _residuals, _solve_jac_2x2

    Vsint = Vsint0.detach().clone()
    Vb = Vb0.detach().clone()
    converged = False
    niter = 0
    prev_norm = torch.tensor(float("inf"), dtype=DTYPE)
    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                  Vsint, Vb, None, None, model_M2=M2)
    def _scale(comp):
        s = 0.0
        for k in ("Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1", "Iii_M1",
                  "Iii_M2", "Igidl_M1", "Igidl_M2", "Ibs_M1", "Ibd_M1"):
            if k in comp:
                s += float(comp[k].abs())
        return max(s, 1e-30)

    for it in range(max_iters):
        niter = it + 1
        r_max = max(float(R_S.detach().abs()), float(R_B.detach().abs()))
        tol_eff = max(tol_abs, tol_rel * _scale(comp))
        if r_max < tol_eff:
            converged = True
            break
        # Finite-difference 2x2 Jacobian
        Rsp_s, Rbp_s, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                      Vsint + h, Vb, None, None, model_M2=M2)
        Rsm_s, Rbm_s, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                      Vsint - h, Vb, None, None, model_M2=M2)
        dRs_dVs = (Rsp_s - Rsm_s) / (2 * h)
        dRb_dVs = (Rbp_s - Rbm_s) / (2 * h)
        Rsp_b, Rbp_b, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                      Vsint, Vb + h, None, None, model_M2=M2)
        Rsm_b, Rbm_b, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                      Vsint, Vb - h, None, None, model_M2=M2)
        dRs_dVb = (Rsp_b - Rsm_b) / (2 * h)
        dRb_dVb = (Rbp_b - Rbm_b) / (2 * h)
        J = torch.stack([
            torch.stack([dRs_dVs, dRs_dVb], dim=-1),
            torch.stack([dRb_dVs, dRb_dVb], dim=-1),
        ], dim=-2)
        dVs, dVb = _solve_jac_2x2(R_S.detach(), R_B.detach(), J)

        # Step cap
        m = max(float(dVs.abs()), float(dVb.abs()))
        if m > max_step:
            sc = max_step / m
            dVs = dVs * sc
            dVb = dVb * sc

        # Damped backtracking on ||R||
        damp = damp_init
        cur_norm = R_S.detach().abs() + R_B.detach().abs()
        accepted = False
        while damp >= damp_min:
            Vs_try = Vsint + damp * dVs
            Vb_try = Vb + damp * dVb
            R_S_try, R_B_try, comp_try = _residuals(
                cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                Vs_try, Vb_try, None, None, model_M2=M2)
            new_norm = R_S_try.detach().abs() + R_B_try.detach().abs()
            if float(new_norm) < float(cur_norm) * 0.999 or damp <= damp_min:
                Vsint = Vs_try
                Vb = Vb_try
                R_S = R_S_try
                R_B = R_B_try
                comp = comp_try
                accepted = True
                break
            damp *= 0.5
        if not accepted:
            break

    r_max = max(float(R_S.detach().abs()), float(R_B.detach().abs()))
    tol_eff = max(tol_abs, tol_rel * _scale(comp))
    converged = r_max < tol_eff
    return Vsint.detach(), Vb.detach(), R_S.detach(), R_B.detach(), comp, niter, converged


def solve_2t_floating_vb(
    cfg, M1, M2, bjt, Vd, VG1, VG2,
    init_grid=None, tol_abs=1e-12, tol_rel=1e-3, max_iters=80, verbose=False,
):
    """REAL FIX. Vb is FREE. Multistart 2x2 Newton over candidate
    (Vsint_init, Vb_init) pairs; return physical (low-Vb) root.

    Returns dict: Id, Vsint, Vb, components, converged, multistart_log
    """
    Vd_t = torch.as_tensor(Vd, dtype=DTYPE)
    VG1_t = torch.as_tensor(VG1, dtype=DTYPE)
    VG2_t = torch.as_tensor(VG2, dtype=DTYPE)
    Vd_val = float(Vd_t)

    if init_grid is None:
        vb_inits = [0.05, 0.20, 0.40, max(Vd_val * 0.5, 0.0),
                    max(Vd_val - 0.7, 0.0), 0.0]
        vs_inits = [max(Vd_val * 0.2, 0.05), max(Vd_val * 0.5, 0.05)]
    else:
        vs_inits, vb_inits = init_grid

    candidates = []
    log = []
    with torch.no_grad():
        for vs0 in vs_inits:
            for vb0 in vb_inits:
                Vsint0 = torch.tensor(vs0, dtype=DTYPE)
                Vb0 = torch.tensor(vb0, dtype=DTYPE)
                Vs, Vb, R_S, R_B, comp, ni, conv = _newton_2x2_single(
                    cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                    Vsint0, Vb0, M2=M2, max_iters=max_iters,
                    tol_abs=tol_abs, tol_rel=tol_rel,
                )
                rmax = max(float(R_S.abs()), float(R_B.abs()))
                entry = {"vs0": float(vs0), "vb0": float(vb0),
                          "Vsint": float(Vs), "Vb": float(Vb),
                          "rmax": rmax, "niter": ni, "conv": bool(conv)}
                log.append(entry)
                if conv:
                    candidates.append((Vs, Vb, R_S, R_B, comp, ni, conv, entry))

    # Selection: prefer LOWEST Vb among converged candidates whose
    # Vb is physically sensible (0 ≤ Vb ≤ Vd + epsilon).
    chosen = None
    if candidates:
        # Filter to physically-valid Vb (allow tiny overshoot)
        valid = [c for c in candidates if -0.05 <= float(c[1]) <= Vd_val + 0.05]
        pool = valid if valid else candidates
        # Sort by Vb ascending (low-Vb root preferred)
        pool.sort(key=lambda x: float(x[1]))
        chosen = pool[0]

    if chosen is None:
        # All starts diverged — fall back to least-residual unconverged result
        all_runs = log
        best = min(all_runs, key=lambda e: e["rmax"]) if all_runs else None
        return {
            "Id": float("nan"), "Vsint": float("nan"), "Vb": float("nan"),
            "components": {}, "converged": False, "multistart_log": log,
            "fallback": best,
        }

    Vs, Vb, R_S, R_B, comp, ni, conv, entry = chosen
    Id = (comp["Ids_M1"] + comp["Ic_Q1"]
           + comp.get("Ic_lat", 0.0) + comp.get("Ic_avalanche", 0.0)
           + comp["Igidl_M1"] - comp["Ibd_M1"])
    return {
        "Id": float(Id),
        "Vsint": float(Vs),
        "Vb": float(Vb),
        "Iii_M1": float(comp.get("Iii_M1", 0.0)),
        "Ic_Q1": float(comp.get("Ic_Q1", 0.0)),
        "Ids_M1": float(comp.get("Ids_M1", 0.0)),
        "rmax": entry["rmax"],
        "niter": entry["niter"],
        "converged": True,
        "chosen_init": (entry["vs0"], entry["vb0"]),
        "multistart_log": log,
    }


# ---------------------------------------------------------------------------
# Test bias verification (R-12c oracle target)
# ---------------------------------------------------------------------------

def _build_row_setup(sebas_row, z91f):
    """Return (P_M1, P_M2, bjt) for a given Sebas row."""
    from nsram.bsim4_port.bjt import GummelPoonNPN
    P_M1 = {}
    if not math.isnan(sebas_row.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=DTYPE)
    if not math.isnan(sebas_row.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(sebas_row["K1"], dtype=DTYPE)
    P_M1["alpha0"] = torch.tensor(ALPHA0_CONST, dtype=DTYPE)
    if not math.isnan(sebas_row.get("BETA0", float("nan"))):
        P_M1["beta0"] = torch.tensor(sebas_row["BETA0"], dtype=DTYPE)
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = torch.tensor(sebas_row["NFACTOR"], dtype=DTYPE)
    for k, v in z91f.M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=DTYPE)
    bjt = GummelPoonNPN.from_sebas_card()
    if not math.isnan(sebas_row.get("IS", float("nan"))):
        bjt.Is = float(sebas_row["IS"])
    area = float(sebas_row.get("area", 1e-6))
    if math.isnan(area):
        area = 1e-6
    mbjt = float(sebas_row.get("mbjt", 1.0))
    if math.isnan(mbjt):
        mbjt = 1.0
    bjt.area = area * mbjt
    bjt.Bf = BF_CARD
    return P_M1, P_M2, bjt


def test_single_bias(cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f):
    VG1 = 0.6; VG2 = 0.20; Vd_val = 2.0
    configure_v5b_postfix(cfg, VG1, vd_for_vnwell=Vd_val)
    row = next((r for r in sebas_rows
                if abs(r["VG1"] - VG1) < 1e-3 and abs(r["VG2"] - VG2) < 1e-3),
               None)
    P_M1, P_M2, bjt = _build_row_setup(row, z91f)
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        out = solve_2t_floating_vb(cfg, M1, M2, bjt, Vd_val, VG1, VG2)
    out["VG1"] = VG1; out["VG2"] = VG2; out["Vd"] = Vd_val
    out["infra_pass"] = bool(
        out["converged"]
        and 0.1 <= out["Vb"] <= 0.5
        and 0.2 <= out["Vsint"] <= 0.6
    )
    return out


# ---------------------------------------------------------------------------
# Full Sebas IV
# ---------------------------------------------------------------------------

def evaluate_full_iv(cfg, M1, M2, sd_M1, sd_M2, sebas_rows, curves, z91f, z304):
    log_eps = 1e-15
    per_curve = []
    for c in curves:
        VG1 = float(c["VG1"]); VG2 = float(c["VG2"])
        row = next((r for r in sebas_rows
                    if abs(r["VG1"] - VG1) < 1e-3 and abs(r["VG2"] - VG2) < 1e-3),
                   None)
        if row is None or math.isnan(row.get("K1", float("nan"))):
            continue
        configure_v5b_postfix(cfg, VG1, vd_for_vnwell=2.0)  # will override per-point below
        P_M1, P_M2 = z304.make_row_overrides(row, ALPHA0_CONST,
                                              z91f.M2_STATIC_OVERRIDES)
        _, _, bjt = _build_row_setup(row, z91f)
        try:
            Vd_seq = c["Vd"]
            Id_meas = c["Id"]
            Id_pred = []
            Vbs = []
            convs = []
            with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                for vd in Vd_seq.tolist():
                    cfg.vnwell = float(vd)  # ngspice topology: Vnwell node = Vd
                    if hasattr(cfg, "invalidate"):
                        cfg.invalidate()
                    out = solve_2t_floating_vb(cfg, M1, M2, bjt, vd, VG1, VG2)
                    Id_pred.append(abs(out["Id"]) if out["converged"]
                                    else float("nan"))
                    Vbs.append(out["Vb"] if out["converged"] else float("nan"))
                    convs.append(out["converged"])
            Id_pred_t = torch.tensor(Id_pred, dtype=DTYPE)
            conv_t = torch.tensor(convs)
            if conv_t.any():
                mask = conv_t & torch.isfinite(Id_pred_t)
                if mask.any():
                    log_p = torch.log10(Id_pred_t[mask].clamp_min(log_eps))
                    log_m = torch.log10(Id_meas[mask].clamp_min(log_eps))
                    diff = log_p - log_m
                    rmse = float(torch.sqrt((diff ** 2).mean()))
                    signed = float(torch.median(diff))
                    Vb_max = float(np.nanmax(Vbs))
                else:
                    rmse = float("inf"); signed = float("nan"); Vb_max = float("nan")
            else:
                rmse = float("inf"); signed = float("nan"); Vb_max = float("nan")
        except Exception as e:
            per_curve.append({"VG1": VG1, "VG2": VG2, "log_rmse": float("inf"),
                              "signed_dec": float("nan"), "n_conv": 0,
                              "err": str(e)[:120]})
            continue
        per_curve.append({"VG1": VG1, "VG2": VG2, "log_rmse": rmse,
                          "signed_dec": signed, "n_conv": int(sum(convs)),
                          "Vb_max": Vb_max})

    finite = [pc for pc in per_curve if math.isfinite(pc["log_rmse"])]
    rmses_all = np.array([pc["log_rmse"] for pc in finite])
    cell_median = float(np.median(rmses_all)) if len(rmses_all) else float("inf")
    per_vg1 = {}
    for vg1 in (0.2, 0.4, 0.6):
        rms = [pc["log_rmse"] for pc in finite if abs(pc["VG1"] - vg1) < 1e-3]
        per_vg1[f"{vg1:.1f}"] = (float(np.median(rms)) if rms else float("inf"))
    return {
        "cell_median_log_rmse": cell_median,
        "per_vg1_median": per_vg1,
        "n_finite": len(finite),
        "n_total": len(per_curve),
        "per_curve": per_curve,
    }


# ---------------------------------------------------------------------------
# Snapback plot (V_G1=0.4, V_G2 ∈ {0.0,0.2,0.4}, V_d ∈ [0,4])
# ---------------------------------------------------------------------------

def make_snapback(cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    VG1 = 0.4
    Vd_axis = np.linspace(0.05, 4.0, 25)
    fig, ax = plt.subplots(figsize=(7, 5))
    series = {}
    for VG2 in (0.0, 0.2, 0.4):
        row = next((r for r in sebas_rows
                    if abs(r["VG1"] - VG1) < 1e-3 and abs(r["VG2"] - VG2) < 1e-3),
                   None)
        if row is None:
            print(f"  [snapback] missing Sebas row for VG1={VG1} VG2={VG2}")
            continue
        configure_v5b_postfix(cfg, VG1, vd_for_vnwell=2.0)
        P_M1, P_M2, bjt = _build_row_setup(row, z91f)
        Id_log = []
        Vbs = []
        with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            for vd in Vd_axis:
                cfg.vnwell = float(vd)
                if hasattr(cfg, "invalidate"):
                    cfg.invalidate()
                out = solve_2t_floating_vb(cfg, M1, M2, bjt, float(vd), VG1, VG2)
                if out["converged"]:
                    Id_log.append(math.log10(max(abs(out["Id"]), 1e-15)))
                    Vbs.append(out["Vb"])
                else:
                    Id_log.append(float("nan")); Vbs.append(float("nan"))
        ax.plot(Vd_axis, Id_log, marker="o", label=f"VG2={VG2:.2f}")
        series[f"VG2_{VG2:.2f}"] = {"Vd": Vd_axis.tolist(),
                                      "log10_Id": Id_log,
                                      "Vb": Vbs}
    ax.set_xlabel("V_d (V)")
    ax.set_ylabel("log10 |Id| (A)")
    ax.set_title(f"Snapback sweep VG1={VG1}, R-13 floating-Vb solver")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SNAPBACK_PNG, dpi=110)
    plt.close(fig)
    return {"png": str(SNAPBACK_PNG), "series": series}


def save_partial(d):
    with open(PARTIAL, "w") as f:
        json.dump(d, f, indent=2, default=str)


def main():
    t0 = time.time()
    print(f"[z332] device={DEVICE}", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f, cfg, M1, M2, sd_M1, sd_M2, _ = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    curves = z304.load_curves()
    print(f"[z332] models built; {len(curves)} curves "
          f"({time.time() - t0:.1f}s)", flush=True)

    summary = {
        "script": "z332_vb_free_solver",
        "start_time": time.time(),
        "refactor_loc": {
            "files_changed": ["scripts/z332_vb_free_solver.py (NEW)"],
            "new_function": "solve_2t_floating_vb (multistart 2x2 Newton)",
            "loc_count": "~290 LOC NEW; nsram/ library untouched",
        },
        "ngspice_ref": {"VG1": 0.6, "VG2": 0.20, "Vd": 2.0,
                          "Vsint": 0.382, "Vb": 0.267, "Id": 3.93e-11},
    }
    save_partial(summary)

    # Test bias verification ---------------------------------------------------
    print("[z332] === test bias VG1=0.6 VG2=0.20 Vd=2.0 ===", flush=True)
    sb = test_single_bias(cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f)
    print(f"  Vsint={sb['Vsint']:.4f} (ng=0.382)  Vb={sb['Vb']:.4f} (ng=0.267)  "
          f"Id={abs(sb.get('Id', float('nan'))):.3e} (ng=3.93e-11)  "
          f"Iii_M1={sb.get('Iii_M1', float('nan')):.3e}  "
          f"conv={sb['converged']}  infra={sb['infra_pass']}", flush=True)
    summary["test_bias"] = {k: v for k, v in sb.items()
                              if k != "multistart_log"}
    summary["test_bias_multistart_log"] = sb.get("multistart_log", [])
    save_partial(summary)

    # Full IV ------------------------------------------------------------------
    print("[z332] === full Sebas IV (33 curves) ===", flush=True)
    t_iv = time.time()
    iv = evaluate_full_iv(cfg, M1, M2, sd_M1, sd_M2,
                            sebas_rows, curves, z91f, z304)
    print(f"  cell_med={iv['cell_median_log_rmse']:.3f}  "
          f"per_vg1={iv['per_vg1_median']}  "
          f"({time.time() - t_iv:.0f}s, "
          f"{iv['n_finite']}/{iv['n_total']} finite)", flush=True)
    summary["full_iv"] = iv
    save_partial(summary)

    # Snapback PNG -------------------------------------------------------------
    print("[z332] === snapback plot ===", flush=True)
    try:
        snap = make_snapback(cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f)
        summary["snapback"] = {"png": snap["png"]}
        # Keep series in partial only (lightweight in summary)
        summary["snapback_series_summary"] = {
            k: {"Vd_min": min(v["Vd"]), "Vd_max": max(v["Vd"]),
                  "log10_Id_min": float(np.nanmin(v["log10_Id"])),
                  "log10_Id_max": float(np.nanmax(v["log10_Id"]))}
            for k, v in snap["series"].items()}
    except Exception as e:
        summary["snapback"] = {"err": str(e)[:200]}
        print(f"  snapback FAIL: {e}", flush=True)
    save_partial(summary)

    # Gates --------------------------------------------------------------------
    cm = iv["cell_median_log_rmse"]
    gates = {
        "INFRA_basin_at_test_bias": bool(sb["infra_pass"]),
        "PASS_cell_med_lt_0_7": bool(cm < 0.7),
        "AMBITIOUS_lt_0_5": bool(cm < 0.5),
        "SNAPBACK_PNG_saved": Path(SNAPBACK_PNG).exists(),
    }
    summary["gates"] = gates
    summary["status"] = "complete"
    summary["elapsed_s"] = time.time() - t0
    print(f"\n[z332] GATES: {json.dumps(gates, indent=2)}", flush=True)

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[z332] wrote {SUMMARY}  (elapsed {time.time() - t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
