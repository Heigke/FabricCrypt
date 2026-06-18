"""z313 — pyport_v4: P1 oracle-locked fix-order + P2 free-physics findings.

Strict scope (master fix plan P3):

P1 oracle-locked fix-order (NO REORDER):
  1. VNwell→Vb diode FIX POLARITY: turn OFF cfg.use_well_diode (wrong polarity:
     anode=Vnwell, cathode=Vb) AND turn ON cfg.body_pdiode_to="vnwell" (correct
     polarity: anode=Vb, cathode=Vnwell). z310 patch is NOT installed here.
  2. Distributed R_body via per-V_G1 table (proxy for rbodymod=1):
     R_body[0.2]=1e10, R_body[0.4]=1e9, R_body[0.6]=1e8  (sweepable).
  3. Drain-end avalanche M(V_bc): cfg.use_lateral_collector=True with
     Vbr_av=3.0 (cfg.lat_BV=3.0), N=4.

P2 free-physics findings to ACTIVATE:
  - BSIM4 TAT term between V_Nwell and V_b. The actual M1/M2 cards have
    jtss=0 (default off). The task overrides with the TAT block values
    mentioned in oracle: njts=20, vtss=10, xtss=0.02, jtss=3.4e-7. TAT
    OVERLAPS with #1 — controllable via flags `enable_tat`. We test both:
      RUN_A: P1 fixes only (no TAT)
      RUN_B: P1 fixes + TAT activated

  - Quantitative snapback law (hard gate): V_peak(V_G2) = 2.73 − 0.625·V_G2
    at V_G1=0.3, trise=200µs. We run a DC sweep V_d ∈ [0.5, 3.5] V at
    V_G1=0.3 (interpolated Sebas params between 0.2 and 0.4 rows) for
    several V_G2 ∈ {0.05, 0.1, 0.2, 0.3, 0.5} and locate the V_d at peak I_d.

LOCKED gates (from MASTER_FIX_PLAN):
  - PASS-conservative: cell-wide median log-RMSE < 0.7 dec
  - AMBITIOUS:                                    < 0.5 dec
  - FALSIFY (oracle P1 spec): improvement < 0.5 dec OR V_G1=0.2 bias ≤ -1.0 dec
  - SNAPBACK GATE: |V_peak_sim - (2.73-0.625·V_G2)| < 0.2 V on ≥3 V_G2 points.

Output: results/z313_pyport_v4/summary.json
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
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
OUT_DIR = ROOT / "results/z313_pyport_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# Locked cell from z304/z310 (middle of grid)
BF = 500
ALPHA0 = 1e-4

# Per-V_G1 R_body table (proxy for rbodymod=1 distributed R_body)
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}

# Drain-end avalanche
VBR_AV = 3.0
N_AV = 4.0

# BSIM4 TAT block (oracle values)
TAT_JTSS = 3.4e-7   # A (total TAT saturation current of well-body junction)
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Residual patch: add BSIM4 TAT current between VNwell and Vb                 #
# Activates only when cfg.z313_enable_tat is True.                            #
# TAT current direction: TAT typically generates carriers across a reverse-   #
# biased junction. We treat the well-body junction TAT as INTO Vb when the    #
# junction is forward-biased (Vnwell>Vb), matching well_diode polarity.       #
# But P1 says VNwell→Vb (anode=VN) is WRONG. So TAT here should follow the    #
# CORRECTED polarity: anode=Vb cathode=VN → I_TAT > 0 when Vb>Vnwell leaves   #
# Vb. We implement: I_TAT_leave = jtss·(exp((V_Vb−V_VN)/(njts·Vt))−1) with    #
# vtss/xtss as temperature acceleration (T-dep negligible at 300K → 1.0).    #
# This is "consistent" with P1 (same polarity as body_pdiode_to=vnwell).     #
# --------------------------------------------------------------------------- #
_PATCH_INSTALLED = False
_ORIG_RESIDUALS = None


def install_z313_tat_patch():
    global _PATCH_INSTALLED, _ORIG_RESIDUALS
    if _PATCH_INSTALLED:
        return
    from nsram.bsim4_port import nsram_cell_2T as mod
    _ORIG_RESIDUALS = mod._residuals

    def _residuals_z313(cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb,
                         P_M1=None, P_M2=None, model_M2=None):
        R_Sint, R_B, comps = _ORIG_RESIDUALS(
            cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb,
            P_M1=P_M1, P_M2=P_M2, model_M2=model_M2,
        )
        if getattr(cfg, "z313_enable_tat", False):
            Vt = 0.02585 * (273.15 + cfg.T_C) / 300.0
            jtss = float(getattr(cfg, "z313_tat_jtss", TAT_JTSS))
            njts = float(getattr(cfg, "z313_tat_njts", TAT_NJTS))
            # TAT polarity consistent with P1: anode=Vb, cathode=VNwell.
            arg = ((Vb - cfg.vnwell) / (njts * Vt)).clamp(-40.0, 40.0)
            I_tat = jtss * (torch.exp(arg) - 1.0)
            I_tat = I_tat.clamp(min=-1.0e-2, max=1.0e-2)  # ±10 mA hard ceiling
            # +TAT leaves body
            R_B = R_B - I_tat
            comps["I_z313_tat"] = I_tat
        return R_Sint, R_B, comps

    mod._residuals = _residuals_z313
    _PATCH_INSTALLED = True


def configure_v4_cell(cfg, vg1, enable_tat=False):
    """Apply P1 fixes + P2 TAT for one V_G1 branch."""
    # P1 fix #1: VNwell diode polarity
    cfg.use_well_diode = False                # turn OFF wrong-polarity diode
    cfg.body_pdiode_to = "vnwell"             # turn ON correct-polarity diode

    # P1 fix #2: per-V_G1 R_body (proxy for distributed rbodymod=1)
    cfg.vnwell_Rs = float(R_BODY_TABLE.get(round(vg1, 2), 1.0e9))

    # P1 fix #3: drain-end avalanche M(V_bc)
    cfg.use_lateral_collector = True
    cfg.lat_BV = float(VBR_AV)
    cfg.lat_N = float(N_AV)
    cfg.lat_BV_max = float(VBR_AV * 1.1)
    cfg.lat_M_smooth_delta = 0.5

    # P2: BSIM4 TAT
    cfg.z313_enable_tat = bool(enable_tat)
    cfg.z313_tat_jtss = TAT_JTSS
    cfg.z313_tat_njts = TAT_NJTS

    # Make sure prior z310 hook is OFF if present
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False


# --------------------------------------------------------------------------- #
# Snapback DC sweep at V_G1=0.3                                               #
# --------------------------------------------------------------------------- #
def _interp_sebas_row(rows, vg1_target, vg2):
    """Linear-interpolate sebas params between V_G1=0.2 and V_G1=0.4 at fixed V_G2."""
    r02 = None
    r04 = None
    for r in rows:
        if abs(r["VG2"] - vg2) < 1e-3:
            if abs(r["VG1"] - 0.2) < 1e-3:
                r02 = r
            elif abs(r["VG1"] - 0.4) < 1e-3:
                r04 = r
    if r02 is None or r04 is None:
        return None
    w = (vg1_target - 0.2) / 0.2   # 0.3 → 0.5
    out = {}
    for k in r02.keys():
        v02 = r02.get(k, float("nan"))
        v04 = r04.get(k, float("nan"))
        try:
            out[k] = (1.0 - w) * float(v02) + w * float(v04)
        except (TypeError, ValueError):
            out[k] = float("nan")
    out["VG1"] = vg1_target
    out["VG2"] = vg2
    return out


def snapback_peak_sweep(*, z91f_mod, z304_mod, cfg, M1, M2, sd_M1, sd_M2,
                         forward_2t, sebas_rows, vg1=0.3,
                         vg2_list=(0.05, 0.10, 0.20, 0.30, 0.50),
                         vd_min=0.5, vd_max=3.5, npts=80):
    """Sweep V_d, find V_peak (V_d at I_d max in the [1.5, 3.5] range)."""
    from nsram.bsim4_port.bjt import GummelPoonNPN
    patch_sd_scaled = z91f_mod.patch_sd_scaled

    results = []
    Vd_seq = torch.linspace(vd_min, vd_max, npts, dtype=DTYPE)

    # Use V_G1=0.4 R_body for V_G1=0.3 (closer; mbjt step happens at 0.2→0.4)
    cfg.vnwell_Rs = R_BODY_TABLE[0.4]

    for vg2 in vg2_list:
        sebas_row = _interp_sebas_row(sebas_rows, vg1, vg2)
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            results.append({"VG2": vg2, "V_peak_sim": None, "reason": "no_sebas_row"})
            continue
        P_M1, P_M2 = z304_mod.make_row_overrides(
            sebas_row, ALPHA0, z91f_mod.M2_STATIC_OVERRIDES)
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
        bjt.Bf = float(BF)
        try:
            with torch.no_grad(), \
                  patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, M1, bjt, Vd_seq,
                                  torch.tensor(vg1, dtype=DTYPE),
                                  torch.tensor(vg2, dtype=DTYPE),
                                  warm_start=True, use_homotopy=True,
                                  dense_vd_in_snapback=True,
                                  snapback_vd_threshold=1.4,
                                  snapback_vd_step=0.025)
            Id_pred = out["Id"].abs().cpu().numpy()
            conv = np.array([bool(x) for x in out["converged"]])
        except Exception as e:
            results.append({"VG2": vg2, "V_peak_sim": None,
                              "reason": "solver_fail", "err": str(e)[:120]})
            continue

        Vd_np = Vd_seq.cpu().numpy()
        mask = conv & (Vd_np >= 1.5) & (Vd_np <= vd_max)
        if not mask.any():
            results.append({"VG2": vg2, "V_peak_sim": None,
                              "reason": "no_conv_in_range",
                              "n_conv": int(conv.sum())})
            continue
        # Find first local peak (snapback fold) in masked region
        Id_m = Id_pred[mask]
        Vd_m = Vd_np[mask]
        # peak: argmax. Snapback creates a local max.
        idx_max = int(np.argmax(Id_m))
        V_peak = float(Vd_m[idx_max])
        V_peak_law = 2.73 - 0.625 * vg2
        results.append({
            "VG2": float(vg2),
            "V_peak_sim": V_peak,
            "V_peak_law": float(V_peak_law),
            "delta_V": float(V_peak - V_peak_law),
            "I_at_peak": float(Id_m[idx_max]),
            "n_conv": int(conv.sum()),
            "n_pts": int(npts),
        })
    return results


# --------------------------------------------------------------------------- #
# Slide V_d>2V evaluation (143 samples from O52)                              #
# --------------------------------------------------------------------------- #
def evaluate_slide_samples(*, z91f_mod, z304_mod, cfg, M1, M2, sd_M1, sd_M2,
                            forward_2t, sebas_rows):
    """Evaluate model on O52 slide-extracted measurement points (V_d>=2V)."""
    samples_path = ROOT / "results/z308_slide_v2v_extract/samples.json"
    if not samples_path.exists():
        return {"n": 0, "log_rmse": float("nan"), "note": "samples.json missing"}
    with open(samples_path) as f:
        d = json.load(f)
    from nsram.bsim4_port.bjt import GummelPoonNPN
    patch_sd_scaled = z91f_mod.patch_sd_scaled
    cfg.vnwell_Rs = R_BODY_TABLE[0.4]

    diffs = []
    # slide_15: VG1=0.3, 5 measurement curves (squares). The samples lack
    # VG2 metadata — we map "low/mid-low/mid/mid-high/high" to a guess grid.
    vg2_guess_map = {
        "low VG2": 0.05, "mid‑low VG2": 0.10, "mid VG2": 0.20,
        "mid‑high VG2": 0.30, "high VG2": 0.50,
        "mid-low VG2": 0.10, "mid-high VG2": 0.30,
    }
    n_match = 0
    for curve in d.get("data", {}).get("slide_15", []):
        label = curve.get("curve_label", "")
        if "Measurements" not in label:
            continue
        vg2_guess = None
        for k, v in vg2_guess_map.items():
            if k in label:
                vg2_guess = v
                break
        if vg2_guess is None:
            continue
        sebas_row = _interp_sebas_row(sebas_rows, 0.3, vg2_guess)
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        # Solve at the sample V_d points
        vds = [s[0] for s in curve["samples"]]
        ids = [s[1] for s in curve["samples"]]
        Vd_seq = torch.tensor(vds, dtype=DTYPE)
        P_M1, P_M2 = z304_mod.make_row_overrides(
            sebas_row, ALPHA0, z91f_mod.M2_STATIC_OVERRIDES)
        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area): area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt): mbjt = 1.0
        bjt.area = area * mbjt
        bjt.Bf = float(BF)
        try:
            with torch.no_grad(), \
                  patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, M1, bjt, Vd_seq,
                                  torch.tensor(0.3, dtype=DTYPE),
                                  torch.tensor(vg2_guess, dtype=DTYPE),
                                  warm_start=True, use_homotopy=True,
                                  dense_vd_in_snapback=True)
            Id_pred = out["Id"].abs().cpu().numpy()
            conv = np.array([bool(x) for x in out["converged"]])
        except Exception:
            continue
        ids_np = np.array(ids)
        log_p = np.log10(np.maximum(Id_pred, 1e-15))
        log_m = np.log10(np.maximum(ids_np, 1e-15))
        for i, ok in enumerate(conv):
            if ok and np.isfinite(log_p[i]) and np.isfinite(log_m[i]):
                diffs.append(log_p[i] - log_m[i])
                n_match += 1
    if not diffs:
        return {"n": 0, "log_rmse": float("nan"),
                "note": "no convergent samples"}
    diffs = np.array(diffs)
    return {
        "n": int(len(diffs)),
        "log_rmse": float(np.sqrt((diffs ** 2).mean())),
        "signed_median_dec": float(np.median(diffs)),
        "abs_median_dec": float(np.median(np.abs(diffs))),
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run_dc_cell(*, label, enable_tat,
                z91f_mod, z304_mod, cfg, M1, M2, sd_M1, sd_M2,
                forward_2t, sebas_rows):
    """Run forward DC fit on the 33 IV curves; per-branch + cell-wide stats."""
    per_branch = {}
    all_rmses = []
    for vg1 in [0.2, 0.4, 0.6]:
        configure_v4_cell(cfg, vg1, enable_tat=enable_tat)
        curves = z304_mod.load_curves(vg1_filter=vg1)
        print(f"[z313/{label}] branch V_G1={vg1}: {len(curves)} curves "
              f"R_body={cfg.vnwell_Rs:.0e}", flush=True)
        r = z304_mod.evaluate_cell(
            vg1=vg1, bf=BF, alpha0=ALPHA0, rs=cfg.vnwell_Rs,
            curves=curves, sebas_rows=sebas_rows,
            z91f_mod=z91f_mod, cfg=cfg, M1=M1, M2=M2,
            sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        )
        # NOTE: evaluate_cell internally sets cfg.vnwell_Rs from `rs` arg → we
        # passed our per-branch table value so the body R is preserved.
        per_branch[str(vg1)] = {
            "median_log_rmse": r["median_log_rmse"],
            "signed_dec_median": r["signed_dec_median"],
            "p90_log_rmse": r["p90_log_rmse"],
            "n_finite": r["n_finite"], "n_total": r["n_total"],
            "R_body": cfg.vnwell_Rs,
            "per_curve": r["per_curve"],
        }
        all_rmses.extend([pc["log_rmse"] for pc in r["per_curve"]
                            if math.isfinite(pc["log_rmse"])])
        print(f"[z313/{label}] vg1={vg1}: med={r['median_log_rmse']:.3f} "
              f"signed={r['signed_dec_median']:+.3f} "
              f"n_finite={r['n_finite']}/{r['n_total']}", flush=True)
    cell_wide = float(np.median(all_rmses)) if all_rmses else float("inf")
    return per_branch, cell_wide, all_rmses


def main():
    t0 = time.time()
    print(f"[z313] device={DEVICE}  TAT block: jtss={TAT_JTSS} njts={TAT_NJTS}",
          flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f = _load_module("z91f",
                          ROOT / "scripts/z91f_validate_with_sebas_params.py")
    # Install TAT patch up-front; the flag toggles its activation
    install_z313_tat_patch()

    sebas_rows = z304.load_sebas_params()
    z91f_built, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    print(f"[z313] models built ({time.time()-t0:.1f}s)", flush=True)

    # ----- RUN A: P1 fixes only ------------------------------------------- #
    print("\n[z313] === RUN A: P1 fixes only (TAT disabled) ===", flush=True)
    A_per_branch, A_cell, A_rmses = run_dc_cell(
        label="A", enable_tat=False,
        z91f_mod=z91f_built, z304_mod=z304, cfg=cfg, M1=M1, M2=M2,
        sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        sebas_rows=sebas_rows)

    # ----- RUN B: P1 + TAT ------------------------------------------------ #
    print("\n[z313] === RUN B: P1 fixes + TAT activated ===", flush=True)
    B_per_branch, B_cell, B_rmses = run_dc_cell(
        label="B", enable_tat=True,
        z91f_mod=z91f_built, z304_mod=z304, cfg=cfg, M1=M1, M2=M2,
        sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        sebas_rows=sebas_rows)

    # Choose best for downstream snapback/slide eval
    best_label = "A" if A_cell <= B_cell else "B"
    best_cell = min(A_cell, B_cell)
    print(f"\n[z313] best DC fit: RUN {best_label} (cell-wide={best_cell:.3f})",
          flush=True)

    # ----- Snapback sweep at V_G1=0.3 ------------------------------------ #
    print("\n[z313] === Snapback DC sweep at V_G1=0.3 (best config) ===", flush=True)
    configure_v4_cell(cfg, vg1=0.3, enable_tat=(best_label == "B"))
    snapback_results = snapback_peak_sweep(
        z91f_mod=z91f_built, z304_mod=z304, cfg=cfg, M1=M1, M2=M2,
        sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        sebas_rows=sebas_rows)
    for sr in snapback_results:
        if sr.get("V_peak_sim") is not None:
            print(f"[z313] V_G2={sr['VG2']:.2f}  V_peak_sim={sr['V_peak_sim']:.3f} "
                  f"V  V_peak_law={sr['V_peak_law']:.3f} V  "
                  f"Δ={sr['delta_V']:+.3f} V", flush=True)
        else:
            print(f"[z313] V_G2={sr['VG2']:.2f}  FAIL ({sr.get('reason')})",
                  flush=True)

    # Snapback gate: |ΔV| < 0.2 V on ≥3 V_G2 points
    pass_pts = [s for s in snapback_results
                 if s.get("V_peak_sim") is not None and abs(s["delta_V"]) < 0.2]
    snapback_gate = "PASS" if len(pass_pts) >= 3 else "FAIL"

    # ----- Slide V_d>2V eval --------------------------------------------- #
    print("\n[z313] === Slide V_d>2V samples eval (best config) ===", flush=True)
    slide_eval = evaluate_slide_samples(
        z91f_mod=z91f_built, z304_mod=z304, cfg=cfg, M1=M1, M2=M2,
        sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        sebas_rows=sebas_rows)
    print(f"[z313] slide eval: n={slide_eval.get('n')} "
          f"log_rmse={slide_eval.get('log_rmse'):.3f}" if slide_eval.get("n") else
          f"[z313] slide eval: {slide_eval}", flush=True)

    # ----- Gates ---------------------------------------------------------- #
    Z304_BASELINE = 0.99
    improvement = Z304_BASELINE - best_cell
    falsify_bias_02 = min(
        A_per_branch.get("0.2", {}).get("signed_dec_median", float("inf")),
        B_per_branch.get("0.2", {}).get("signed_dec_median", float("inf")),
    )
    falsify_triggered = (improvement < 0.5) or (falsify_bias_02 <= -1.0)

    if best_cell < 0.5:
        dc_verdict = "AMBITIOUS-PASS"
    elif best_cell < 0.7:
        dc_verdict = "PASS-conservative"
    else:
        dc_verdict = "FAIL"
    if falsify_triggered:
        dc_verdict += "/FALSIFIED"

    summary = {
        "script": "z313_pyport_v4",
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "config": {
            "bf": BF, "alpha0": ALPHA0,
            "R_body_table": R_BODY_TABLE,
            "Vbr_av": VBR_AV, "N_av": N_AV,
            "TAT_jtss": TAT_JTSS, "TAT_njts": TAT_NJTS,
            "TAT_vtss": TAT_VTSS, "TAT_xtss": TAT_XTSS,
        },
        "z304_baseline_median": Z304_BASELINE,
        "run_A_TAT_off": {
            "cell_wide_median_log_rmse": A_cell,
            "per_branch": {k: {kk: vv for kk, vv in v.items() if kk != "per_curve"}
                            for k, v in A_per_branch.items()},
        },
        "run_B_TAT_on": {
            "cell_wide_median_log_rmse": B_cell,
            "per_branch": {k: {kk: vv for kk, vv in v.items() if kk != "per_curve"}
                            for k, v in B_per_branch.items()},
        },
        "best": {
            "label": best_label,
            "cell_wide_median_log_rmse": best_cell,
            "improvement_dec_vs_z304": improvement,
        },
        "snapback_law": "V_peak = 2.73 - 0.625*V_G2  (V_G1=0.3)",
        "snapback_results": snapback_results,
        "snapback_pass_pts": len(pass_pts),
        "snapback_gate": snapback_gate,
        "slide_v2v_eval": slide_eval,
        "gates": {
            "PASS_conservative_<0.7": best_cell < 0.7,
            "AMBITIOUS_<0.5":         best_cell < 0.5,
            "FALSIFY_triggered":      falsify_triggered,
            "SNAPBACK_>=3pts":        snapback_gate == "PASS",
        },
        "dc_verdict": dc_verdict,
        "per_branch_full_A": A_per_branch,
        "per_branch_full_B": B_per_branch,
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z313] best cell-wide = {best_cell:.3f} dec "
          f"(z304={Z304_BASELINE}, Δ={improvement:+.3f})", flush=True)
    print(f"[z313] DC verdict: {dc_verdict}", flush=True)
    print(f"[z313] snapback gate: {snapback_gate} ({len(pass_pts)} of "
          f"{len(snapback_results)} pts within 0.2 V)", flush=True)
    print(f"[z313] wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
