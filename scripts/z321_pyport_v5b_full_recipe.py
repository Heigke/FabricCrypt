"""R-4b z321 — Full R-1 + R-1b recipe applied on R-4 v5 wired infrastructure.

Builds on z320_pyport_v5 (which fixed wiring 5/5 unit tests) and applies
the COMPLETE recipe from R1_zoom_audit.md + R1b_zoom_DEEP.md:

  1. Cb = 7 fF on cfg.Cbody (transient-only; set for completeness)
  2. Adiode = 22 µm² (already in z320)
  3. ALPHA0 = 7.842e-5 CONSTANT across all rows (xlsx finding)
  4. K1 = V_G1-only LUT (already in CSV: 0.55825/0.53825/0.41825) —
     ENFORCED via per-row override (still uses CSV K1, already V_G1-only)
  5. mbjt binary step: 0.001 @ V_G1<=0.2, 1.0 @ V_G1>=0.4 — already in CSV
  6. DROP avalanche/Chynoweth lateral collector path entirely
       cfg.use_lateral_collector = False
       cfg.lat_BV = 1.0e6 (disabled belt+braces)
  7. LDE stress block on M1 only (saref/sbref/ku0/kvth0) — already in M1 card
  8. parasiticBJT Bf swept {50, 200, 500, 1000, 3000}
  9. body_pdiode_Js swept {1e-6, 1e-4, 1e-2, 1.0, 2.44e4}

Locked gates (from MASTER plan):
  PASS:        cell-wide median < 0.7 dec  (beat z304 0.99)
  AMBITIOUS:   < 0.5 dec
  SAFETY:      V_G1=0.2 < 1.5 dec (down from 4.7 catastrophe)

Plus: snapback peak law at V_G1=0.4 sweeping V_G2 — slope sign + 4/6 within 0.3V.

Output: results/z321_pyport_v5b/summary.json
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
OUT_DIR = ROOT / "results/z321_pyport_v5b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# ---- Full-recipe constants (R-1 + R-1b) ----
ALPHA0_CONST = 7.842e-5        # xlsx finding: constant across all 33 rows
CBODY_FF = 7e-15               # 7 fF body junction cap (transient-only)
PDIODE_AREA = 2.2e-11          # 22 µm² = 5 × 4.4 µm
PDIODE_N = 1.0535
PDIODE_IS_TOTAL = 5.3675e-7
PDIODE_JS_SEBAS = PDIODE_IS_TOTAL / PDIODE_AREA  # 2.44e4 A/m²

# Per-V_G1 series resistance (z313/z320 gradient — keep as-is)
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}

# TAT (oracle values, retained from z320)
TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02

# Avalanche/Chynoweth DISABLED (recipe item 6)
LAT_BV_DISABLED = 1.0e6

# Ablation grid
BF_GRID = [50, 200, 500, 1000, 3000]
JS_GRID = [
    ("js_1e-6",   1e-6),
    ("js_1e-4",   1e-4),
    ("js_1e-2",   1e-2),
    ("js_1e0",    1.0),
    ("js_sebas_2.44e4", PDIODE_JS_SEBAS),
]

# Snapback peak law (from z317)
SNAPBACK_VG1 = 0.4
SNAPBACK_VG2_LIST = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]
SNAPBACK_VD_MIN, SNAPBACK_VD_MAX = 0.5, 3.5
SNAPBACK_VD_STEP = 0.05
LAW_INTERCEPT = 2.73
LAW_SLOPE = -0.625


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_v5b(cfg, vg1):
    """Apply FULL recipe (z320 v5 + R-1/R-1b additions) for one V_G1 branch."""
    # ---- Polarity (z313 P1 fix #1) ----
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "vnwell"
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False

    # ---- Pdiode physics (recipe items 2 + matched to Sebas card) ----
    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N

    # Recipe item 1: Cb = 7 fF (transient-only; set on cfg.Cbody)
    cfg.Cbody = CBODY_FF

    # ---- Per-V_G1 R_body on body_pdiode_Rs (z320 wiring) ----
    cfg.body_pdiode_Rs = float(R_BODY_TABLE.get(round(vg1, 2), 1.0e9))
    cfg.vnwell_Rs = 1.0e30

    # ---- Recipe item 6: DISABLE avalanche/Chynoweth lateral collector ----
    cfg.use_lateral_collector = False
    cfg.lat_BV = LAT_BV_DISABLED
    cfg.lat_BV_max = LAT_BV_DISABLED * 1.1
    cfg.lat_N = 4.0
    cfg.lat_M_smooth_delta = 0.5

    # ---- TAT core (R-4 wiring, retained) ----
    cfg.enable_tat = True
    cfg.tat_jtss = TAT_JTSS
    cfg.tat_njts = TAT_NJTS
    cfg.tat_vtss = TAT_VTSS
    cfg.tat_xtss = TAT_XTSS

    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False

    cfg.invalidate() if hasattr(cfg, "invalidate") else None


# ---- Snapback sweep (adapted from z317) ----
@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
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


def sweep_vpeak(*, vg1, vg2, vd_grid, sebas_rows, z91f, cfg, M1, sd_M1, sd_M2,
                 forward_2t, bf):
    from nsram.bsim4_port.bjt import GummelPoonNPN
    # Find closest row
    best = None; bestd = 1e9
    for r in sebas_rows:
        if abs(r["VG1"] - vg1) < 1e-3:
            d = abs(r["VG2"] - vg2)
            if d < bestd:
                bestd = d; best = r
    if best is None or math.isnan(best.get("K1", float("nan"))):
        return {"err": "no_row", "vg2_requested": vg2}

    P_M1 = {}
    if not math.isnan(best.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(best["ETAB"], dtype=DTYPE)
    if not math.isnan(best.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(best["K1"], dtype=DTYPE)
    # ALPHA0 constant (recipe item 3)
    P_M1["alpha0"] = torch.tensor(float(ALPHA0_CONST), dtype=DTYPE)
    if not math.isnan(best.get("BETA0", float("nan"))):
        P_M1["beta0"] = torch.tensor(best["BETA0"], dtype=DTYPE)
    P_M2 = {}
    if not math.isnan(best.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = torch.tensor(best["NFACTOR"], dtype=DTYPE)
    for k, v in z91f.M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=DTYPE)

    bjt = GummelPoonNPN.from_sebas_card()
    if not math.isnan(best.get("IS", float("nan"))):
        bjt.Is = float(best["IS"])
    area = float(best.get("area", 1e-6))
    if math.isnan(area): area = 1e-6
    mbjt = float(best.get("mbjt", 1.0))
    if math.isnan(mbjt): mbjt = 1.0
    bjt.area = area * mbjt
    bjt.Bf = float(bf)

    Vd = torch.tensor(vd_grid, dtype=DTYPE)
    try:
        with torch.no_grad(), \
              patch_sd_scaled(sd_M1, P_M1), \
              patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, M1, bjt,
                              Vd, torch.tensor(vg1), torch.tensor(vg2),
                              warm_start=True, use_homotopy=True,
                              dense_vd_in_snapback=True,
                              snapback_vd_threshold=1.4,
                              snapback_vd_step=0.025)
    except Exception as e:
        return {"err": str(e)[:200], "vg2_requested": vg2}
    Id = out["Id"].abs().cpu().numpy()
    conv = np.array([bool(x) for x in out["converged"]])
    vd_np = Vd.cpu().numpy()
    if conv.any():
        idx_conv = np.where(conv)[0]
        ipk = int(idx_conv[np.argmax(Id[idx_conv])])
    else:
        ipk = int(np.argmax(Id))
    return {
        "vg2_requested": vg2, "row_vg2": float(best["VG2"]),
        "v_peak": float(vd_np[ipk]), "i_peak": float(Id[ipk]),
        "n_conv": int(conv.sum()), "n_total": int(len(vd_np)),
    }


def evaluate_snapback_law(cfg, bf, sebas_rows, z91f, M1, sd_M1, sd_M2, forward_2t):
    configure_v5b(cfg, SNAPBACK_VG1)
    vd_grid = np.round(np.arange(SNAPBACK_VD_MIN,
                                   SNAPBACK_VD_MAX + 1e-9,
                                   SNAPBACK_VD_STEP), 4).tolist()
    res = []
    for vg2 in SNAPBACK_VG2_LIST:
        r = sweep_vpeak(vg1=SNAPBACK_VG1, vg2=vg2, vd_grid=vd_grid,
                         sebas_rows=sebas_rows, z91f=z91f, cfg=cfg,
                         M1=M1, sd_M1=sd_M1, sd_M2=sd_M2,
                         forward_2t=forward_2t, bf=bf)
        res.append(r)
    valid = [(r["vg2_requested"], r["v_peak"]) for r in res if "err" not in r]
    if len(valid) < 2:
        return {"per_vg2": res, "slope": None, "intercept": None,
                "n_within_03": 0, "slope_sign_negative": False}
    vg2_arr = np.array([g for g, _ in valid])
    vp_arr = np.array([v for _, v in valid])
    # Linear fit
    A = np.vstack([vg2_arr, np.ones_like(vg2_arr)]).T
    slope, intercept = np.linalg.lstsq(A, vp_arr, rcond=None)[0]
    # Distance from law, allowing best-fit offset
    law_arr = LAW_INTERCEPT + LAW_SLOPE * vg2_arr
    raw_delta = vp_arr - law_arr
    shift = float(np.median(raw_delta))
    shifted_delta = raw_delta - shift
    return {
        "per_vg2": res,
        "slope": float(slope),
        "intercept": float(intercept),
        "median_shift_vs_law": shift,
        "n_within_03_shifted": int(np.sum(np.abs(shifted_delta) <= 0.3)),
        "n_within_02_shifted": int(np.sum(np.abs(shifted_delta) <= 0.2)),
        "slope_sign_negative": bool(slope < 0),
        "law_4of6_PASS": bool(np.sum(np.abs(shifted_delta) <= 0.3) >= 4),
    }


def main():
    t0 = time.time()
    print(f"[z321_v5b] device={DEVICE}", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f_built, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    print(f"[z321_v5b] models built ({time.time()-t0:.1f}s) "
          f"recipe: ALPHA0={ALPHA0_CONST} Cb={CBODY_FF} "
          f"use_lat_coll=False lat_BV={LAT_BV_DISABLED}", flush=True)

    # Cache curves per V_G1
    curves_by_vg1 = {vg1: z304.load_curves(vg1_filter=vg1) for vg1 in [0.2, 0.4, 0.6]}

    ablation = {}
    t_last_snap = time.time()

    def _snapshot(reason):
        snap = {
            "script": "z321_pyport_v5b_full_recipe",
            "reason": reason,
            "elapsed_s": time.time() - t0,
            "ablation_so_far": ablation,
        }
        (OUT_DIR / "progress.json").write_text(
            json.dumps(snap, indent=2, default=float))

    n_cells = len(BF_GRID) * len(JS_GRID)
    cell_i = 0
    for bf in BF_GRID:
        for js_lbl, js_val in JS_GRID:
            cell_i += 1
            cell_key = f"bf_{bf}__{js_lbl}"
            print(f"\n=== [z321_v5b] cell {cell_i}/{n_cells} {cell_key} ===",
                  flush=True)
            per_branch = {}
            all_rmses = []
            cell_failed = False
            for vg1 in [0.2, 0.4, 0.6]:
                configure_v5b(cfg, vg1)
                cfg.body_pdiode_Js = js_val
                cfg.invalidate() if hasattr(cfg, "invalidate") else None
                curves = curves_by_vg1[vg1]
                t_branch = time.time()
                try:
                    r = z304.evaluate_cell(
                        vg1=vg1, bf=bf, alpha0=ALPHA0_CONST, rs=1.0e30,
                        curves=curves, sebas_rows=sebas_rows,
                        z91f_mod=z91f_built, cfg=cfg, M1=M1, M2=M2,
                        sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
                    )
                except Exception as e:
                    print(f"[z321_v5b/{cell_key}] V_G1={vg1} EXC: {str(e)[:120]}",
                          flush=True)
                    cell_failed = True
                    break
                per_branch[str(vg1)] = {
                    "median_log_rmse": r["median_log_rmse"],
                    "signed_dec_median": r["signed_dec_median"],
                    "p90_log_rmse": r["p90_log_rmse"],
                    "n_finite": r["n_finite"], "n_total": r["n_total"],
                    "body_pdiode_Rs": cfg.body_pdiode_Rs,
                }
                all_rmses.extend([pc["log_rmse"] for pc in r["per_curve"]
                                  if math.isfinite(pc["log_rmse"])])
                print(f"[z321_v5b/{cell_key}] V_G1={vg1}: "
                      f"med={r['median_log_rmse']:.3f} "
                      f"signed={r['signed_dec_median']:+.3f} "
                      f"({time.time()-t_branch:.1f}s)", flush=True)
            cw = float(np.median(all_rmses)) if all_rmses else float("inf")
            ablation[cell_key] = {
                "bf": bf, "body_pdiode_Js": js_val,
                "cell_wide_median_log_rmse": cw,
                "per_branch": per_branch,
                "failed": cell_failed,
                "gate_PASS_lt_0_70":      bool(cw < 0.70),
                "gate_AMBITIOUS_lt_0_50": bool(cw < 0.50),
                "gate_SAFETY_vg1_0_2_lt_1_5": bool(
                    per_branch.get("0.2", {}).get("median_log_rmse",
                                                   math.inf) < 1.5),
            }
            print(f"[z321_v5b/{cell_key}] cell-wide = {cw:.3f} dec  "
                  f"({time.time()-t0:.0f}s total)", flush=True)
            # Snapshot every 30 min
            if time.time() - t_last_snap > 1800:
                _snapshot(f"interim_after_{cell_key}")
                t_last_snap = time.time()
            else:
                _snapshot(f"after_{cell_key}")

    # ---- Best cell ----
    valid_cells = {k: v for k, v in ablation.items()
                   if math.isfinite(v["cell_wide_median_log_rmse"])}
    if not valid_cells:
        print("[z321_v5b] ALL cells failed!", flush=True)
        summary = {"script": "z321_pyport_v5b_full_recipe", "all_failed": True,
                   "ablation": ablation}
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2,
                                                           default=float))
        return 1
    best_key = min(valid_cells,
                    key=lambda k: valid_cells[k]["cell_wide_median_log_rmse"])
    best = ablation[best_key]
    cell_wide = best["cell_wide_median_log_rmse"]
    per_branch = best["per_branch"]
    Z304_BASELINE = 0.99
    Z320_BEST = 2.91  # R-4 v5 with old params (no recipe)

    # ---- Snapback peak law on best cell ----
    print(f"\n[z321_v5b] best={best_key} cw={cell_wide:.3f} → snapback law",
          flush=True)
    cfg.body_pdiode_Js = best["body_pdiode_Js"]
    snapback = evaluate_snapback_law(cfg, best["bf"], sebas_rows,
                                       z91f_built, M1, sd_M1, sd_M2, forward_2t)
    print(f"[z321_v5b] snapback slope={snapback['slope']} "
          f"4of6={snapback.get('law_4of6_PASS')}", flush=True)

    summary = {
        "script": "z321_pyport_v5b_full_recipe",
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "config": {
            "ALPHA0_CONST": ALPHA0_CONST,
            "CBODY_FF": CBODY_FF,
            "PDIODE_AREA": PDIODE_AREA,
            "PDIODE_N": PDIODE_N,
            "R_BODY_TABLE": R_BODY_TABLE,
            "use_lateral_collector": False,
            "lat_BV_disabled": LAT_BV_DISABLED,
            "TAT_JTSS": TAT_JTSS, "TAT_NJTS": TAT_NJTS,
            "TAT_VTSS": TAT_VTSS, "TAT_XTSS": TAT_XTSS,
            "BF_GRID": BF_GRID,
            "JS_GRID": [(l, v) for l, v in JS_GRID],
        },
        "z304_baseline_median": Z304_BASELINE,
        "z320_v5_old_params_median": Z320_BEST,
        "ablation": ablation,
        "best_cell": best_key,
        "best_bf": best["bf"],
        "best_body_pdiode_Js": best["body_pdiode_Js"],
        "cell_wide_median_log_rmse": cell_wide,
        "improvement_dec_vs_z304": Z304_BASELINE - cell_wide,
        "improvement_dec_vs_z320": Z320_BEST - cell_wide,
        "per_branch": per_branch,
        "gate_PASS_lt_0_70":      bool(cell_wide < 0.70),
        "gate_AMBITIOUS_lt_0_50": bool(cell_wide < 0.50),
        "gate_SAFETY_vg1_0_2_lt_1_5": bool(
            per_branch.get("0.2", {}).get("median_log_rmse",
                                            math.inf) < 1.5),
        "snapback_peak_law": snapback,
    }
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[z321_v5b] cell-wide = {cell_wide:.3f} dec "
          f"(z304={Z304_BASELINE}, z320={Z320_BEST}, "
          f"Δz304={Z304_BASELINE-cell_wide:+.3f}, "
          f"Δz320={Z320_BEST-cell_wide:+.3f})", flush=True)
    print(f"[z321_v5b] gate PASS<0.70 = {summary['gate_PASS_lt_0_70']}",
           flush=True)
    print(f"[z321_v5b] gate AMBITIOUS<0.50 = {summary['gate_AMBITIOUS_lt_0_50']}",
           flush=True)
    print(f"[z321_v5b] gate SAFETY V_G1=0.2 <1.5 = "
          f"{summary['gate_SAFETY_vg1_0_2_lt_1_5']}", flush=True)
    print(f"[z321_v5b] wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
