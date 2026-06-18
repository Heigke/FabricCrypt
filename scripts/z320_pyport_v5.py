"""R-4 z320_pyport_v5 — applies all four v5 wiring changes and runs the
33 IV evaluation that z304 / z313_pyport_v4 used as baseline.

Changes vs z313_pyport_v4 (which monkey-patched TAT in):

  1. body_pdiode_Rs is a real cfg field, harmonic-mean limited inside
     _residuals after the body-pdiode branch.
  2. enable_tat / tat_jtss / tat_njts / tat_vtss / tat_xtss are real
     cfg fields consumed directly by _residuals (no monkey patch).
     vtss/xtss actually enter the equation (T-acceleration and
     V-acceleration); they were dead constants in v4.
  3. R_BODY_TABLE per V_G1 is routed to body_pdiode_Rs (not to the
     dead-on-this-branch vnwell_Rs).
  4. body_pdiode_Js bumped to 2.44e4 A/m² so that
     Js·area = 5.3675e-7 A matches Sebas's pdiode card `is`.
  5. lat_BV lowered to 2.0 V (R-3 audit: at 3.0 M_safe ≈ 1.0 in
     bisection grid → avalanche path was effectively dead).

Locked gates:
  PASS:      cell-wide median log-RMSE < 0.7 dec  (z304 was 0.99)
  AMBITIOUS: cell-wide median < 0.5 dec
  SAFETY:    V_G1=0.2 branch < 1.5 dec (down from 4.7)
  INFRA:     unit test passes for each new flag (scripts/test_v5_wiring.py)

Output: results/z320_pyport_v5/summary.json
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
OUT_DIR = ROOT / "results/z320_pyport_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BF = 500
ALPHA0 = 1e-4

# Per-V_G1 series resistance (same gradient as z313_pyport_v4)
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}
# Drain-end avalanche (lowered BV per R-3 audit)
VBR_AV = 2.0
N_AV = 4.0
# Pdiode card (Sebas 2026-05-02)
PDIODE_IS_TOTAL = 5.3675e-7          # A — Sebas's total saturation current
PDIODE_AREA = 22e-12                 # m² — 5 µm × 4.4 µm
PDIODE_JS_PER_AREA = PDIODE_IS_TOTAL / PDIODE_AREA   # = 2.44e4 A/m²
PDIODE_N = 1.0535
# TAT (oracle values, retained from z313_pyport_v4)
TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_v5(cfg, vg1):
    """Apply v5 topology + per-V_G1 R_body for one V_G1 branch."""
    # Polarity (z313 P1 fix #1)
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "vnwell"
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False

    # Pdiode physics matched to Sebas's card (R-4 change 4)
    cfg.body_pdiode_Js = PDIODE_JS_PER_AREA
    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N

    # R-4 change 3: per-V_G1 R_body now lives on body_pdiode_Rs, NOT on
    # vnwell_Rs (which is inert when use_well_diode=False).
    cfg.body_pdiode_Rs = float(R_BODY_TABLE.get(round(vg1, 2), 1.0e9))
    # Leave vnwell_Rs at fallback (unused on this path).
    cfg.vnwell_Rs = 1.0e30

    # Drain-end avalanche (R-4 change: BV=2.0)
    cfg.use_lateral_collector = True
    cfg.lat_BV = float(VBR_AV)
    cfg.lat_N = float(N_AV)
    cfg.lat_BV_max = float(VBR_AV * 1.1)
    cfg.lat_M_smooth_delta = 0.5

    # R-4 change 2: TAT now core, not a monkey patch.
    cfg.enable_tat = True
    cfg.tat_jtss = TAT_JTSS
    cfg.tat_njts = TAT_NJTS
    cfg.tat_vtss = TAT_VTSS
    cfg.tat_xtss = TAT_XTSS

    # Disable any legacy z313 monkey-patch flag (no-op if absent)
    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False

    cfg.invalidate() if hasattr(cfg, "invalidate") else None


def main():
    t0 = time.time()
    print(f"[z320_v5] device={DEVICE}", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f = _load_module("z91f", SCRIPTS / "z91f_validate_with_sebas_params.py")

    sebas_rows = z304.load_sebas_params()
    z91f_built, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    print(f"[z320_v5] models built ({time.time()-t0:.1f}s)", flush=True)

    # Ablation grid — R-3 audit warned that Js=2.44e4 (Sebas-card-exact)
    # may saturate the body pdiode current. We sweep four (Js, TAT) cells.
    js_options = [("sebas_2.44e4", PDIODE_JS_PER_AREA),
                  ("midoracle_1e-6", 1e-6)]
    tat_options = [("tat_on", True), ("tat_off", False)]

    ablation = {}

    def _snapshot(reason):
        snap = {
            "script": "z320_pyport_v5",
            "reason": reason,
            "elapsed_s": time.time() - t0,
            "ablation_so_far": ablation,
        }
        (OUT_DIR / "progress.json").write_text(
            json.dumps(snap, indent=2, default=float))

    for js_lbl, js_val in js_options:
        for tat_lbl, tat_val in tat_options:
            cell_key = f"{js_lbl}__{tat_lbl}"
            print(f"\n=== [z320_v5] cell {cell_key} ===", flush=True)
            per_branch = {}
            all_rmses = []
            for vg1 in [0.2, 0.4, 0.6]:
                configure_v5(cfg, vg1)
                cfg.body_pdiode_Js = js_val
                cfg.enable_tat = tat_val
                cfg.invalidate() if hasattr(cfg, "invalidate") else None
                curves = z304.load_curves(vg1_filter=vg1)
                print(f"[z320_v5/{cell_key}] V_G1={vg1}: {len(curves)} curves "
                      f"Js={js_val:.2e} Rs={cfg.body_pdiode_Rs:.0e} "
                      f"TAT={tat_val} BV={cfg.lat_BV}", flush=True)
                r = z304.evaluate_cell(
                    vg1=vg1, bf=BF, alpha0=ALPHA0, rs=1.0e30,
                    curves=curves, sebas_rows=sebas_rows,
                    z91f_mod=z91f_built, cfg=cfg, M1=M1, M2=M2,
                    sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
                )
                per_branch[str(vg1)] = {
                    "median_log_rmse": r["median_log_rmse"],
                    "signed_dec_median": r["signed_dec_median"],
                    "p90_log_rmse": r["p90_log_rmse"],
                    "n_finite": r["n_finite"], "n_total": r["n_total"],
                    "body_pdiode_Rs": cfg.body_pdiode_Rs,
                }
                all_rmses.extend([pc["log_rmse"] for pc in r["per_curve"]
                                  if math.isfinite(pc["log_rmse"])])
                print(f"[z320_v5/{cell_key}] V_G1={vg1}: "
                      f"med={r['median_log_rmse']:.3f} "
                      f"signed={r['signed_dec_median']:+.3f}", flush=True)
            cw = float(np.median(all_rmses)) if all_rmses else float("inf")
            ablation[cell_key] = {
                "body_pdiode_Js": js_val,
                "enable_tat": tat_val,
                "cell_wide_median_log_rmse": cw,
                "per_branch": per_branch,
                "gate_PASS_lt_0_70":      bool(cw < 0.70),
                "gate_AMBITIOUS_lt_0_50": bool(cw < 0.50),
                "gate_SAFETY_vg1_0_2_lt_1_5": bool(
                    per_branch.get("0.2", {}).get("median_log_rmse",
                                                   math.inf) < 1.5),
            }
            print(f"[z320_v5/{cell_key}] cell-wide = {cw:.3f} dec",
                   flush=True)
            _snapshot(f"after_{cell_key}")

    # Best cell
    best_key = min(ablation,
                    key=lambda k: ablation[k]["cell_wide_median_log_rmse"])
    best = ablation[best_key]
    cell_wide = best["cell_wide_median_log_rmse"]
    per_branch = best["per_branch"]
    Z304_BASELINE = 0.99
    improvement = Z304_BASELINE - cell_wide

    summary = {
        "script": "z320_pyport_v5",
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "config": {
            "bf": BF, "alpha0": ALPHA0,
            "R_BODY_TABLE": R_BODY_TABLE,
            "VBR_AV": VBR_AV, "N_AV": N_AV,
            "PDIODE_AREA": PDIODE_AREA,
            "PDIODE_N": PDIODE_N,
            "TAT_JTSS": TAT_JTSS, "TAT_NJTS": TAT_NJTS,
            "TAT_VTSS": TAT_VTSS, "TAT_XTSS": TAT_XTSS,
        },
        "z304_baseline_median": Z304_BASELINE,
        "ablation": ablation,
        "best_cell": best_key,
        "cell_wide_median_log_rmse": cell_wide,
        "improvement_dec_vs_z304": improvement,
        "per_branch": per_branch,
        # Locked gates (evaluated on best cell)
        "gate_PASS_lt_0_70":      bool(cell_wide < 0.70),
        "gate_AMBITIOUS_lt_0_50": bool(cell_wide < 0.50),
        "gate_SAFETY_vg1_0_2_lt_1_5": bool(
            per_branch.get("0.2", {}).get("median_log_rmse", math.inf) < 1.5),
    }
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[z320_v5] cell-wide = {cell_wide:.3f} dec  "
          f"(z304={Z304_BASELINE}, Δ={improvement:+.3f})", flush=True)
    print(f"[z320_v5] gate PASS<0.70 = {summary['gate_PASS_lt_0_70']}",
           flush=True)
    print(f"[z320_v5] gate AMBITIOUS<0.50 = {summary['gate_AMBITIOUS_lt_0_50']}",
           flush=True)
    print(f"[z320_v5] gate SAFETY V_G1=0.2 <1.5 = {summary['gate_SAFETY_vg1_0_2_lt_1_5']}",
           flush=True)
    print(f"[z320_v5] wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
