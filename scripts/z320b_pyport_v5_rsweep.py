"""R-4 z320b — Second-pass ablation for v5.

z320a (first pass) showed the audit's recommended R_BODY_TABLE
(1e10/1e9/1e8 per V_G1) HURTS when routed to body_pdiode_Rs:
- It collapses the pdiode forward branch via the harmonic mean limiter,
  removing the body-current path TAT used to set Vb.
- In z313_pyport_v4 the same numbers were on `vnwell_Rs` but THAT path
  was gated off (use_well_diode=False) — so they were inert and the
  pdiode ran unlimited. z313_pyport_v4's "good" 0.79 fit was an
  accident of the dead-flag pattern R-3 identified.

This script sweeps body_pdiode_Rs ∈ {1e30 (off), 1e8, 1e6, 1e4} as a
flat global value (no per-V_G1 table), with TAT on/off. Goal: find a
v5 cell-wide < 0.99 (improvement over z304).
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

BF = 500
ALPHA0 = 1e-4
PDIODE_AREA = 22e-12
PDIODE_JS = 1e-6
PDIODE_N = 1.0535
VBR_AV = 2.0
N_AV = 4.0
TAT = dict(jtss=3.4e-7, njts=20.0, vtss=10.0, xtss=0.02)


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def configure(cfg, *, body_pdiode_Rs, enable_tat):
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "vnwell"
    cfg.body_pdiode_Js = PDIODE_JS
    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N
    cfg.body_pdiode_Rs = float(body_pdiode_Rs)
    cfg.vnwell_Rs = 1.0e30
    cfg.use_lateral_collector = True
    cfg.lat_BV = float(VBR_AV); cfg.lat_N = float(N_AV)
    cfg.lat_BV_max = float(VBR_AV * 1.1); cfg.lat_M_smooth_delta = 0.5
    cfg.enable_tat = bool(enable_tat)
    cfg.tat_jtss = TAT["jtss"]; cfg.tat_njts = TAT["njts"]
    cfg.tat_vtss = TAT["vtss"]; cfg.tat_xtss = TAT["xtss"]
    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False
    cfg.invalidate() if hasattr(cfg, "invalidate") else None


def main():
    t0 = time.time()
    z304 = _load("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f = _load("z91f", SCRIPTS / "z91f_validate_with_sebas_params.py")
    rows = z304.load_sebas_params()
    z91fb, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()

    cells = {}
    rs_options = [("Rs_off_1e30", 1.0e30), ("Rs_1e8", 1.0e8),
                  ("Rs_1e6", 1.0e6), ("Rs_1e4", 1.0e4)]
    tat_options = [("tat_on", True), ("tat_off", False)]

    def _snap():
        (OUT_DIR / "progress_b.json").write_text(
            json.dumps({"elapsed_s": time.time()-t0, "cells": cells},
                        indent=2, default=float))

    for rs_lbl, rs_val in rs_options:
        for tat_lbl, tat_val in tat_options:
            key = f"{rs_lbl}__{tat_lbl}"
            print(f"\n=== [z320b] {key} ===", flush=True)
            pb = {}; all_rmses = []
            for vg1 in [0.2, 0.4, 0.6]:
                configure(cfg, body_pdiode_Rs=rs_val, enable_tat=tat_val)
                curves = z304.load_curves(vg1_filter=vg1)
                print(f"[z320b/{key}] V_G1={vg1}: {len(curves)} curves "
                      f"Rs={rs_val:.0e} TAT={tat_val}", flush=True)
                r = z304.evaluate_cell(
                    vg1=vg1, bf=BF, alpha0=ALPHA0, rs=1.0e30,
                    curves=curves, sebas_rows=rows, z91f_mod=z91fb,
                    cfg=cfg, M1=M1, M2=M2, sd_M1=sd_M1, sd_M2=sd_M2,
                    forward_2t=forward_2t)
                pb[str(vg1)] = {
                    "median_log_rmse": r["median_log_rmse"],
                    "signed_dec_median": r["signed_dec_median"],
                    "p90_log_rmse": r["p90_log_rmse"],
                    "n_finite": r["n_finite"], "n_total": r["n_total"]}
                all_rmses.extend([pc["log_rmse"] for pc in r["per_curve"]
                                  if math.isfinite(pc["log_rmse"])])
                print(f"[z320b/{key}] V_G1={vg1}: "
                      f"med={r['median_log_rmse']:.3f} "
                      f"signed={r['signed_dec_median']:+.3f}", flush=True)
            cw = float(np.median(all_rmses)) if all_rmses else float("inf")
            cells[key] = {
                "body_pdiode_Rs": rs_val, "enable_tat": tat_val,
                "cell_wide_median_log_rmse": cw, "per_branch": pb,
                "gate_PASS_lt_0_70": bool(cw < 0.70),
                "gate_AMBITIOUS_lt_0_50": bool(cw < 0.50),
                "gate_SAFETY_vg1_0_2_lt_1_5": bool(
                    pb.get("0.2", {}).get("median_log_rmse",
                                            math.inf) < 1.5)}
            print(f"[z320b/{key}] cell-wide = {cw:.3f}", flush=True)
            _snap()

    best_key = min(cells, key=lambda k: cells[k]["cell_wide_median_log_rmse"])
    best = cells[best_key]
    summary = {
        "script": "z320b_pyport_v5_rsweep",
        "elapsed_s": time.time() - t0,
        "z304_baseline_median": 0.99,
        "cells": cells,
        "best_cell": best_key,
        "cell_wide_median_log_rmse": best["cell_wide_median_log_rmse"],
        "improvement_dec_vs_z304": 0.99 - best["cell_wide_median_log_rmse"],
        "per_branch": best["per_branch"],
        "gate_PASS_lt_0_70": best["gate_PASS_lt_0_70"],
        "gate_AMBITIOUS_lt_0_50": best["gate_AMBITIOUS_lt_0_50"],
        "gate_SAFETY_vg1_0_2_lt_1_5": best["gate_SAFETY_vg1_0_2_lt_1_5"],
    }
    (OUT_DIR / "summary_b.json").write_text(json.dumps(
        summary, indent=2, default=float))
    print(f"\n[z320b] BEST = {best_key}: cw={best['cell_wide_median_log_rmse']:.3f}",
           flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
