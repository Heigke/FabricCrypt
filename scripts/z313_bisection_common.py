"""Shared helper for z313 bisection variants (b/c/d/e).

Each variant builds the same DC eval pipeline as z313_pyport_v4 but toggles
which P1 elements are active and which R_body table is used.

Variants:
  b: polarity only            (no R_body table, no avalanche)
  c: polarity + avalanche     (no R_body table)
  d: polarity + R_body table  (no avalanche)
  e: polarity + avalanche + REFINED R_body table (reversed gradient)
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
OUT_DIR = ROOT / "results/z313_bisection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

BF = 500
ALPHA0 = 1e-4

# z313 P1 #2 (original) per-V_G1 R_body table
RBODY_Z313 = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}
# z313e refined R_body table (reversed gradient)
RBODY_REFINED = {0.2: 1.0e8, 0.4: 1.0e9, 0.6: 1.0e10}

# Drain-end avalanche
# R-4 (2026-05-13): lowered VBR_AV from 3.0 → 2.0. R-3 audit found that
# at VBR_AV=3.0 with Vd_max ~ 3.0V (bisection grid), `rev_mag = max(Vd-Vb,
# 0)` stays small (Vb tracks Vd via floating body) → M_safe ≈ 1.0 → the
# avalanche path rounds to zero contribution. Lowering BV to 2.0 keeps
# the BV_max ceiling above Vd_max but lets M_safe actually exceed 1 in
# the snapback regime.
VBR_AV = 2.0
N_AV = 4.0

# Fallback when no per-VG1 table applies
RBODY_DEFAULT = 1.0e9


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_variant(cfg, vg1, *, rbody_table, enable_avalanche):
    """Apply variant-specific config for one V_G1 branch.

    Polarity fix (VNwell→Vb) is ALWAYS applied (shared across all variants).
    """
    # --- Polarity fix (always on) ---
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "vnwell"
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False
    # Make sure z313 TAT patch (if installed) is off — bisection covers P1 only.
    cfg.z313_enable_tat = False

    # --- R_body ---
    if rbody_table is None:
        cfg.vnwell_Rs = RBODY_DEFAULT
    else:
        cfg.vnwell_Rs = float(rbody_table.get(round(vg1, 2), RBODY_DEFAULT))

    # --- Drain-end avalanche ---
    if enable_avalanche:
        cfg.use_lateral_collector = True
        cfg.lat_BV = float(VBR_AV)
        cfg.lat_N = float(N_AV)
        cfg.lat_BV_max = float(VBR_AV * 1.1)
        cfg.lat_M_smooth_delta = 0.5
    else:
        cfg.use_lateral_collector = False


def run_variant(label, rbody_table, enable_avalanche):
    """Run one bisection variant; write summary JSON; return summary dict."""
    t0 = time.time()
    print(f"[z313/{label}] device={DEVICE} rbody_table={rbody_table} "
          f"avalanche={enable_avalanche}", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f_path = ROOT / "scripts/z91f_validate_with_sebas_params.py"
    z91f = _load_module("z91f", z91f_path)

    sebas_rows = z304.load_sebas_params()
    z91f_built, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    print(f"[z313/{label}] models built ({time.time()-t0:.1f}s)", flush=True)

    per_branch = {}
    all_rmses = []
    for vg1 in [0.2, 0.4, 0.6]:
        configure_variant(cfg, vg1,
                          rbody_table=rbody_table,
                          enable_avalanche=enable_avalanche)
        curves = z304.load_curves(vg1_filter=vg1)
        print(f"[z313/{label}] branch V_G1={vg1}: {len(curves)} curves "
              f"R_body={cfg.vnwell_Rs:.0e} aval={cfg.use_lateral_collector}",
              flush=True)
        r = z304.evaluate_cell(
            vg1=vg1, bf=BF, alpha0=ALPHA0, rs=cfg.vnwell_Rs,
            curves=curves, sebas_rows=sebas_rows,
            z91f_mod=z91f_built, cfg=cfg, M1=M1, M2=M2,
            sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        )
        per_branch[str(vg1)] = {
            "median_log_rmse": r["median_log_rmse"],
            "signed_dec_median": r["signed_dec_median"],
            "p90_log_rmse": r["p90_log_rmse"],
            "n_finite": r["n_finite"], "n_total": r["n_total"],
            "R_body": cfg.vnwell_Rs,
            "avalanche": bool(enable_avalanche),
        }
        all_rmses.extend([pc["log_rmse"] for pc in r["per_curve"]
                          if math.isfinite(pc["log_rmse"])])
        print(f"[z313/{label}] vg1={vg1}: med={r['median_log_rmse']:.3f} "
              f"signed={r['signed_dec_median']:+.3f} "
              f"n_finite={r['n_finite']}/{r['n_total']}", flush=True)

    cell_wide = float(np.median(all_rmses)) if all_rmses else float("inf")
    Z304_BASELINE = 0.99
    improvement = Z304_BASELINE - cell_wide

    summary = {
        "script": f"z313{label}",
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "config": {
            "bf": BF, "alpha0": ALPHA0,
            "rbody_table": rbody_table,
            "enable_avalanche": bool(enable_avalanche),
            "Vbr_av": VBR_AV if enable_avalanche else None,
            "N_av": N_AV if enable_avalanche else None,
        },
        "z304_baseline_median": Z304_BASELINE,
        "cell_wide_median_log_rmse": cell_wide,
        "improvement_dec_vs_z304": improvement,
        "per_branch": per_branch,
        "gate_lt_0_95": bool(cell_wide < 0.95),
        "gate_PASS_conservative_lt_0_70": bool(cell_wide < 0.70),
        "gate_AMBITIOUS_lt_0_50": bool(cell_wide < 0.50),
    }
    out_path = OUT_DIR / f"{label}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z313/{label}] cell-wide = {cell_wide:.3f} dec  "
          f"(z304={Z304_BASELINE}, Δ={improvement:+.3f})", flush=True)
    print(f"[z313/{label}] wrote {out_path}  ({time.time()-t0:.0f}s)",
          flush=True)
    return summary
