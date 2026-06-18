"""z310 — Minimal pyport_v3: add VNwell→VB diode INTO body KCL.

Strict scope (per task spec 2026-05-13):
  Adds ONE extra current term to Vb-node KCL residual:
      I_diode = Is * (exp((V_VN - V_VB) / (n * Vt)) - 1)        (INTO body)
  with Is=1e-13, n=1.5.

  Reuses z304's model build and curve loader. NO sweep. Single fixed cell:
  Bf=500, alpha0=1e-4, Rs disabled (vnwell_Rs=1e30).

  Compares to z304 baseline 0.99 dec cell-wide median.
  PASS-conservative gate: cell-wide median < 0.85 dec.

Output: results/z310_pyport_v3_minimal/summary.json
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
OUT_DIR = ROOT / "results/z310_pyport_v3_minimal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make src/ importable for any v2-style helpers if needed
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# Fixed minimal cell — reasonable middle of z304 grid
BF = 500
ALPHA0 = 1e-4
RS = 0           # 0 → disabled (vnwell_Rs = 1e30)
RS_FALLBACK = 1.0e30

# z310 diode params (locked per task)
DIODE_IS = 1.0e-13
DIODE_N  = 1.5


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Residual patch: add VN→Vb diode INTO body                                   #
# --------------------------------------------------------------------------- #
_PATCH_INSTALLED = False
_ORIG_RESIDUALS = None


def install_z310_diode_patch():
    global _PATCH_INSTALLED, _ORIG_RESIDUALS
    if _PATCH_INSTALLED:
        return
    from nsram.bsim4_port import nsram_cell_2T as mod
    _ORIG_RESIDUALS = mod._residuals

    def _residuals_z310(cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb,
                         P_M1=None, P_M2=None, model_M2=None):
        R_Sint, R_B, comps = _ORIG_RESIDUALS(
            cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb,
            P_M1=P_M1, P_M2=P_M2, model_M2=model_M2,
        )
        if getattr(cfg, "z310_enable_vnwell_diode", False):
            Vt = 0.02585 * (273.15 + cfg.T_C) / 300.0
            Is = float(getattr(cfg, "z310_diode_Is", DIODE_IS))
            n  = float(getattr(cfg, "z310_diode_n",  DIODE_N))
            arg = ((cfg.vnwell - Vb) / (n * Vt)).clamp(max=40.0)
            I_d = Is * (torch.exp(arg) - 1.0)
            I_d = I_d.clamp(min=-1.0e-1, max=1.0e-1)  # ±100 mA hard ceiling
            # Current flows FROM vnwell INTO body → +R_B (INTO body, same sign
            # convention as existing forward vnwell coupling)
            I_d = I_d * float(getattr(cfg, "vnwell_mbjt", 1.0))
            R_B = R_B + I_d
            comps["I_z310_vn_to_vb"] = I_d
        return R_Sint, R_B, comps

    mod._residuals = _residuals_z310
    _PATCH_INSTALLED = True


def enable_z310(cfg, Is=DIODE_IS, n=DIODE_N):
    install_z310_diode_patch()
    cfg.z310_enable_vnwell_diode = True
    cfg.z310_diode_Is = float(Is)
    cfg.z310_diode_n  = float(n)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"[z310] device={DEVICE}  Is={DIODE_IS}  n={DIODE_N}", flush=True)

    # Reuse z304 helpers
    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")

    sebas_rows = z304.load_sebas_params()
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    print(f"[z310] models built  ({time.time()-t0:.1f}s)", flush=True)

    # Enable our extra diode
    enable_z310(cfg, Is=DIODE_IS, n=DIODE_N)
    print(f"[z310] z310 diode installed; cfg.vnwell={cfg.vnwell}", flush=True)

    # Set Rs and evaluate one cell per V_G1 branch
    cfg.vnwell_Rs = RS_FALLBACK if RS == 0 else float(RS)

    per_branch = {}
    all_log_rmses = []
    for vg1 in [0.2, 0.4, 0.6]:
        curves = z304.load_curves(vg1_filter=vg1)
        print(f"[z310] branch V_G1={vg1}: {len(curves)} curves", flush=True)
        r = z304.evaluate_cell(
            vg1=vg1, bf=BF, alpha0=ALPHA0, rs=RS,
            curves=curves, sebas_rows=sebas_rows,
            z91f_mod=z91f, cfg=cfg, M1=M1, M2=M2,
            sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        )
        per_branch[str(vg1)] = {
            "median_log_rmse": r["median_log_rmse"],
            "signed_dec_median": r["signed_dec_median"],
            "p90_log_rmse": r["p90_log_rmse"],
            "n_finite": r["n_finite"], "n_total": r["n_total"],
            "per_curve": r["per_curve"],
        }
        rmses = [pc["log_rmse"] for pc in r["per_curve"]
                 if math.isfinite(pc["log_rmse"])]
        all_log_rmses.extend(rmses)
        print(f"[z310] vg1={vg1}: med={r['median_log_rmse']:.3f} "
              f"signed={r['signed_dec_median']:+.3f} "
              f"n_finite={r['n_finite']}/{r['n_total']}", flush=True)

    cell_wide_median = float(np.median(all_log_rmses)) if all_log_rmses else float("inf")

    # z304 baseline reference
    Z304_BASELINE = 0.99
    GATE_PASS = 0.85
    improvement = Z304_BASELINE - cell_wide_median
    verdict = ("PASS-conservative" if cell_wide_median < GATE_PASS
               else "FAIL")

    summary = {
        "script": "z310_pyport_v3_minimal",
        "diode_Is": DIODE_IS, "diode_n": DIODE_N,
        "bf": BF, "alpha0": ALPHA0, "rs": RS,
        "vnwell_Rs": cfg.vnwell_Rs,
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "n_curves_total": len(all_log_rmses),
        "cell_wide_median_log_rmse": cell_wide_median,
        "z304_baseline_median": Z304_BASELINE,
        "improvement_dec": improvement,
        "gate_pass_threshold": GATE_PASS,
        "verdict": verdict,
        "per_branch": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_curve"}
            for k, v in per_branch.items()
        },
        "per_branch_full": per_branch,
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z310] cell-wide median = {cell_wide_median:.3f} dec "
          f"(vs z304 baseline {Z304_BASELINE})  Δ={improvement:+.3f}", flush=True)
    print(f"[z310] verdict: {verdict}", flush=True)
    print(f"[z310] wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
