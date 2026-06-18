#!/usr/bin/env python3
"""Track Triode — focused cross-check: k1 fix at VG1=0.6 ONLY.

Findings from sweep:
 - k1 x1.0 (= card 0.53825) → VG1=0.6 triode RMSE 0.428 (PASS, <0.5)
 - k1 x1.2 (= 0.6459)       → VG1=0.6 triode RMSE 0.125 (PASS, <0.5)
   BUT applied globally, destroys VG1=0.2 (Δ+0.94) and VG1=0.4 (Δ+0.94).

Hypothesis: Sebas's CSV K1=0.41825 at VG1=0.6 is the bug (he kept card
0.53825 at VG1=0.4 / VG1=0.2). Setting K1=0.53825 at VG1=0.6 ONLY closes
the gap without touching the other VG1 levels.

This script:
  1. Applies override ONLY at VG1=0.6 (CSV value replaced with card 0.53825).
  2. Re-runs ALL 33 biases (full grid) fwd+bwd → reports per-VG1 medians.
  3. Also tests k1×1.2 = 0.6459 (the optimum).
"""
from __future__ import annotations
import os, sys, json, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/track_triode_vg1_60"; OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("pic3", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pic3 = importlib.util.module_from_spec(sp); sp.loader.exec_module(pic3)
from nsram.bsim4_port.nsram_cell_2T import forward_2t

def run_grid_k1_fix(cfg, M1, M2, bjt, curves, sebas_rows, k1_at_vg06):
    rows = []
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    for c in curves:
        row_sebas, _ = pic3.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
        P_M1, P_M2 = pic3.make_overrides(row_sebas)
        # Apply k1 fix ONLY at VG1=0.6
        if abs(c["VG1"] - 0.6) < 1e-6:
            if P_M1 is None: P_M1 = {}
            P_M1["k1"] = float(k1_at_vg06)
        for branch, vdk, idk in (("fwd","fwd_Vd","fwd_Id"), ("bwd","bwd_Vd","bwd_Id")):
            Vd_np = c[vdk]; Id_np = c[idk]
            Vd = torch.tensor(Vd_np, dtype=torch.float64)
            try:
                with pic3.patch_sd_scaled(sd_M1, P_M1), pic3.patch_sd_scaled(sd_M2, P_M2):
                    out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                                     VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                     VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                                     warm_start=True)
                I_pred = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
                if not np.all(np.isfinite(I_pred)):
                    I_pred = np.where(np.isfinite(I_pred), I_pred, 0.0)
            except Exception:
                I_pred = np.zeros_like(Vd_np)
            res = pic3.log_residuals(Id_np, I_pred, Vd_np, vmin=0.3)
            med_dec = float(np.median(res)) if res.size else float("nan")
            mt = (Vd_np > 0.05) & (Vd_np <= 0.5) & (np.abs(Id_np) > pic3.DEC_FLOOR_MEAS) & (I_pred > 0)
            if mt.sum() >= 3:
                lm = np.log10(np.clip(np.abs(Id_np[mt]), pic3.DEC_FLOOR_MEAS, None))
                lp = np.log10(np.clip(I_pred[mt], pic3.DEC_FLOOR_PRED, None))
                triode_rmse = float(np.sqrt(np.mean((lm - lp)**2)))
            else:
                triode_rmse = float("nan")
            rows.append({"VG1": c["VG1"], "VG2": c["VG2"], "branch": branch,
                         "med_dec": med_dec, "triode_rmse_dec": triode_rmse})
    return rows

def per_vg1(rows):
    out = {}
    for vg1 in (0.2, 0.4, 0.6):
        sub = [r for r in rows if abs(r["VG1"]-vg1)<1e-6 and np.isfinite(r["med_dec"])]
        med = np.array([r["med_dec"] for r in sub])
        tri = np.array([r["triode_rmse_dec"] for r in sub if np.isfinite(r["triode_rmse_dec"])])
        out[f"VG1={vg1}"] = {
            "n": len(sub),
            "med_dec_median": float(np.median(med)) if med.size else float("nan"),
            "triode_rmse_median": float(np.median(tri)) if tri.size else float("nan"),
        }
    all_med = np.array([r["med_dec"] for r in rows if np.isfinite(r["med_dec"])])
    out["ALL"] = {"n": len(all_med),
                  "med_dec_median": float(np.median(all_med)) if all_med.size else float("nan")}
    return out

def main():
    t0 = time.time()
    print(f"[xcheck] start {time.strftime('%H:%M:%S')}", flush=True)
    cfg, M1, M2, bjt = pic3.build_pyport_base()
    sebas_rows = pic3.load_sebas_params()
    all_curves = pic3.load_curves()
    print(f"[xcheck] total curves: {len(all_curves)}", flush=True)

    cases = {
        "baseline (Sebas K1=0.41825 @ VG06)": None,           # no override
        "K1=0.53825 @ VG06 (card)":            0.53825,
        "K1=0.6459 @ VG06 (x1.2)":             0.6459,
        "K1=0.50 @ VG06":                      0.50,
        "K1=0.48 @ VG06":                      0.48,
        "K1=0.46 @ VG06":                      0.46,
    }
    summary = {}
    for label, k1v in cases.items():
        if k1v is None:
            # baseline: pass a k1_at_vg06 that equals what CSV says (0.41825)
            rows = run_grid_k1_fix(cfg, M1, M2, bjt, all_curves, sebas_rows, 0.41825)
        else:
            rows = run_grid_k1_fix(cfg, M1, M2, bjt, all_curves, sebas_rows, k1v)
        s = per_vg1(rows)
        summary[label] = s
        print(f"\n[xcheck] {label}", flush=True)
        for k, v in s.items():
            print(f"   {k}: {v}", flush=True)

    out = OUT / "xcheck_k1_localized.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n[xcheck] wrote {out}", flush=True)
    print(f"[xcheck] done in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
