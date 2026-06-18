"""z309 — rbodymod=1 falsification test.

Gemini (HIGH) / gpt-5 (MED): floating-body BJT-feedback NS-RAM cell needs
BSIM4 rbodymod=1 (distributed body resistance), not rbodymod=0. z304 used 0.
Cheap test: flip flag in M1+M2 cards, re-run sweep, compare cell-wide median
and the V_G1=0.2 vs V_G1=0.6 Rs incompatibility.

Sweep:
  Bf  ∈ {500, 1000, 3000, 9000}
  Rs  ∈ {0, 1e9, 1e10}
  V_G1 ∈ {0.2, 0.4, 0.6}
  → 12 cells per branch × 3 = 36 cells total
  alpha0 fixed at 1e-4 (z304 showed weak alpha0 sensitivity in best cells).

Gates:
  PASS-conservative : at least one cell achieves cell-wide median < 0.9 dec
                      (improvement ≥0.1 dec over z304's 0.99 baseline)
  AMBITIOUS         : < 0.5 dec  (falsify topology-mandatory)
  DIAGNOSTIC        : best-Rs at V_G1=0.2 vs V_G1=0.6, gap < 1 order

NOTE: rbodymod is parsed (model_card_data.py:573) but no evaluator code path
in nsram.bsim4_port uses it. Expected outcome: medians match z304 within
numerical noise. This run formally falsifies the oracle's structural claim.
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
DATA = ROOT / "data/sebas_2026_04_22"
OUT_DIR = ROOT / "results/z309_rbodymod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# Reuse z304 helpers via spec_from_file_location
def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

z304 = _load("z304", ROOT / "scripts/z304_sebas_three_branch_refit.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

BF_GRID = [500, 1000, 3000, 9000]
RS_GRID = [0, 1.0e9, 1.0e10]
VG1_LIST = [0.2, 0.4, 0.6]
ALPHA0 = 1.0e-4


def build_models_rbodymod1():
    """Same as z304.build_models_once but patches rbodymod=0→1 in card text."""
    z91f = _load("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
    from nsram.bsim4_port.temp import compute_size_dep
    from nsram.bsim4_port.geometry import Geometry

    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M1_patched = text_M1.replace("rbodymod = 0", "rbodymod = 1")
    assert "rbodymod = 1" in text_M1_patched, "M1 patch failed"
    M1 = BSIM4Model.from_spice(text_M1_patched, model_type="nmos")
    z91f.patch_model_values(M1, type_n=True)

    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    text_M2_patched = text_M2.replace("rbodymod = 0", "rbodymod = 1")
    assert "rbodymod = 1" in text_M2_patched, "M2 patch failed"
    M2 = BSIM4Model.from_spice(text_M2_patched, model_type="nmos")
    z91f.patch_model_values(M2, type_n=True)

    print(f"[z309] M1.rbodymod={getattr(M1,'rbodymod','?')} "
          f"M2.rbodymod={getattr(M2,'rbodymod','?')}", flush=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                              newton_max_iters=50)
    sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t


def main():
    t0 = time.time()
    print(f"[z309] device={DEVICE} start {time.strftime('%H:%M:%S')}", flush=True)

    sebas_rows = z304.load_sebas_params()
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = build_models_rbodymod1()
    print(f"[z309] models built ({time.time()-t0:.1f}s)", flush=True)

    curves_per_branch = {vg: z304.load_curves(vg1_filter=vg) for vg in VG1_LIST}
    for vg, cs in curves_per_branch.items():
        print(f"[z309] branch V_G1={vg}: {len(cs)} curves", flush=True)

    rows = []
    n_cells = len(VG1_LIST) * len(BF_GRID) * len(RS_GRID)
    ci = 0
    for vg1 in VG1_LIST:
        curves = curves_per_branch[vg1]
        for bf in BF_GRID:
            for rs in RS_GRID:
                ci += 1
                tc = time.time()
                r = z304.evaluate_cell(
                    vg1=vg1, bf=bf, alpha0=ALPHA0, rs=rs,
                    curves=curves, sebas_rows=sebas_rows,
                    z91f_mod=z91f, cfg=cfg, M1=M1, M2=M2,
                    sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
                )
                rows.append(r)
                print(f"[z309] cell {ci}/{n_cells}: vg1={vg1} bf={bf} "
                      f"Rs={rs:g} → med={r['median_log_rmse']:.3f} "
                      f"signed={r['signed_dec_median']:+.3f} "
                      f"({time.time()-tc:.1f}s, total {time.time()-t0:.0f}s)",
                      flush=True)

    # Aggregate: per cell (bf, rs) compute cell-wide median across 3 branches.
    by_cell = {}
    for r in rows:
        k = (r["bf"], r["rs"])
        by_cell.setdefault(k, {})[r["vg1"]] = r["median_log_rmse"]

    cell_summary = []
    for (bf, rs), branches in sorted(by_cell.items()):
        meds = [branches.get(v, float("inf")) for v in VG1_LIST]
        finite_meds = [m for m in meds if math.isfinite(m)]
        cell_summary.append({
            "bf": bf, "rs": rs,
            "vg1_02_med": branches.get(0.2, float("inf")),
            "vg1_04_med": branches.get(0.4, float("inf")),
            "vg1_06_med": branches.get(0.6, float("inf")),
            "cell_wide_median": (float(np.median(finite_meds))
                                  if finite_meds else float("inf")),
            "cell_wide_worst": (float(max(finite_meds))
                                 if finite_meds else float("inf")),
        })

    cell_summary.sort(key=lambda d: d["cell_wide_median"])
    best = cell_summary[0] if cell_summary else None

    # Diagnostic: per V_G1, which Rs gives best median (at any Bf)?
    rs_pref = {}
    for vg in VG1_LIST:
        per_rs = {}
        for r in rows:
            if abs(r["vg1"] - vg) > 1e-3:
                continue
            rs = r["rs"]
            per_rs.setdefault(rs, []).append(r["median_log_rmse"])
        per_rs_med = {rs: float(np.median(ms)) for rs, ms in per_rs.items()}
        best_rs = min(per_rs_med, key=per_rs_med.get)
        rs_pref[str(vg)] = {"best_rs": best_rs,
                              "per_rs_median": per_rs_med}

    # Compare to z304 baseline
    z304_summary_path = ROOT / "results/z304_sebas_refit/summary.json"
    z304_baseline = None
    if z304_summary_path.exists():
        z304_d = json.load(open(z304_summary_path))
        z304_baseline = {
            "best_cellwide_compromise": z304_d.get("best_cellwide_compromise"),
            "gates": z304_d.get("gates"),
        }

    # Gates
    gates = {
        "PASS_conservative": bool(best and best["cell_wide_median"] < 0.9),
        "AMBITIOUS":         bool(best and best["cell_wide_median"] < 0.5),
    }
    # Diagnostic: V_G1=0.2 best_rs vs V_G1=0.6 best_rs; if same, gap is 0.
    rs_02 = rs_pref["0.2"]["best_rs"]; rs_06 = rs_pref["0.6"]["best_rs"]
    # Rs "order" gap: log10 of ratio, treat 0 as 1.
    def _ord(x): return math.log10(max(x, 1.0))
    rs_gap = abs(_ord(rs_02) - _ord(rs_06))
    gates["DIAGNOSTIC_rs_split_closes"] = rs_gap < 1.0

    summary = {
        "script": "z309_rbodymod_test",
        "rbodymod_flipped": "0->1 in M1 and M2",
        "bf_grid": BF_GRID, "rs_grid": RS_GRID,
        "alpha0_fixed": ALPHA0, "vg1_list": VG1_LIST,
        "n_cells": len(cell_summary), "n_rows": len(rows),
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "best_cell": best,
        "top5_cells": cell_summary[:5],
        "rs_preference_by_vg1": rs_pref,
        "rs_gap_dec_orders": rs_gap,
        "gates": gates,
        "z304_baseline": z304_baseline,
        "all_cells": cell_summary,
        "rows": rows,
    }
    out = OUT_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z309] wrote {out} ({time.time()-t0:.0f}s total)", flush=True)
    print(f"[z309] best cell-wide median: {best['cell_wide_median']:.3f} "
          f"(bf={best['bf']} rs={best['rs']:g})", flush=True)
    print(f"[z309] gates: {gates}", flush=True)
    print(f"[z309] rs_pref: 0.2→{rs_02:g}  0.6→{rs_06:g}  gap={rs_gap:.2f} orders",
          flush=True)


if __name__ == "__main__":
    main()
