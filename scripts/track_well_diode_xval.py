#!/usr/bin/env python3
"""Run track_ngspice_xval (9-bias xval vs ngspice) with BOTH the Tlpe1 fix
(`tlpe1_disable=True`) AND the well_diode fix (`well_diode_mode='ngspice_match'`).

Outputs:
  results/track_well_diode_fix/ngspice_xval.json
  results/track_well_diode_fix/verdict_xval.md
  results/track_well_diode_fix/plot_xval.png
"""
from __future__ import annotations
import os, sys, importlib.util
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

sp = importlib.util.spec_from_file_location("xv", ROOT / "scripts/track_ngspice_xval.py")
xv = importlib.util.module_from_spec(sp); sp.loader.exec_module(xv)

xv.OUT = ROOT / "results/track_well_diode_fix"
xv.DECKS = xv.OUT / "decks_xval"
xv.OUT.mkdir(parents=True, exist_ok=True)
xv.DECKS.mkdir(parents=True, exist_ok=True)

_orig_build = xv.build_pyport_with_hurkx

def build_with_both_fixes(use_hurkx=True):
    cfg, M1, M2, bjt = _orig_build(use_hurkx=use_hurkx)
    M1._values["tlpe1_disable"] = True
    M2._values["tlpe1_disable"] = True
    cfg.well_diode_mode = "ngspice_match"
    return cfg, M1, M2, bjt

xv.build_pyport_with_hurkx = build_with_both_fixes


if __name__ == "__main__":
    rc = xv.main()
    for old, new in [("ablation.json", "ngspice_xval.json"),
                     ("verdict.md",    "verdict_xval.md"),
                     ("plot.png",      "plot_xval.png")]:
        op = xv.OUT / old; np_ = xv.OUT / new
        if op.exists():
            op.replace(np_)
            print(f"  renamed {old} → {new}")
    sys.exit(rc)
