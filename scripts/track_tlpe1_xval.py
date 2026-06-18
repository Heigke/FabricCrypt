#!/usr/bin/env python3
"""Run track_ngspice_xval (9-bias xval vs ngspice) with the new
`tlpe1_disable=True` model flag applied to both M1 and M2.

Outputs (separate directory so the original xval results are preserved):
  results/track_tlpe1_fix/ngspice_xval.json
  results/track_tlpe1_fix/verdict_xval.md
  results/track_tlpe1_fix/plot_xval.png
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

# Load the existing xval module and monkey-patch its OUT/DECKS dirs + the
# pyport builder to enable tlpe1_disable on both transistors.
sp = importlib.util.spec_from_file_location("xv", ROOT / "scripts/track_ngspice_xval.py")
xv = importlib.util.module_from_spec(sp); sp.loader.exec_module(xv)

# Redirect outputs
xv.OUT = ROOT / "results/track_tlpe1_fix"
xv.DECKS = xv.OUT / "decks_xval"
xv.OUT.mkdir(parents=True, exist_ok=True)
xv.DECKS.mkdir(parents=True, exist_ok=True)

# Wrap the builder to set the flag on M1 and M2
_orig_build = xv.build_pyport_with_hurkx

def build_with_tlpe1_disable(use_hurkx=True):
    cfg, M1, M2, bjt = _orig_build(use_hurkx=use_hurkx)
    # Inject flag at model-card level
    M1._values["tlpe1_disable"] = True
    M2._values["tlpe1_disable"] = True
    return cfg, M1, M2, bjt

xv.build_pyport_with_hurkx = build_with_tlpe1_disable


if __name__ == "__main__":
    # Run the existing main()
    rc = xv.main()
    # Rename outputs so they don't collide with original track_tlpe1_fix files
    for old, new in [("ablation.json", "ngspice_xval.json"),
                     ("verdict.md",    "verdict_xval.md"),
                     ("plot.png",      "plot_xval.png")]:
        op = xv.OUT / old; np_ = xv.OUT / new
        if op.exists():
            op.replace(np_)
            print(f"  renamed {old} → {new}")
    sys.exit(rc)
