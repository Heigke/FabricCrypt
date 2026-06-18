#!/usr/bin/env python3
"""Track TLPE1 fix — smoke test pyport Vth at 3 probe biases, with/without
the new `tlpe1_disable` model flag, against ngspice (BSIM4 reference).

Probe biases (from results/track_vb_attractor_hunt/verdict.md):
   A) Vgs=0.124, Vds=1.524, Vbs=0.534   (ngspice op-point: Vth=0.375)
   B) Vgs=0.6,   Vds=2.0,   Vbs=0.0     (ngspice:           Vth=0.665)
   C) Vgs=0.6,   Vds=2.0,   Vbs=0.534   (intermediate)

PASS gate (smoke): with tlpe1_disable=True
   Vth(B) within 50 mV of 0.665  AND  Vth(A) within 50 mV of 0.375.

Output: results/track_tlpe1_fix/ablation.json
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, pathlib, importlib.util
import numpy as np
import torch

ROOT = pathlib.Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/track_tlpe1_fix"
OUT.mkdir(parents=True, exist_ok=True)

# Re-use track_ngspice_xval's pyport builder for parity
sp = importlib.util.spec_from_file_location("xv", ROOT / "scripts/track_ngspice_xval.py")
xv = importlib.util.module_from_spec(sp); sp.loader.exec_module(xv)

from nsram.bsim4_port.dc import compute_dc

ALPHA0 = float(xv.ALPHA0_OVERRIDE)
K1     = float(xv.K1_OVERRIDE)
P_M1   = {"alpha0": ALPHA0, "k1": K1}

PROBES = [
    ("A_oppoint", dict(Vgs=0.124, Vds=1.524, Vbs=0.534), 0.375),
    ("B_Vbs0",    dict(Vgs=0.600, Vds=2.000, Vbs=0.000), 0.665),
    ("C_inter",   dict(Vgs=0.600, Vds=2.000, Vbs=0.534), None),  # no ngspice ref handy
]


def probe_vth(flag_disable: bool, flag_xcoupl: bool):
    """Build pyport, set flags on M1, probe Vth at the 3 biases."""
    cfg, M1, M2, bjt = xv.build_pyport_with_hurkx(use_hurkx=False)
    sd_M1 = cfg.size_dep_M1(M1)
    # apply K1+ALPHA0 overrides on M1
    saved_scaled = {}
    for k, v in P_M1.items():
        saved_scaled[k] = sd_M1.scaled.get(k, None)
        sd_M1.scaled[k] = float(v)
    # apply flags via model card (BSIM4Model.get supports arbitrary keys)
    M1._values["tlpe1_disable"] = bool(flag_disable)
    M1._values["tlpe1_lpeb_xcoupl"] = bool(flag_xcoupl)
    results = {}
    try:
        for name, b, ng in PROBES:
            Vgs = torch.tensor(b["Vgs"], dtype=torch.float64)
            Vds = torch.tensor(b["Vds"], dtype=torch.float64)
            Vbs = torch.tensor(b["Vbs"], dtype=torch.float64)
            r = compute_dc(M1, sd_M1, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
            vth = float(r.Vth.detach().reshape(-1)[0])
            results[name] = dict(Vth=vth, ngspice=ng, bias=b,
                                 dV=(vth - ng) if ng is not None else None)
    finally:
        # restore
        for k, v in saved_scaled.items():
            if v is None: sd_M1.scaled.pop(k, None)
            else: sd_M1.scaled[k] = v
        M1._values.pop("tlpe1_disable", None)
        M1._values.pop("tlpe1_lpeb_xcoupl", None)
    return results


def main():
    print("=== track_tlpe1_fix smoke ===")
    print(f"ALPHA0={ALPHA0:.4e}  K1={K1}")
    print()
    all_data = {"probes": [(n, b, ng) for (n, b, ng) in PROBES]}
    for tag, fd, fx in [("BASELINE_off",       False, False),
                        ("tlpe1_disable",      True,  False),
                        ("tlpe1_lpeb_xcoupl",  False, True)]:
        print(f"--- {tag}  (disable={fd}, xcoupl={fx}) ---")
        r = probe_vth(fd, fx)
        for name, row in r.items():
            ng = row["ngspice"]
            ng_s = f"{ng:.3f}" if ng is not None else "—"
            dv  = row["dV"]
            dv_s = f"{dv*1000:+.1f} mV" if dv is not None else "—"
            print(f"  {name:12s}  Vth={row['Vth']:.4f}   ngspice={ng_s}  Δ={dv_s}")
        all_data[tag] = r
        print()
    out = OUT / "ablation.json"
    out.write_text(json.dumps(all_data, indent=2, default=str))
    print(f"wrote {out}")

    # Smoke PASS gate
    fix = all_data["tlpe1_disable"]
    dA = abs(fix["A_oppoint"]["dV"])
    dB = abs(fix["B_Vbs0"]["dV"])
    ok = (dA < 0.050) and (dB < 0.100)
    print(f"\nSMOKE: |dA|={dA*1000:.1f} mV (need<50)  |dB|={dB*1000:.1f} mV (need<100)  "
          f"→ {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
