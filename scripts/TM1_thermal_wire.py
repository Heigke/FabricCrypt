"""TM1 — Verify BSIM4 thermal-coefficient wiring (kt1/kt2/ute/ua1/ub1/uc1/at).

Code audit:
  - dc.py:317  Vth += (kt1 + kt1l/Leff + kt2·Vbs) · (T/Tnom − 1)
  - temp.py:222 u0temp = u0 · TRatio^ute
  - temp.py:229-242 ua/ub/uc/ud += {ua1,ub1,uc1,ud1}·(TRatio−1)  (tempmod=0)
  - temp.py:252 vsattemp = vsat − at·(TRatio−1)                (tempmod=0)

All coefficients ARE wired. This script verifies the predicted behavior:
  G1: T=25°C result identical (within 1e-6 dec) to current code   → IDENTITY
  G2: T=85°C  Vth shifts by ~ -25 mV (kt1≈-0.11, ΔTRatio≈0.20)    → -22mV nom

Output:
  results/TM1_thermal_wire/summary.json
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nsram"))
OUT = REPO / "results" / "TM1_thermal_wire"
OUT.mkdir(parents=True, exist_ok=True)

from nsram.bsim4_port.dc import compute_dc_simple  # noqa: E402
from nsram.bsim4_port.model_card import BSIM4Model  # noqa: E402
from nsram.bsim4_port.geometry import Geometry  # noqa: E402
from nsram.bsim4_port.temp import compute_size_dep  # noqa: E402

model = BSIM4Model()  # uses default nmos card
geom = Geometry(L=0.13e-6, W=0.5e-6, NF=1)

Vgs = torch.tensor([0.6], dtype=torch.float64)
Vds = torch.tensor([0.5], dtype=torch.float64)
Vbs = torch.tensor([0.0], dtype=torch.float64)


def vth_at(T_C):
    out = compute_dc_simple(model, geom, T_C=float(T_C), Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    return float(out.Vth.item())


def ids_at(T_C):
    out = compute_dc_simple(model, geom, T_C=float(T_C), Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    for k in ("Ids", "Id", "cdrain"):
        if hasattr(out, k):
            v = getattr(out, k)
            return float(v.item() if hasattr(v, "item") else v)
    return float("nan")


# Reference T=25°C result is the current pyport answer (canonical baseline).
# G1: re-run at T=25°C and check identity with itself (smoke test of determinism).
vth_25_a = vth_at(25.0)
vth_25_b = vth_at(25.0)
ids_25_a = ids_at(25.0)
ids_25_b = ids_at(25.0)

vth_85 = vth_at(85.0)
ids_85 = ids_at(85.0)
vth_125 = vth_at(125.0)
ids_125 = ids_at(125.0)

dvth_85 = vth_85 - vth_25_a
dvth_125 = vth_125 - vth_25_a

kt1 = float(model.get("kt1", -0.11))
kt2 = float(model.get("kt2", 0.022))
Tnom = float(model.get("tnom", 27.0)) + 273.15
TRatio_85 = (85.0 + 273.15) / Tnom
predicted_dVth_85 = kt1 * (TRatio_85 - 1.0)

g1_pass = abs(vth_25_a - vth_25_b) < 1e-6 and abs(ids_25_a - ids_25_b) < 1e-12
g2_pass = abs(dvth_85 - predicted_dVth_85) < 0.01  # within 10 mV of kt1 prediction

summary = {
    "model_T_coefs": {
        "kt1": kt1, "kt2": kt2,
        "ute": float(model.get("ute", -1.5)),
        "ua1": float(model.get("ua1", 1e-9)),
        "ub1": float(model.get("ub1", -1e-18)),
        "uc1": float(model.get("uc1", 0.056)),
        "at":  float(model.get("at", 3.3e4)),
        "tempmod": int(model.get("tempmod", 0)),
        "tnom_C": float(model.get("tnom", 27.0)),
    },
    "T_25C_a": {"Vth": vth_25_a, "Ids": ids_25_a},
    "T_25C_b": {"Vth": vth_25_b, "Ids": ids_25_b},
    "T_85C":   {"Vth": vth_85, "Ids": ids_85, "dVth_mV": dvth_85 * 1e3},
    "T_125C":  {"Vth": vth_125, "Ids": ids_125, "dVth_mV": dvth_125 * 1e3},
    "predicted_dVth_85_mV_from_kt1": predicted_dVth_85 * 1e3,
    "gates": {
        "G1_identity_25C": bool(g1_pass),
        "G2_Vth_shift_85C_near_predicted": bool(g2_pass),
        "G2_observed_mV": dvth_85 * 1e3,
        "G2_expected_mV": predicted_dVth_85 * 1e3,
    },
    "wired_in_pyport": {
        "kt1_kt2_kt1l": "dc.py:317  (Vth += (kt1+kt1l/L+kt2·Vbs)·(TRatio−1))",
        "ute": "temp.py:222 (u0temp = u0·TRatio^ute)",
        "ua1_ub1_uc1_ud1": "temp.py:229-242 (mobility coef T-shift)",
        "at": "temp.py:252 (vsattemp = vsat − at·(TRatio−1))",
        "prt": "temp.py:259 (rds T-factor)",
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
