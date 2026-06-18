"""z451 — Capacitance audit on M1 floating-body node.

Compute analytical body-node C_eff from loaded M1/M2 model cards + parasitic NPN
+ deep-N-well diode. Compare against z448's "12 fF" KILL_SHOT claim.

Bias point: V_DS=2 V, V_GS=0.6 V, V_BS=0.7 V (body charged near turn-on).
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path

# ---------- Constants ----------
EPS0 = 8.8541878128e-12        # F/m
EPS_SI = 11.7 * EPS0
EPS_OX = 3.9 * EPS0
Q = 1.602176634e-19
KB = 1.380649e-23
NI = 1.07e16                   # cm^-3 at 300K (used only for VBI sanity)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
RES = ROOT / "results" / "z451_cap_audit"
RES.mkdir(parents=True, exist_ok=True)

# ---------- Hard-coded from Sebas M1_130DNWFB.txt + parasiticBJT.txt ----------
# (Avoiding model_card.py to keep audit self-contained / independent.)

# M1 NMOS bulk-source junction (the floating-body diode in deep-N-well device)
# Note: BSIM4 only has CJS/CJSWS for SOURCE-bulk; SAME params used for D-bulk via cjd defaulted to cjs.
CJS = 0.0016995            # F/m^2  bottom-junction cap density
CJSWS = 2.9299e-11         # F/m   sidewall non-gate
CJSWGS = 2.677e-10         # F/m   sidewall gate-edge
MJS = 0.51829
MJSWS = 0.57223
MJSWGS = 0.50288
PBS = 0.74883
PBSWS = 0.6836
PBSWGS = 0.70856

# Overlap caps
CGSO = 3.65e-10            # F/m
CGDO = 3.65e-10
CGBO = 0.0

# Oxide
TOXE = 4e-9
COXE = EPS_OX / TOXE       # F/m^2

# Parasitic NPN (Pazos parasiticBJT.txt)
CJE_BJT = 0.7e-15           # F  emitter junction cap (zero-bias)
CJC_BJT = 1.0e-15           # F  collector junction cap (zero-bias)
VJE_BJT = 0.7               # V
FC_BJT = 0.5
MJE_BJT = 0.33              # default
MJC_BJT = 0.33

# Geometry (M1)
W = 360e-9
L = 180e-9
AS_AREA = W * L             # source area (no S/D extension specified) ≈ 6.48e-14 m^2
AD_AREA = W * L
PS = 2.0 * (W + L)          # source perimeter
PD = 2.0 * (W + L)

# Deep N-well diode: assume nwell area ≥ device active area; reverse-biased
# at V(nwell)=2 V, V(body)~0.7 V → V_R ≈ 1.3 V across n-well-to-pwell junction.
# Use a TYPICAL 130nm CMOS nwell-to-substrate cap density.
# Literature: nwell-to-psub Cj0 ≈ 1e-4 F/m^2, Mj ≈ 0.4, PB ≈ 0.7 V
NWELL_CJ0 = 1.0e-4
NWELL_MJ = 0.4
NWELL_PB = 0.7
# Reasonable nwell footprint for an isolated 360n x 180n DNW device:
# typically ~5-10x device area (DRC keep-out). Try 5x.
NWELL_AREA = 5.0 * W * L      # m^2 (5x device area)


# ---------- Helpers ----------
def cj_reverse(cj0_per_area, area, vbi, mj, v_reverse):
    """Junction cap at reverse bias V_R (>0 means reverse). C = Cj0 / (1 + V_R/PB)^MJ"""
    return cj0_per_area * area / (1.0 + v_reverse / vbi) ** mj


def cj_forward(cj0_per_area, area, vbi, mj, v_forward, fc=0.5):
    """BSIM4-style: for V_f < FC*PB use depletion formula with positive V; beyond, linear extrapolation."""
    if v_forward < fc * vbi:
        # V_f positive means forward bias → denominator (1 - V_f/PB)^-MJ
        return cj0_per_area * area / (1.0 - v_forward / vbi) ** mj
    else:
        # Linear extrapolation past FC*PB (BSIM4 §6.3 / SPICE-default)
        C_at_FC = cj0_per_area * area / (1.0 - fc) ** mj
        slope = mj * cj0_per_area * area / (vbi * (1.0 - fc) ** (mj + 1))
        return C_at_FC + slope * (v_forward - fc * vbi)


# ---------- Bias ----------
VDS = 2.0
VGS = 0.6
VBS = 0.7        # body forward-biased near turn-on
VB_NWELL = 2.0
V_nwell_to_body = VB_NWELL - VBS   # = 1.3 V reverse on nwell-pwell diode


# ---------- Compute components ----------
results = {}

# 1. Source-body junction (forward biased: VBS = +0.7 V means V_BS = +0.7 → V_f = +0.7)
Cj_bottom_S = cj_forward(CJS, AS_AREA, PBS, MJS, VBS)
Cj_sw_S    = cj_forward(CJSWS, PS, PBSWS, MJSWS, VBS)
Cj_swg_S   = cj_forward(CJSWGS, W, PBSWGS, MJSWGS, VBS)
Cjs_total = Cj_bottom_S + Cj_sw_S + Cj_swg_S

# 2. Drain-body junction (reverse biased: VBD = VBS - VDS = 0.7-2.0 = -1.3 V → V_R = +1.3 V)
V_BD_reverse = VDS - VBS    # = 1.3 V reverse
Cj_bottom_D = cj_reverse(CJS, AD_AREA, PBS, MJS, V_BD_reverse)
Cj_sw_D     = cj_reverse(CJSWS, PD, PBSWS, MJSWS, V_BD_reverse)
Cj_swg_D    = cj_reverse(CJSWGS, W, PBSWGS, MJSWGS, V_BD_reverse)
Cjd_total = Cj_bottom_D + Cj_sw_D + Cj_swg_D

# 3. Gate-bulk (Meyer, BSIM4 capmod=2)
# In saturation V_GS=0.6 ≥ V_th ≈ 0.5 → mostly inversion, Cgb ≈ 0 to small.
# Use simplified: in inversion Cgb ≈ 0; in deep-depletion small. Use CGBO*L as floor.
Cgb_overlap = CGBO * L
# In moderate inversion, take Meyer linear Cgb ≈ 0.1*Cox*W*L as conservative
COX_WL = COXE * W * L
Cgb_meyer = 0.05 * COX_WL    # small, body-charge-modulation residual
Cgb_total = Cgb_overlap + Cgb_meyer

# 4. Parasitic NPN junctions (Pazos)
# The NPN sits with E=source, B=body, C=drain. At our bias:
#   B-E forward (0.7V), B-C reverse (1.3V).
Cbe_bjt = cj_forward(CJE_BJT, 1.0, VJE_BJT, MJE_BJT, VBS, FC_BJT)   # 1.0 area-factor since CJE is total
Cbc_bjt = cj_reverse(CJC_BJT, 1.0, VJE_BJT, MJC_BJT, V_BD_reverse)

# 5. Deep N-well diode (nwell ↔ pwell-body). Reverse-biased 1.3 V.
Cnwell = cj_reverse(NWELL_CJ0, NWELL_AREA, NWELL_PB, NWELL_MJ, V_nwell_to_body)

# 6. Channel-to-body via depletion (Csb_channel) — for a floating body, the
# inverted channel sits above the body region; in strong inversion the body
# depletion cap Cdep ≈ EPS_SI / Wd, where Wd ≈ sqrt(2*EPS_SI*phi_F/(q*NDEP))
NDEP_CM = 1.7e17               # cm^-3
NDEP = NDEP_CM * 1e6           # m^-3
phi_F = (KB * 300 / Q) * math.log(NDEP_CM / 1.07e10)   # ~0.42 V
Wd = math.sqrt(2 * EPS_SI * 2 * phi_F / (Q * NDEP))    # ≈ 60 nm at strong-inv
Cdep_channel = EPS_SI / Wd * W * L                       # F

# ---------- Sum ----------
total = (Cjs_total + Cjd_total + Cgb_total + Cbe_bjt + Cbc_bjt + Cnwell + Cdep_channel)

components = [
    ("M1 Cjs (S-body, fwd 0.7V)",  Cjs_total),
    ("M1 Cjd (D-body, rev 1.3V)",  Cjd_total),
    ("M1 Cgb (Meyer + overlap)",   Cgb_total),
    ("Parasitic NPN Cbe (fwd)",    Cbe_bjt),
    ("Parasitic NPN Cbc (rev)",    Cbc_bjt),
    ("Deep N-well diode (rev 1.3V, 5xWL)", Cnwell),
    ("Channel-body depletion",     Cdep_channel),
]

# ---------- Write JSON + bar chart ----------
summary = {
    "bias": {"VDS": VDS, "VGS": VGS, "VBS": VBS, "V_nwell": VB_NWELL},
    "geometry": {"W_m": W, "L_m": L, "AS_m2": AS_AREA, "PS_m": PS, "NWELL_AREA_m2": NWELL_AREA},
    "components_fF": {name: round(c * 1e15, 3) for name, c in components},
    "C_eff_total_fF": round(total * 1e15, 3),
    "z448_claim_fF": 12.1,
    "ratio_z451_over_z448": round((total * 1e15) / 12.1, 3),
    "notes": [
        "NWELL_AREA=5xWL chosen as plausible 130nm DNW footprint; varies ±2x with layout.",
        "CJE/CJC of parasitic NPN are TOTAL caps (not per-area); Pazos card units.",
        "Cgb in inversion is small; 0.05*Cox*WL is conservative upper bound.",
        "Cjs uses BSIM4 forward extrapolation (FC=0.5) since VBS=0.7 > FC*PB=0.37.",
    ],
}

(RES / "cap_breakdown.json").write_text(json.dumps(summary, indent=2))

# ASCII bar chart
log_path = RES / "run.log"
with open(log_path, "w") as f:
    f.write("z451 capacitance audit — body-node C_eff breakdown\n")
    f.write(f"Bias: V_DS={VDS}V, V_GS={VGS}V, V_BS={VBS}V, V_nwell={VB_NWELL}V\n")
    f.write("=" * 70 + "\n\n")
    max_c = max(c for _, c in components)
    for name, c in components:
        bar_len = int(50 * c / max_c) if max_c > 0 else 0
        f.write(f"{name:42s} {c*1e15:8.3f} fF |{'#'*bar_len}\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'TOTAL C_eff':42s} {total*1e15:8.3f} fF\n")
    f.write(f"{'z448 KILL_SHOT claim':42s} {12.1:8.3f} fF\n")
    f.write(f"Ratio z451 / z448 = {(total*1e15)/12.1:.2f}\n\n")
    f.write("VERDICT:\n")
    if total * 1e15 > 24:
        f.write(f"  → z448 UNDERESTIMATED. Actual C_eff = {total*1e15:.1f} fF is >2x worse.\n")
        f.write("    τ would be even longer, KILL_SHOT confirmed and AMPLIFIED.\n")
    elif total * 1e15 < 6:
        f.write(f"  → z448 OVERESTIMATED. Actual C_eff = {total*1e15:.1f} fF is <0.5x.\n")
        f.write("    KILL_SHOT claim WRONG → real bug is in I_iion / BJT coupling.\n")
    else:
        f.write(f"  → z448 estimate within 2x. C_eff = {total*1e15:.1f} fF.\n")
        f.write("    KILL_SHOT diagnosis (structurally inadequate I) stands.\n")

print(json.dumps(summary, indent=2))
print(f"\nWrote {RES / 'cap_breakdown.json'} and {log_path}")
