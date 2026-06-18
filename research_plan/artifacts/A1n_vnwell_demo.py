"""A1n — Standalone well-to-body diode current scoping.

Computes the deep-N-well -> P-body junction current at vnwell=+2V vs the
NPN base current at the operating Vb. No source edits; no full re-solve
of the 2T cell. Just first-principles diode equation with realistic
series resistance, compared against A.1.l results.

Run:
    cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
    source venv/bin/activate
    python research_plan/artifacts/A1n_vnwell_demo.py
"""
from __future__ import annotations
import math, json
from pathlib import Path

# ---- physical constants
kT_q = 0.02585  # V at 300K (thermal voltage Vt)

# ---- M1 model card values (from data/sebas_2026_04_22/M1_130DNWFB.txt)
JSS = 3.4089e-7     # A/m^2  (sidewall) — bottom-junction Js0 same order
NJS = 1.017         # ideality factor for body-source junction

# ---- Geometry (M1 is W=0.36um, L=0.18um; deep-N-well typically 5-50x bigger)
W_n = 0.36e-6
L_n = 0.18e-6
A_active = W_n * L_n               # 6.48e-14 m^2 (channel area, lower bound)
A_dnw_typ = 50 * A_active          # ~3.2e-12 m^2 (deep-well area, ~7x7 um typical)
A_dnw_big = 500 * A_active         # ~3.2e-11 m^2 (worst case, full cell DNW)

# ---- Bias from A.1.l converged op-point (WORST: VG1=0.6, VG2=0.0, Vd=1.5)
VB_CONV = -0.2531    # converged Vb from A1l_vb_trace.json
VNWELL  = +2.0       # measurement bias per dataset folder name
V_FWD   = VNWELL - VB_CONV   # = +2.2531 V FORWARD bias on well->body diode

# ---- Reference: BJT base current at Vb=0.7 (from A.1.l)
IB_BJT_AT_VB0p7 = 2.8373e-7   # A — what would charge body to NPN-firing if available


def diode_current(area_m2, V_fwd, Js=JSS, n=NJS, Rs=None):
    """Ideal diode + optional series resistance via iterative solve.

    With Rs, V_diode = V_fwd - I*Rs:
        I = Js*A * (exp((V_fwd - I*Rs)/(n*Vt)) - 1)
    Solve by Newton iteration; cap at very large currents.
    """
    Is_total = Js * area_m2
    Vt_n = n * kT_q

    # Ideal (no Rs)
    if Rs is None or Rs <= 0:
        # Cap exponent to avoid overflow
        x = min(V_fwd / Vt_n, 80.0)
        return Is_total * (math.exp(x) - 1.0), V_fwd  # V_diode = V_fwd

    # With Rs — iterative Newton on f(I) = I - Is*(exp((Vf-I*Rs)/Vt_n) - 1) = 0
    I = 1e-9
    for _ in range(200):
        Vd = V_fwd - I * Rs
        x = min(Vd / Vt_n, 80.0)
        e = math.exp(x)
        f = I - Is_total * (e - 1.0)
        # df/dI = 1 + Is*(Rs/Vt_n)*e
        dfdI = 1.0 + Is_total * (Rs / Vt_n) * e
        dI = f / dfdI
        I_new = I - dI
        if I_new < 0:
            I_new = I * 0.5
        if abs(I_new - I) / max(I_new, 1e-30) < 1e-9:
            I = I_new
            break
        I = I_new
    return I, V_fwd - I * Rs


def main():
    print("="*72)
    print("A1n — Well-to-body diode current at vnwell=+2V, Vb=-0.253V")
    print("="*72)
    print(f"  Vt (300K)                 = {kT_q:.5f} V")
    print(f"  Jss (M1 model card)       = {JSS:.4e} A/m^2")
    print(f"  njs                       = {NJS}")
    print(f"  V_forward (vnwell - Vb)   = {V_FWD:+.4f} V  -- FORWARD BIASED")
    print(f"  exp(V_fwd / (n*Vt))       = {math.exp(V_FWD/(NJS*kT_q)):.3e}  (unphysical w/o Rs)")
    print()

    rows = []
    for label, area in [
        ("A_active (W*L)         ", A_active),
        ("A_DNW_typical (~50x)   ", A_dnw_typ),
        ("A_DNW_big     (~500x)  ", A_dnw_big),
    ]:
        for Rs_label, Rs in [("Rs=0 (ideal)", 0.0),
                             ("Rs=100 ohm",  1e2),
                             ("Rs=1 kohm",   1e3),
                             ("Rs=10 kohm",  1e4)]:
            I, Vd = diode_current(area, V_FWD, Rs=Rs)
            rows.append((label, Rs_label, area, Rs, I, Vd))
            print(f"  {label}  {Rs_label:14s}  "
                  f"I_well->body = {I:.3e} A   V_diode = {Vd:.3f} V")
        print()

    # Compare to BJT base current at Vb=0.7 (from A.1.l)
    print("-"*72)
    print(f"  BJT base current at Vb=0.7 (from A.1.l)  Ib = {IB_BJT_AT_VB0p7:.3e} A")
    print(f"  Ratio (typical-area diode @ Rs=1k)       /Ib = "
          f"{rows[1*4+2][4]/IB_BJT_AT_VB0p7:.1f}x")
    print(f"  Ratio (typical-area diode @ Rs=10k)      /Ib = "
          f"{rows[1*4+3][4]/IB_BJT_AT_VB0p7:.1f}x")

    # Verdict
    print()
    print("="*72)
    print("VERDICT")
    print("="*72)
    Ityp_1k = rows[1*4+2][4]
    if Ityp_1k > 100 * IB_BJT_AT_VB0p7:
        verdict = ("CONFIRMED: well-body diode delivers >>100x the BJT base "
                   "draw, easily charging Vb to NPN-firing.")
    elif Ityp_1k > IB_BJT_AT_VB0p7:
        verdict = ("PARTIAL: well-body diode exceeds BJT base draw but not "
                   "by orders of magnitude; may need higher Vb to balance.")
    else:
        verdict = "NOT CONFIRMED: diode current too small at realistic Rs."
    print("  " + verdict)

    out = {
        "Vt": kT_q, "Jss": JSS, "njs": NJS,
        "Vb_conv": VB_CONV, "vnwell": VNWELL, "V_forward": V_FWD,
        "Ib_BJT_at_Vb0p7": IB_BJT_AT_VB0p7,
        "rows": [
            dict(area_label=r[0].strip(), Rs_label=r[1], area_m2=r[2],
                 Rs_ohm=r[3], I_diode=r[4], V_diode=r[5]) for r in rows
        ],
        "verdict": verdict,
    }
    out_path = Path(__file__).with_name("A1n_vnwell_trace.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
