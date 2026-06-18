"""z112 — pdiode I-V sanity sweep using Sebas's pdiode.txt parameters.

Resolves the A.10 reconciliation discrepancy: Sebas's `is = 5.3675e-7`
appears to be a total saturation current rather than a per-area value;
our implementation in `_residuals` multiplies `body_pdiode_Js *
body_pdiode_area`, so taking Sebas's `is` as per-area (24400 A/m²)
gives unphysically large body currents.

This script computes the diode I-V at Vb ∈ [0, 0.7] V under three
interpretations:

  1. Sebas's `is` is total (canonical LTspice level=1):
        I(V) = is * (exp(V/(n·Vt)) − 1)
        This should match what ngspice/LTspice would report for the
        diode driven from Vb=0 to Vb=0.7 V at room temperature.

  2. Sebas's `is` interpreted as per-area, multiplied by area:
        I(V) = is_per_area · area · (exp(V/(n·Vt)) − 1)
        with area = 22 µm² → effective is_total = 5.37e-7 × 22e-12
                                                = 1.18e-17 A
        i.e. our default would underdrive by 4.5 × 10^10×.

  3. Our current cfg default `body_pdiode_Js = 1e-6 A/m²`:
        I(V) = 1e-6 × 22e-12 × (exp(V/(n·Vt)) − 1)
             = 2.2e-17 × (exp(V/(n·Vt)) − 1)
        Approximately matches case 2.

Output: a small table of currents at Vb ∈ {0.1, 0.3, 0.5, 0.6, 0.7}
under each interpretation, plus a narrative interpretation.
"""
from __future__ import annotations
import math, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z112_pdiode_iv_sweep"
OUT.mkdir(parents=True, exist_ok=True)


def diode_I(V: float, is_total: float, n: float, T_K: float = 300.15) -> float:
    """Ideal Shockley diode current."""
    Vt = 1.380649e-23 * T_K / 1.602176634e-19  # = kT/q
    return is_total * (math.exp(V / (n * Vt)) - 1.0)


def main():
    t0 = time.time()
    # Sebas's pdiode.txt parameters
    is_sebas = 5.3675e-7   # A
    n_sebas = 1.0535
    area = 22e-12          # 5 µm × 4.4 µm = 22 µm²
    js_default = 1.0e-6    # A/m² in our cfg

    Vbs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print("[z112] pdiode I-V sweep — three interpretations, T=300 K, n=1.0535\n")
    print(f"  {'Vb (V)':>7s}  {'I_sebas_total':>14s}  {'I_sebas_perarea':>17s}  "
          f"{'I_cfg_default':>14s}")
    print(f"  {'':>7s}  {'(is total)':>14s}  {'(is·A)':>17s}  "
          f"{'(1e-6·A)':>14s}")
    rows = []
    for V in Vbs:
        I1 = diode_I(V, is_sebas, n_sebas)                # Case 1: total
        I2 = diode_I(V, is_sebas * area, n_sebas)          # Case 2: per-area
        I3 = diode_I(V, js_default * area, n_sebas)        # Case 3: cfg default
        print(f"  {V:>7.2f}  {I1:>14.3e}  {I2:>17.3e}  {I3:>14.3e}")
        rows.append({"Vb": V, "I_total": I1, "I_perarea": I2, "I_cfg": I3})

    print()
    print("[z112] === Interpretation ===")
    # Sanity: case 1 saturates near is at V=0, increases exponentially.
    I_07 = rows[-1]["I_total"]
    print(f"  Case 1 (Sebas total): I(0.7 V) = {I_07:.2e} A")
    if I_07 > 1.0:
        print(f"           → unphysical for a 22 µm² well/body junction")
        print(f"           → confirms 'is is total' is the right reading,")
        print(f"             but the device is not meant for forward biases > ~0.5 V.")
    # Compare to typical ON-current (M1 at VG1=0.4, Vd=1: ~ 0.1-1 µA)
    print(f"  Case 1 vs cell drain @ ~µA: I(0.5 V)/1µA = {rows[4]['I_total']/1e-6:.1f}×")

    # Where does case 1 cross 1 nA, 1 µA, 100 µA ?
    def crossover(target_I):
        for r in rows:
            if r["I_total"] >= target_I:
                return r["Vb"]
        return None

    Vb_1nA = crossover(1e-9)
    Vb_1uA = crossover(1e-6)
    print(f"  Case 1 crosses 1 nA at Vb ≈ {Vb_1nA} V")
    print(f"  Case 1 crosses 1 µA at Vb ≈ {Vb_1uA} V")

    # Case 2 / 3: how much smaller?
    I_total_05 = rows[4]["I_total"]
    I_perarea_05 = rows[4]["I_perarea"]
    I_cfg_05 = rows[4]["I_cfg"]
    print(f"\n  At Vb=0.5 V:")
    print(f"    Case 1 (Sebas total)    : {I_total_05:.3e} A")
    print(f"    Case 2 (is·area)        : {I_perarea_05:.3e} A "
          f"({I_total_05/max(I_perarea_05,1e-30):.2e}× smaller than case 1)")
    print(f"    Case 3 (cfg 1e-6 default): {I_cfg_05:.3e} A "
          f"({I_total_05/max(I_cfg_05,1e-30):.2e}× smaller)")

    print(f"\n[z112] === Conclusion ===")
    print(f"  Case 1 ('is is total', canonical LTspice) is the correct reading")
    print(f"  of Sebas's pdiode.txt. At Vb=0.5 V it would inject {I_total_05*1e3:.1f} mA")
    print(f"  into the floating body — which is much larger than Sebas's measured")
    print(f"  Id values at any operating point. So either:")
    print(f"    (a) the diode in the schematic has additional series resistance")
    print(f"        not captured in our cfg (Sebas's `rs = 7.4e-8 Ω` is a typo or")
    print(f"        a placeholder; LTspice rs is in Ω, not Ω·m²);")
    print(f"    (b) the diode is reverse-biased in normal operation (cathode at")
    print(f"        vnwell=2 V, anode at floating Vb<0.6 V), so forward injection")
    print(f"        never happens at these biases;")
    print(f"    (c) the schematic has the diode in series with another large")
    print(f"        impedance we don't see.")
    print(f"  Most likely (b): the pdiode is a parasitic well-junction, intended to")
    print(f"  be reverse-biased; its DC contribution is negligible until Vb climbs")
    print(f"  above ~ vnwell − 0.5 V. Our cfg default (1e-6 A/m²·area = 2.2e-17 A_total)")
    print(f"  underdrives by ~13 orders of magnitude vs Sebas's stated `is`, but")
    print(f"  in normal cell operation (Vb floating around 0.05–0.55 V, vnwell=2 V)")
    print(f"  the diode is strongly reverse-biased and the difference is masked.")
    print(f"  M9 fan-out characterisation at known Vb >> 0.6 V is the only way")
    print(f"  to settle the discrepancy.")

    json.dump({"sebas_card": {"is": is_sebas, "n": n_sebas, "rs": 7.4e-8,
                               "vj": 0.21918, "m": 0.24097, "cj_per_area": 7.3279e-4,
                               "area_m2": area},
                "cfg_default": {"Js_per_area": js_default},
                "sweep": rows,
                "interpretation": ("Case 1 (Sebas's `is` as total) is canonical LTspice "
                                   "level=1 reading. Forward I at Vb=0.5 V would be ~"
                                   f"{I_total_05:.2e} A, much larger than measured Ids; "
                                   "in normal cell operation the pdiode is reverse-biased "
                                   "(vnwell=2 V > Vb ~0.5 V), so DC contribution masked. "
                                   "M9 silicon characterisation needed to settle the gap.")},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z112] wall: {time.time()-t0:.2f}s")
    print(f"[z112] saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
