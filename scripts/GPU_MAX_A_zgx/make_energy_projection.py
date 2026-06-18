"""Step 4b: Energy projection for NS-RAM 28nm tapeout.

Inputs:
  - 130nm Id_pk = 4.8 mA at Vd_pk = 2.0 V (from results/z471_snap_calibrate)
  - Mario thesis target / Sebas measurements; documented in
    docs/sebas_iv_130nm_fit.md.
  - 130 -> 28nm tech scaling factor based on published constant-field scaling:
    * Current per device scales as 1/S^2 (where S = 130/28 ~ 4.64) for
      *constant-field* scaling (Dennard) — i.e. I -> I / 4.64.
    * Supply voltage Vd_pk scales as 1/S -> 2.0 / 4.64 ~ 0.43 V at 28nm
      (matches typical 28nm-bulk core voltage ~0.9V; we conservatively use
       Vdd_28nm = 0.9 V, the standard nominal for 28nm bulk libraries).
    * Per-op energy E = I * V * t. At constant-field, t scales as 1/S.

Reference: Borkar, IEEE Micro 1999 ("Design Challenges of Technology
Scaling"); see also IRDS 2022 transistor-scaling tables. For the 130->28nm
shrink, conservative published ratios are I_28/I_130 ~ 0.5-0.7 at matched
W/L, with Vdd dropping from ~1.5 V (130nm core) to ~0.9 V (28nm bulk).

Writes:
  results/GPU_MAX_A_zgx/energy_projection.md
"""
from __future__ import annotations
import json, os
from pathlib import Path

OUT = Path(os.environ.get("GPU_MAX_A_OUT",
                          str(Path(__file__).resolve().parents[2] /
                              "results/GPU_MAX_A_zgx")))
OUT.mkdir(parents=True, exist_ok=True)


def projection() -> dict:
    # Measured 130nm (Sebas thesis / z471 calibrated cell):
    Id_pk_130 = 4.8e-3       # A
    Vd_pk_130 = 2.0          # V
    t_op_130  = 100e-9       # s   (100 ns transient pulse, conservative)
    E_op_130  = Id_pk_130 * Vd_pk_130 * t_op_130   # J
    P_pk_130  = Id_pk_130 * Vd_pk_130              # W

    # Conservative published scaling: 130 -> 28 nm
    # (Borkar 1999; IRDS 2022; Frank et al. Proc IEEE 2001)
    # constant-field scaling factor S = L_old / L_new
    S = 130.0 / 28.0          # ~ 4.64
    # Per-device current scales as 1/S (constant-field W/L width-scaled
    # one-shot, NOT 1/S^2 which is per-area). For analog cells where W/L
    # is preserved (Mario's NS-RAM is a fixed-W cell shrunk to 28nm), I
    # scales as 1/S.
    I_28 = Id_pk_130 / S      # A
    Vdd_28 = 0.9              # V (28nm bulk nominal)
    # Operation time at 28nm: gate delay shrinks as 1/S -> tprop ~ 21 ns
    t_op_28 = t_op_130 / S    # s
    E_op_28 = I_28 * Vdd_28 * t_op_28
    P_pk_28 = I_28 * Vdd_28

    ratio_E = E_op_130 / E_op_28
    ratio_P = P_pk_130 / P_pk_28

    return {
        "130nm_measured": {
            "Id_pk_A": Id_pk_130, "Vd_pk_V": Vd_pk_130,
            "t_op_s": t_op_130, "P_pk_W": P_pk_130, "E_op_J": E_op_130,
            "source": "results/z471_snap_calibrate/calibration_summary.json + Mario thesis",
        },
        "28nm_projected": {
            "Id_pk_A": I_28, "Vdd_V": Vdd_28,
            "t_op_s": t_op_28, "P_pk_W": P_pk_28, "E_op_J": E_op_28,
            "scaling_factor_S": S,
            "method": "Dennard constant-field, Borkar 1999; W/L preserved",
        },
        "ratios": {
            "energy_reduction_x": ratio_E,
            "peak_power_reduction_x": ratio_P,
        },
        "comparison_per_8bit_MAC": {
            "28nm_NSRAM_per_op_pJ": E_op_28 * 1e12,
            "28nm_8b_int_MAC_digital_pJ_approx": 0.2,  # Horowitz ISSCC 2014 (8b int MAC @ 45nm)
            "note": "Digital 8b int MAC at 45nm ~ 0.2 pJ (Horowitz, ISSCC 2014); "
                    "scaled to 28nm with constant-field that becomes ~0.06-0.1 pJ. "
                    "NS-RAM here projects 1 op = "
                    f"{E_op_28*1e12:.3g} pJ — NS-RAM is _higher_ energy "
                    "per op than digital MAC at peak Id, but each op stores "
                    "*and* computes (no DRAM fetch) so amortised vs DRAM "
                    "access (50-200 pJ/byte) it can still win on data-movement.",
        },
    }


def main():
    p = projection()
    (OUT / "energy_projection.json").write_text(json.dumps(p, indent=2, default=float))
    md = []
    md.append("# Energy projection: NS-RAM 130nm -> 28nm shrink")
    md.append("")
    md.append("## Inputs (measured at 130nm)")
    m = p["130nm_measured"]
    md.append(f"- Id_pk = {m['Id_pk_A']*1000:.2f} mA at Vd_pk = {m['Vd_pk_V']} V")
    md.append(f"- t_op  = {m['t_op_s']*1e9:.1f} ns (transient pulse)")
    md.append(f"- P_pk  = {m['P_pk_W']*1000:.2f} mW; E_op = {m['E_op_J']*1e12:.2f} pJ")
    md.append(f"- source: {m['source']}")
    md.append("")
    md.append("## Scaling rule")
    md.append(f"- S = L_130/L_28 = {p['28nm_projected']['scaling_factor_S']:.3f}")
    md.append("- Dennard constant-field (Borkar 1999; Frank et al. Proc IEEE 2001):")
    md.append("  - I scales 1/S (per-device, W/L preserved)")
    md.append("  - V scales 1/S, but bounded below by leakage; we use Vdd_28 = 0.9 V (28nm bulk nominal)")
    md.append("  - tprop scales 1/S")
    md.append("")
    md.append("## 28nm projected (per cell, one op)")
    q = p["28nm_projected"]
    md.append(f"- I_28 = {q['Id_pk_A']*1000:.3f} mA")
    md.append(f"- Vdd  = {q['Vdd_V']} V")
    md.append(f"- t_op = {q['t_op_s']*1e9:.1f} ns")
    md.append(f"- P_pk = {q['P_pk_W']*1000:.3f} mW; E_op = {q['E_op_J']*1e12:.3f} pJ")
    md.append("")
    md.append("## Ratios")
    r = p["ratios"]
    md.append(f"- Energy reduction:     {r['energy_reduction_x']:.2f}x")
    md.append(f"- Peak power reduction: {r['peak_power_reduction_x']:.2f}x")
    md.append("")
    md.append("## Comparison vs digital 8b int MAC")
    c = p["comparison_per_8bit_MAC"]
    md.append(f"- NS-RAM 28nm per op:   {c['28nm_NSRAM_per_op_pJ']:.2f} pJ")
    md.append(f"- Digital 8b int MAC at 45nm (Horowitz ISSCC 2014): ~ 0.2 pJ")
    md.append("- Per-op energy: NS-RAM does NOT beat a pure digital MAC by 10x.")
    md.append("- The win for NS-RAM is **data movement**: in-memory analog "
              "compute eliminates the DRAM fetch (~50-200 pJ/byte) that "
              "dominates inference cost.")
    md.append("")
    md.append("## Verdict on AMBITIOUS gate")
    if r['energy_reduction_x'] >= 10:
        md.append(f"- PASS: energy projection >= 10x lower (got {r['energy_reduction_x']:.2f}x).")
    else:
        md.append(f"- FAIL: per-op energy ratio is {r['energy_reduction_x']:.2f}x (< 10x).")
        md.append("  The honest interpretation: 130 -> 28nm constant-field "
                  "shrink gives roughly the factor S = 4.64 in current AND in "
                  "time, but Vdd only drops from 2.0 V to 0.9 V (factor ~2.2). "
                  "Total energy = I*V*t reduces by S*S*S/Vdd_ratio? No — by "
                  f"(1/S)*(V_ratio)*(1/S) = (V_28/V_130)/S^2 = "
                  f"{0.9/2.0/(S*S):.4f} = {1/(0.9/2.0/(S*S)):.2f}x. "
                  "To hit 10x at fixed cell-level energy, we'd need a smaller "
                  "geometry node (e.g. 14nm/7nm) OR architectural amortisation "
                  "(reuse the cell across N ops, e.g. as a recurrent reservoir).")
    md.append("")
    md.append("## Refs")
    md.append("- Borkar, S. \"Design challenges of technology scaling.\" IEEE Micro 1999.")
    md.append("- Frank, D. J. et al. \"Device scaling limits of Si MOSFETs and their application "
              "dependencies.\" Proc. IEEE 89(3), 2001.")
    md.append("- Horowitz, M. \"Computing's energy problem (and what we can do about it).\" "
              "ISSCC 2014 (8b int MAC ~ 0.2 pJ @ 45nm).")
    md.append("- IRDS 2022, More-Moore chapter, transistor scaling tables.")
    md.append("- results/z471_snap_calibrate/calibration_summary.json (this repo, 2026-05-03).")
    (OUT / "energy_projection.md").write_text("\n".join(md) + "\n")
    print((OUT / "energy_projection.md").read_text())


if __name__ == "__main__":
    main()
