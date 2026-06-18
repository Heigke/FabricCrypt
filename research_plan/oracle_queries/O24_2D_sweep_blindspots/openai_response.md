# openai response (gpt-5) — 129s

Short answer: yes—there are a few high‑leverage knobs you haven’t touched (mainly GP Early/high‑injection and BSIM4 S/D resistance bias‑dependence). Expect at most another ~0.1–0.25 dec if they break your worst‑row pattern; >0.3 dec without silicon re‑ID is unlikely.

Ranked by likely impact for a lateral parasitic NPN (low‑doped base) in a floating‑body 2T NS‑RAM:

1) Gummel‑Poon Early and high‑injection
- Knobs: VAF (forward Early), VAR (reverse Early), IKF (forward knee).
- Why: Lateral NPNs have small Early voltages and enter high injection at mA currents; both shape VCE dependence and β roll‑off.
- Sweep:
  - VAF, VAR: 10–100 V (log or geometric steps).
  - IKF: 0.05–50 mA (log).
  - Do 2D sweeps with Bf held at 2e4; then 3D spot checks Bf×VAF×IKF.
- Signature if it matters: largest residual drop where VG2 is high and Vd is low/moderate (output conductance mismatch); also reduced overprediction at the highest‑current rows if IKF was too large.
- Coupling: Strong Bf×IKF anticorrelation; Bf×VAF correlation (low VAF can force higher fitted Bf). 1D sweeps can mislead—use 2D.

2) GP BE/BC recombination at low VBE/VBC
- Knobs: ISE/NE (BE), ISC/NC (BC).
- Why: Floating body bias is set by junction recombination; lateral devices often need extra low‑VBE current beyond IS.
- Sweep: ISE, ISC = 1e‑14…1e‑9; NE, NC = 1.2…2.0. Constrain ISE/ISC ≤ ~0.1×IS to avoid double‑counting.
- Signature: fixes low‑current underfit at near‑turn‑on rows without hurting high‑current rows; reduces row‑to‑row spread tied to body charge.
- Coupling: Strong with IS; keep IS fixed at 1e‑9 while exploring ISE/ISC.

3) BSIM4 S/D resistance bias‑dependence and asymmetry
- Knobs (BSIM4 v4.8.3): RDSMOD, RDSW/RDSWMIN, PRWG/PRWB (Vg/Vb dependence), RD/RS or NRD/NRS (asymmetry), RDW/RSW (width scaling).
- Why: LDD/extension resistance in 130 nm is Vg/Vb‑dependent; it feeds back into impact‑ionization and the parasitic BJT.
- Sweep:
  - RDSMOD ∈ {0,1,2,3}; RDSW ±3× nominal; RDSWMIN 0.5–1× RDSW.
  - PRWG: 0–1; PRWB: −1–0.
  - NRD/NRS: 0.2–5 to probe asymmetry.
- Signature: selective improvement at low‑Vd/high‑VG rows (triode corner) without changing high‑Vd saturation much; if asymmetry helps only one polarity, you likely found the right direction.
- Ref: BSIM4 manual “Source/Drain Resistance Model” (RDSMOD/RDSW/PRWG/PRWB).

4) Cell‑level body network and vertical path
- Knobs: BSIM4 rbody/rbpb/rbdb/rbsb; explicit substrate diode scale to n‑well/p‑sub; optional small RC from body to guard‑ring node if present in silicon.
- Why: Floating‑body charge partition strongly sets Vb; small changes can re‑bias the parasitic NPN.
- Sweep: 0.5–5× on rbody branches; 0.1–2× on substrate diode area/scale.
- Signature: global shift of rows most sensitive to body charging; reduces cases where the same few rows stay >1 dec.

5) GP series resistances (RB/RC/RE)
- Why: Lateral geometry often has notable RC; you previously saw Rb pathologies but RC/RE may be the safer shaping knobs.
- Sweep: RC 0–200 Ω, RE 0–50 Ω, RB 0.1–10 kΩ.
- Signature: slopes vs current change; biggest effect at high current and low VCE; if residual improves only at top‑current rows you’re adjusting extrinsics correctly.
- Coupling: RC with VAF (both change output conductance).

6) BSIM4 DIBL/short‑channel VT coupling
- Knobs: ETA0/ETAB, PDIBL1/2/PDIBLB, DITSMOD.
- Why: if the worst rows are high‑Vd/low‑VG, DIBL misfit could be masquerading as BJT leakage.
- Sweep: ETA0 0–1, ETAB 0–1; PDIBL1/2 0–1 (coarse).
- Signature: fixes low‑VG, high‑Vd corners; little effect on high‑VG rows.

7) BSIM4 gate leakage
- Knobs: IGCMOD, IGBMOD on; tune NTOX/TOXE/AIGC/AIGB per PDK defaults ±2×.
- Why: At 130 nm it’s small; may clean up a few µA‑level points at high VG2.
- Signature: only affects high‑VG2/low‑Id rows; no change elsewhere.
- Ref: BSIM4 “Gate Current Model” (IGB/IGC).

8) Numerics
- Knobs: GMIN 1e‑14…1e‑9; source‑stepping, transient homotopy; initial Vb from prior row.
- Why: avoids wrong roots in floating‑body loops.
- Signature: if the same 5 rows drop by >0.2 dec with numerics only, you were stuck on the wrong branch.

Couplings to watch
- (Bf, IKF, VAF) triangle: use 2D/3D sweeps; 1D is misleading.
- (IS, ISE/ISC, NE/NC): fit ratios, not absolutes.
- (RDSW/PRWG/PRWB) × (body network): both modulate Vb vs Id.
- RC × VAF: both increase rout; don’t “double‑count.”

Expectation on headroom
- If VAF/IKF and ISE/ISC are truly off, 0.1–0.2 dec median improvement is plausible; 0.25 dec is an optimistic ceiling.
- >0.3 dec without re‑identifying the body network from silicon/TCAD is unlikely; 0.795 dec is probably near the irreducible floor for this model topology.

Critical next steps
- Re‑run ngspice/Xyce cross‑val at (Bf=2e4, IS=1e‑9) before further tuning to rule out solver artifacts.
- Check whether the “>1 dec” outliers are the same rows across all (Bf,IS); if yes, prioritize VAF/IKF and RDSW/PRWG/PRWB. If they move around, look at numerics (GMIN/initial Vb).
