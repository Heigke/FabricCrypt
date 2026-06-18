# A.5 — ngspice cross-validation on isolated M2

**Script:** `scripts/z91j_ngspice_isolated_m2.py`
**Output:** `results/z91j_ngspice_iso_m2/{iso_m2.png, summary.json, details.json}`
**Card:** Sebas's `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` (NMOS variant, .param patcher applied) — same card both engines.
**Geometry:** L=1.8 µm (10× Ln), W=0.36 µm — z91g M2 geometry.
**Bias:** Vbs=0, sweep Vd ∈ [0, 2V] @ ΔVd=0.05 V, VG2 ∈ {-0.1, 0.0, +0.2}.
**ngspice:** v42, level=14 BSIM4, gmin=1e-15, reltol=1e-6, abstol=1e-14.

## Result

| VG2 | log-RMSE (pyport vs ngspice) | Id_pyport peak | Id_ngspice peak |
|-----|------------------------------|----------------|------------------|
| -0.1 | 0.97 | 3.59e-16 | 1.81e-12 |
| 0.0  | 0.68 | 8.26e-15 | 5.38e-13 |
| +0.2 | 1.19 | 4.29e-12 | 3.22e-13 |

Median 0.97, max 1.19 dec.

## Interpretation

The two implementations disagree substantially on the **same** model card.

- **Subthreshold (VG2 ≤ 0):** pyport **under**-predicts by 1–6 decades.
  At VG2=-0.1, ngspice peak = 1.8 pA, pyport peak = 0.36 fA — 4 decades off.
  Likely diagnoses: cdsc / cdscb / nfactor / etab interaction in Vth shift,
  or a missing exponential subthreshold floor.
- **Near-on (VG2=+0.2):** pyport **over**-predicts by 1 decade. Likely
  a saturation-region term (Vdseff smoothing, impact-ion add-back, or
  GIDL contribution) firing too aggressively when ngspice doesn't.

## Implications for z91g

z91g cell residual at v26 = 0.99 dec. This validation gives 0.97 dec
of intra-BSIM4 error on isolated M2 alone. The cell-level residual
likely contains a sizable contribution from `compute_dc` itself, not
only from cell wiring.

That said, the VG1=0.4 bottleneck row in z91g (1.8–2.0 dec) cannot be
fully explained by 1-dec BSIM4 error — there's still cell-coupling
specific gap.

## Falsifiable next step

- Audit the 4 specific subthreshold params (cdsc, cdscb, nfactor, etab)
  by perturbing each in our port and checking whether the discrepancy
  with ngspice closes for VG2 ∈ {-0.1, 0.0}.
- If a sign or polarity error is found in our `n` (subthreshold slope
  factor) computation, fixing it should both close this validation
  and improve z91g VG2-low residual.
