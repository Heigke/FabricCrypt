# Stage 6b probe v2 — pyport vs ngspice bisection (CLOSURE)

**Date:** 2026-05-03
**Bias:** L=0.234 µm (M2 geometry), W=1 µm, Vgs=0.5, Vds=1.0, Vbs=0
**Cards:** raw `M2_130bulkNSRAM.txt`, no overrides, no patch_model_values
**ngspice:** ngspice-42, level=14 BSIM4v5
**pyport:** `compute_size_dep` + `compute_dc`, dtype fp64

## Side-by-side at the same operating point

| quantity | ngspice           | pyport            | rel Δ     |
|----------|------------------:|------------------:|----------:|
| Id       | 2.0859 × 10⁻⁸ A   | 2.0887 × 10⁻⁸ A   | **+0.13 %** |
| gm       | 5.6375 × 10⁻⁷ S   | 5.6458 × 10⁻⁷ S   | +0.15 %   |
| gds      | 4.4956 × 10⁻⁹ S   | 4.5017 × 10⁻⁹ S   | +0.13 %   |
| Vdsat    | 4.4116 × 10⁻² V   | 4.4117 × 10⁻² V   | **+0.00 %** |
| Vth      | 0.72133 V         | 0.72133 V         | **+0.00 %** |

pyport's BSIM4 instance evaluation matches ngspice to ≈0.15 % on
every available quantity. Vdsat and Vth match to numerical precision.

## Implication

**The 1.88-dec faithful-mode pyport-vs-measurement gap is NOT in
pyport's BSIM4 DC evaluator.** Stage 6b is therefore CLOSED with a
clean negative result: there is no binning-evaluation bug to fix.

The remaining gap must be in one of:

1. **Sebas's per-bias CSV overrides** — `make_overrides()` applies
   ETAB, K1, ALPHA0, BETA0, NFACTOR, IS, mbjt, area but only when
   the row is non-NaN. Some biases may need different overrides.
2. **The 2T topology layer** — parasitic NPN (Bf=2×10⁴ optimum found),
   well-body diode, body-source diode, body-pdiode. M3a.1 owned the
   Bf side; the diode parameters are still tuned by hand.
3. **Newton root selection** — the VG1=0.4 catastrophe is a wrong-
   root issue (probe v2 finding), still partially open.
4. **Sebas's card vs his measurements** — possible silicon-calibration
   drift between the SPICE model and the actual silicon Sebas measured.

## What this tells the brief

The brief's Section 5 / Sec. 7 limitations list said "binning gap
localised to pyport binning evaluation." That phrasing should be
walked back in any future revision: at the tested point pyport's
binning IS correct. The localisation should now read "binning gap
localised to the 2T topology and Newton root selection."

## Reproducibility

  cd research_plan/ngspice_repro_harness
  ngspice -b test_instance_ags.sp

then:

  source venv/bin/activate && cd nsram && PYTHONPATH=. python -c "..."

(see Stage 6b log entry in 01_LOG.md for the exact pyport invocation).

## Status

- [x] ngspice deck: `test_instance_ags.sp`
- [x] pyport reference: inline one-liner reproducible from log
- [x] Bisection complete — pyport BSIM4 evaluator correct to 0.15 %
- [ ] Update brief Section 5/7 phrasing "binning evaluation" → "topology + Newton root"

This is a **positive validation** for pyport's evaluator and a
**re-localisation** of the residual to the 2T topology + solver.
