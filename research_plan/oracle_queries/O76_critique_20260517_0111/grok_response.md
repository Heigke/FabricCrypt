# grok response (grok-4-latest) — 26s

**Q1 (overclaim fragility of 1.276 dec)**

The 1.276 headline is fragile on three fronts:

1. **Convergence-rate asymmetry masquerading as invariance**: z446.PT_VBIC reports 25/25 fwd + 25/25 bwd, but the broader campaign (z432, z443) shows PT methods routinely drop VG1=0.2 (7 fails, 32% conv in z432 fwd) or produce wildly asymmetric basins (fwd 1.3 vs bwd 2.86 in z443/z449). The "avg=1.276" is a post-hoc quadratic mean over non-comparable attractors. If the PT warm-start is simply landing in the same Newton basin as DC_VBIC for the easy biases while quietly failing or oscillating on the hard ones, the number is noise.

2. **Basin selection / hidden no-op risk**: The 4-pipeline identity (z443 = z449_A = z449_B = z454_SB_OFF at exactly 1.311/2.864) is not explained by physics; it is the signature of a mechanical wiring or KCL residual bug (exactly like the confirmed z444 BESD no-op). PT_VBIC only moves the number by ~0.09 dec from PT_GP. That delta is smaller than the typical per-bias variance and could be an artifact of how the pseudo-transient continuation is initialized or how the VBIC AVC terms interact with the existing Slotboom multiplier. No ablation has isolated whether the improvement is real or just different solver path length.

3. **Conv asymmetry + VG1=0.2 regression**: Backward sweeps are systematically worse (often >2.8 dec) because they start from the latched high-Vb state. The headline treats fwd+bwd symmetry as given, but the data show the model has a strong directional hysteresis that PT only partially masks. VG1=0.2 remains the killer bias across the entire campaign; any claim that averages it away without a full-grid revalidation after every topology change is overstated.

**Q2 (highest-information falsifier)**

Run z460 exactly as proposed but upgraded: re-execute the z443_VBIC_AVL pipeline (and z446.PT_VBIC) with ALPHA0×10 (literature range) while enforcing (a) strict 25/25 convergence on both directions with no dropped biases, (b) identical warm-start and arc-length settings, and (c) full per-bias fwd/bwd/avg reporting plus I_snap_b and Vb_max diagnostics.  

If the cell-wide numbers remain statistically identical to the 1.311/2.864 baseline, the invariance is a code bug (residual never reaches the body KCL row or the multiplier is short-circuited) and the entire 1.276 headline collapses. If the numbers move materially (especially if the fwd/bwd gap closes and VG1=0.2 improves), the literature hypothesis is live and the PT family is at least partially exonerated. This single experiment simultaneously tests the mechanical-bug hypothesis, the ALPHA0 scaling claim, and the directional asymmetry that has been papered over for weeks.

**Q3 (NO-CHEAT drift)**

- "NEW BEST z446.PT_VBIC fwd=1.396/bwd=1.156/AVG=1.276 dec. PT_GP=1.188, PT_VBIC=1.276 → ONLY PT-family hits <1.5 dec honest avg." (P1b ZGX FINAL COMPLETE). This is a direct overclaim. The oracle 12h review (dispatched the same day) had already returned a 2/3 split warning that the 4-pipeline identity may be a hidden no-op. Declaring it the "NEW BEST" and "defensible" while the falsifier z460 is still pending is narrative cherry-picking.

- "HONEST_BASELINE.md updated with P1b zgx addendum. Headline defensible: z446.PT_VBIC avg=1.276 dec (25/25 biases, fully balanced)." The word "defensible" is doing the cheating. The synthesis document itself had already proven two systematic cherry-pick modes (direction-pick on z430/z443 and bias-pick on z432 VG1=0.2 column). Calling an un-falsified number "defensible" while the oracle split and the z444 precedent are on the record is the exact opposite of NO-CHEAT.

- "P4 R_1M done IDENTICAL again... 4/5 R-values now confirmed no-op." This language quietly buries the assumption that a single lumped Rbody is the right test. The physically relevant regime (high R + weakened NPN + proper body-leak path) was never run; the experiment stopped at the point where the result was convenient. That is a hidden modeling assumption presented as empirical fact.

The campaign is not yet at the level required for a v4.5 brief. The 1.276 headline should be retracted or heavily caveated until z460 (or equivalent) is executed and the mechanical-bug hypothesis is killed.
