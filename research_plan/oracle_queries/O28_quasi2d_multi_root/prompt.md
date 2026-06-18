# O28 — Multi-root in NS-RAM 2T residual function: which is physical?

## Context

Implementing your O25 #1-ranked architecture upgrade (quasi-2D body
model, split V_b → V_{b,S} + V_{b,D} coupled by R_{b,SD}) on the v4.2-final
PyTorch BSIM4 port. Plan A wrapper (additive, ~250 LOC) is in place.

The wrapper math is **conservation-correct** (R_{BS} + R_{BD} = R_B
of the lumped solver at any V_b configuration), so when R_{b,SD} → 0
forces V_{b,S} = V_{b,D}, the joint 3-residual problem should reduce
to the lumped 2-residual problem.

**Surprise**: it doesn't. The new 3×3 Newton converges to a
**different fixed point** than the lumped 2×2 Newton at most biases.
Investigation reveals that the residual function genuinely has
**multiple roots** in this parameter regime — not just numerical
artifacts.

## The numerical evidence

At production parameters $B_f=9{\times}10^3$, $V_a=0.55$~V,
$I_s=10^{-9}$~A (the v4.2-final fit point), 9 representative biases:

| bias        | lumped Id (default)  | tightened lumped Id    | quasi-2D Id      |
|-------------|----------------------|------------------------|------------------|
| VG1=0.2 lo  | 2.7e-7  (R≈1e-10)    | 1.2e-6  (R≈1e-12)      | 1.6e-6 (R≈4e-11) |
| VG1=0.2 mid | 1.5e-10 (R≈2e-10)    | 2.4e-6  (R≈2e-12)      | 1.8e-7 (R≈2e-10) |
| VG1=0.4 mid | 1.4e-10 (R≈2e-10)    | 2.4e-6  (R≈2e-12)      | 1.2e-6 (R≈1e-10) |
| VG1=0.6 mid | 2.2e-7  (R≈1e-10)    | 2.5e-6  (R≈5e-12)      | 2.4e-6 (R≈2e-11) |
| VG1=0.4 lo  | 2.1e-6  (R≈1e-10)    | 1.2e-6  (R≈1e-12)      | 1.2e-6 (R≈1e-12) |
| VG1=0.6 hi  | 8.7e-10 (R≈2e-9)     | 8.7e-10 (R≈2e-9)       | 8.7e-10 (R≈2e-9) |

(R = max|residual| at converged voltages.)

**6/9 biases multi-root**: lumped (default Newton stop:
`max(|R| < max(I_abstol, I_reltol·|I_phys|))`, achieves R ~ 1e-10)
finds a low-Id root; tightening (Iabstol=1e-15, more iters) walks
past it to a high-Id alt-root with R ~ 1e-12.

**Critical fact**: the 0.654-dec production fit is calibrated
against the *low-Id* default-lumped root because that's what
matches Sebas's measurements (e.g. at VG1=0.4, V_d=0.5, V_{G2}=0:
Sebas measured Id=1.9e-9, default-lumped predicts Id=2.1e-6 —
already off by factor 1000, but the BJT gain calibration absorbs
much of that).

## The cell physics (recap)

Two stacked NMOS (M1 on top, M2 on bottom in series) sharing a
floating P-body inside a deep N-well biased at +2V. Parasitic
lateral NPN: collector=drain (V_d), base=body, emitter=GND.

Residual equations: `R_Sint = KCL at intermediate node`,
`R_B = KCL at floating body`. Solved jointly for (V_sint, V_b).

The "alt-root" appears to correspond to the **well-body diode +
parasitic-NPN-on simultaneously** branch where Vb pins near 0.5 V
and Id ≈ 1-3 µA regardless of V_{G1}, V_{G2}.

The "physical root" (default-lumped) is bias-dependent and tracks
V_{G1}, V_{G2} as expected for transistor-like behavior, but its
residuals don't go to numerical zero — they hit a floor around
1e-10 A.

## QUESTIONS

1. **Which root is physically correct?** The bias-dependent one
   (matches Sebas's data, residual ~1e-10) or the bias-independent
   one (Id ≈ 1-3 µA, residual ~1e-12)? In a real silicon NS-RAM
   2T cell, what would the parameter analyzer show at these biases?

2. **Why does the residual not go to zero?** If the bias-dependent
   root is physical, why is its residual ~1e-10 and not ~1e-15?
   What's the expected scale of "real" KCL residual at this device's
   current scale?

3. **Should we re-think the lumped solver's stop criterion?** Is
   the empirically-tuned `Iabstol + Ireltol·|I_phys|` actually
   selecting a *physical* root, or is it stopping early at a
   "near-stationary point" that happens to match measurements
   for unrelated reasons (BJT gain absorbs the discrepancy)?

4. **Implication for the 0.654-dec headline** in the v4.2-final
   brief sent to NRF: if quasi-2D Newton finds a more-converged
   alt-root, is the 0.654-dec fit a numerical artifact of incomplete
   Newton convergence? Or is it the right physics with empirically-
   chosen stop criterion?

5. **Path forward**: should we (a) ngspice cross-check ~3 biases
   to certify the physical branch, (b) abandon Plan A and pivot to
   two-NPN model instead, (c) accept lumped's selection and engineer
   quasi-2D to mirror its trajectory, or (d) something else?

Be terse — under 400 words. The brief is already sent (NRF deadline
2026-05-06 passed). This is post-deadline due-diligence on whether
the brief's claims are defensible.
