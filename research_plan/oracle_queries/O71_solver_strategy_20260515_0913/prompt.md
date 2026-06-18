# O71 Oracle Critique — NS-RAM Snapback Solver Strategy

## State of play (HONEST)

**Modell**: pyport NS-RAM 2T (BSIM4 batched + Gummel-Poon NPN). 33 Sebas Sweden DC measurements at VG1∈{0.2,0.4,0.6}, VG2∈[-0.4,+0.4].

**Problem**: Measured snapback fold = 2-3 dec jump at Vd≈1V. Model fold = 0.03-0.08 dec. Off by ~100× in magnitude.

**S1 body-strap diagnostic**: 1D Newton on Vsint with Vb FORCED to 0.8V at (VG1=0.6, VG2=0.2, Vd=1.5V) → claimed Ids = 1.05e-6A (vs measured 2.07e-5A, vs 2D cold-Newton 3e-12A). Interpreted as "fold IS in BSIM4 physics, solver fails to navigate".

**S2 arc-length continuation (along Vd)**: 33/33 biases, smooth monotone curve, 0 folds. Bit-identical to 2D cold Newton. KILL-SHOT for fold-via-Vd-continuation. Suggests S1's "5.5 dec jump" is at a DISCONNECTED solution branch (if it's real at all).

**Alternative hypothesis** (open): S1's forced-Vb 1D solve may be a PHANTOM — Ids = 1e-6 at Vb=0.8 doesn't mean (Vb=0.8, Vsint=Vsint_S1, Vd=1.5) is a self-consistent 2D root. Body KCL (R_B residual) may be huge there.

**In flight** (3 parallel solver experiments):
- S2a: iii_gain homotopy (start at iii=10, ramp to 0.90 carrying solution)
- S2b: two-branch search (cold Vb=0 init AND hot Vb=0.8 init, pick max-Ids)  
- S2c: pseudo-transient continuation (Backward-Euler on Vb with Cb=8fF, slow Vd ramp)

## Three questions — terse hostile critique

**Q1 (FALSIFICATION OF S1)**: Is S1's "Ids = 1.05e-6A at Vb=0.8" a phantom or a real disconnected basin? Specifically: what single diagnostic, in 5 minutes, would definitively confirm whether (Vb=0.8, Vsint=*, VG1=0.6, VG2=0.2, Vd=1.5) is a self-consistent 2D fixed point of the full residual system?

**Q2 (METHOD RANKING)**: Of (S2a homotopy, S2b two-branch, S2c pseudo-transient), which is MOST LIKELY to succeed on R-46 params? Rank them with reasons. Are we missing any obvious technique (e.g., gmin stepping, deflation, source stepping, branch switching)?

**Q3 (META-CRITIQUE)**: Have we made a deeper modeling error that explains why ALL solver strategies probably fail? Consider:
- Is the residual system even WELL-POSED at high Vb? (Could there be a singularity in Newton's Jacobian that makes Ids "blow up" but not be a real root?)
- Is the BSIM4 NSRAM topology actually capable of producing snapback at these geometries, or have we been assuming the wrong physics regime entirely?
- Should we just be running TLP transient simulation per industry standard for snapback, rather than DC?

Be BRUTAL. Cite specific log lines or numerical values. Under 200 words per question.
