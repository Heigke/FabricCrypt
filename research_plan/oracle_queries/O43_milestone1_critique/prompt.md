# O43 — Oracle Milestone-1 Critique (MEP first wave + D1 sweep)

You are reviewing results from a 24-hour campaign on the NS-RAM
(Pazos & Lanza, Nature Electronics 2025) device-modelling + SNN
benchmark project. We want **CRITICISM and FALSIFICATION**, not
validation. Tell us where the results are fragile, where we may be
fooling ourselves, and what single experiment would best break the
strongest claim.

## What just happened (last 6 hours)

**D1 distributed sweep** (320 cells × 4 seeds, 3-node cluster):
- Best cell: 84.45 ± 0.58% test accuracy on MNIST 28×28
- 11 cells passed CONSERVATIVE gate (≥82.65% with rail≤10%, clip≤5%)
- 0 strict AMBITIOUS PASS, but top cells' CIs overlap the
  N1b Poisson baseline (84.65 ± 0.72%)
- Operating point: C_b = 8 fF, V_G2 = 0.35 V, dt = 1e-7 s, g_in ≥ 0.4
- **All top cells at V_G2 = 0.35 V** (right edge of pre-MEP surrogate's
  V_G2 axis [0, 0.45]) — possible edge bias

**MEP-2** built denser surrogate v3 (24K pts, V_G2 to 0.60, V_b to 1.0)
at 96.78% convergence. Non-conv at high V_b + high V_G2 (frontier).

**MEP-3** added V_Nwell as 5th axis. Diode itself silent at DC, but
**V_Nwell strongly modulates impact ionization Iii through the
well-body diode coupling** — Iii varies 2553× over V_Nwell ∈ [0.5, 5.0] V
at fixed (V_G1, V_G2, V_d, V_b).

**MEP-1** replaced nearest-neighbor with quadrilinear interpolation in
the GPU SNN harness. **MIXED RESULT**:
- d115 (D1 best cell) reproduces within +0.05 pp
- Max improvement +10.31 pp on d047 (5 fF / V_G2=0.25 / dt=5e-6 / g_in=0.8)
- **BUT 6 of 16 cells REGRESS, worst −67.96 pp on d276**

## What we have validated independently

C_b ≈ 8 fF triangulated from THREE measurements:
1. Slide-11 freq-current calibration (8-14 fF)
2. Slide-21 ramp transient PMP-9 (best match at 8 fF)
3. D1 sweep peak accuracy (8 fF row peaks at 84.45%)

Per-V_G1-regime fit (PMP-2) closes 33-bias DC residual to 0.17-0.29 dec
(silicon noise floor); was 0.51 dec averaged. Three regimes physical.

## Three questions — be HARSH

**Q1. Fragility audit on the D1 84.45% result.** Where could we be
cherry-picking or over-claiming? Specifically:
- Is the V_G2 = 0.35 V edge effect real silicon physics or a surrogate
  artifact? (We're at the edge of the pre-MEP V_G2 axis.)
- 4 seeds with bootstrap CI on 1000/2000-image test subsets — is the
  CI tight enough to claim "matches Poisson"?
- Did we forget to bootstrap over Poisson encoding randomness AND
  weight randomness jointly?
- What single experiment best falsifies "NS-RAM is a viable rate-coded
  SNN input neuron"?

**Q2. MEP-1 asymmetry — is it physics or numerics?** The trilinear
interp helps 10/16 cells and hurts 6/16 by up to -68 pp. The hurt
cells are at dt ≥ 5e-7 + low-mid V_G2. Hypothesis a: nearest-neighbor
floor-snap was an artifact giving false-high accuracy in those regions.
Hypothesis b: trilinear interpolation across V_b discontinuities is
introducing spurious values. Hypothesis c: something else.
Which is most likely and how do we test? What would falsify each?

**Q3. The V_Nwell × Iii coupling finding from MEP-3** is striking
(2553× variation). Is this a real physics discovery worth a v4.4
sentence, or a SPICE-model artifact of our specific well-body diode
parameterization? What's the falsification: a single Sebas
measurement we could ask for?

Be specific. Cite log lines or numbers if you point at fragilities.
Skip pleasantries. Maximum 1500 words.
