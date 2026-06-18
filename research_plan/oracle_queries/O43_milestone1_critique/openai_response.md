# openai response (gpt-5) — 186s

Q1. Fragility audit on the D1 84.45% result

Where the claim is weak
- d115 relies on out-of-domain behavior: clip rate = 20% “at edge of surrogate V_G1 domain.” You pre-registered CONSERVATIVE ≤5%. So the headline 84.45% cell is not actually within the conservative gate. Any result that needs 1-in-5 surrogate queries clipped is not a trustworthy estimate of silicon behavior.
- Edge-of-domain bias on V_G2 is plausible: top cells all at V_G2 = 0.35 V. In the new 24k grid, convergence at 0.35 is only 97.75% (conv_by_vg2["0.35"]=0.9775) versus 100% at ≤0.25. Non-conv rises steeply with V_G2 (down to 86% at 0.6). Even if your operating point is not in the worst frontier, it is moving toward a region where the solver is less reliable and the surface curvature is changing. That is textbook “sweet spot at the rim” fragility.
- Too few seeds and the wrong bootstrap: 4 seeds on 1k/2k test subsets gives CI widths that are not dominated by network stochasticity but by data sub-sampling. Worse, you appear to have bootstrapped test-set images but did not jointly resample weight initialization and Poisson encoding randomness. That underestimates uncertainty in a rate-coded pipeline whose output variance has two independent stochastic sources. A 0.20 pp gap to the Poisson baseline with overlapping CIs is not evidence of “tie”; it is “indeterminate with low power.”
- Interpolation dependence is unresolved near your operating regime: MEP-1 shows mixed behavior; while d115 moves only +0.05 pp, nearby low/mid V_G2 regions can change by +5 to −68 pp solely by changing the interpolant. That means the learned “physics → spikes” operator is not yet stable under benign numerical choices. Without an explicit monotonic/causality check on the local hyper-rectangle around (V_G2=0.35, dt=1e-7, C_b=8 fF), this remains a silent failure mode.
- Selection bias across the 4D space: you report “11 cells passed CONSERVATIVE gate,” but the highlight is the best-accuracy cell that fails the clip gate. Reporting the best number while deprioritizing the compliant cells is classic cherry-pick risk.

Is the V_G2=0.35 effect physics or surrogate artifact?
- Evidence for artifact: convergence deteriorates with V_G2, and clip is high on V_G1 near the operating point. MEP-1 shows that interpolation choice can radically alter outcomes in neighboring regions.
- Evidence for physics: multiple C_b rows peak near similar accuracy, suggesting some robustness to C_b; d115 reproduces within +0.05 pp under quadrilinear. But those are weak positives compared to the numerical red flags above.

Is the CI tight enough to claim “matches Poisson”?
- No. Four seeds with bootstraps over 1k/2k subsets are underpowered and improperly scoped. You need joint resampling over model init and Poisson codebooks and evaluation on the full 10k test set to claim a tie. Current overlap of CIs only says “we can’t tell.”

Did you bootstrap Poisson and weights jointly?
- From the logs: “4 seeds” is ambiguous, but the CI arrays per cell suggest resampling images, not joint resampling of both weight init and Poisson encodings. Treat this as a gap unless you can point to a run log proving nested resampling over both sources.

Single best falsification of “NS-RAM is a viable rate-coded SNN input neuron”
- Do a high-power, no-clip, joint-uncertainty head-to-head at the alleged optimum.
  Protocol:
  1) Use surrogate v3 (24k) with quadrilinear interpolation.
  2) Expand V_G1 domain or restrict operating conditions until clip rate ≤5% for the chosen cell(s).
  3) Fix dt=1e-7, g_in≥0.4, C_b=8 fF; sweep V_G2 in [0.25, 0.30, 0.35, 0.40] to show that 0.35 is not an edge-only quirk.
  4) Evaluate on the full 10k test set with 32 matched seeds per condition; each seed re-draws both weights and Poisson codes. Mirror the exact same seeds for the Poisson baseline.
  5) Report a joint bootstrap/hierarchical CI for Δ = acc(NS-RAM) − acc(Poisson). Pre-register H0: Δ≤−0.5 pp. If the 95% CI of Δ is entirely below 0, the “viable” claim fails. If it straddles 0 but the NS-RAM cell needs >5% clip, it also fails on modeling validity.
This single experiment collapses edge-bias, clip-dependence, and stochasticity under one decisive test.

Q2. MEP-1 asymmetry — physics or numerics?

Observed pattern
- Catastrophic regressions are clustered at low/mid V_G2 (0.05–0.25) and larger dt (≥5e-7 to 1e-6): d276 −67.96 pp (V_G2=0.15, dt=5e-7), d258 −40.44 pp (V_G2=0.05, dt=1e-7), d282 −25.90 pp (V_G2=0.15, dt=1e-6), d131 −2.64 pp (V_G2=0.05, dt=1e-7), d194 −6.42 pp (V_G2=0.05, dt=1e-7).
- Improvements of +5 to +10 pp also occur at low/mid V_G2 but larger dt (e.g., d235 +7.00 pp, d039 +5.76 pp, d047 +10.31 pp at dt=5e-6).

Most likely cause
- c) Time-stepping interacting with interpolation across a piecewise, kinked I–V surface (especially in V_b). The device has real regime transitions (impact-ionization onset, parasitic BJT turn-on). Quadrilinear interpolation of a discontinuous or highly non-convex function can produce interior values that violate physical monotonicities. With larger dt, the simulator traverses larger V_b increments per step, sampling deeper into these interior “blends” and amplifying the error. The fact that some cells improve strongly while others collapse is consistent with path-dependent sampling of different hyper-rectangles, not a uniform bias.
- a) “Nearest-neighbor floor-snap helped” is partially true in that ZOH avoids interpolating across kinks; but the magnitude and directionality of changes suggest more than just losing a lucky snap.
- b) “Interpolation across V_b discontinuities introduces spurious values” is a key mechanism inside (c), but (c) explains why dt dependence is so strong.

How to test and falsify each
- Test for (c) path-dependent aliasing:
  - Run the worst cells (d276, d258, d282) with identical parameters except halve dt progressively: 1e-6 → 5e-7 → 2.5e-7 → 1e-7, keeping total presentation time constant. If accuracy recovers monotonically as dt shrinks under quadrilinear but not under nearest-neighbor, it implicates interpolation-path interaction. Falsification: no dt trend or both interpolants move together.
  - Add “semi-linear” interpolation: linear in (V_G1, V_G2, V_d), nearest-neighbor in V_b only. If this recovers the catastrophic cases, it isolates V_b as the problematic axis and supports the “kink” story.
- Test for (a) floor-snap artifact:
  - Jitter the grid phase: add a small random dither to query points before nearest-neighbor lookup (e.g., ±2 mV per axis), averaged over many jitters. If the nearest-neighbor accuracies for the “good” NN cells (e.g., d276) drop toward the quadrilinear values, then the original high scores were snap artifacts. Falsification: NN remains stable under dither.
- Test for (b) spurious interior values:
  - Local monotonicity checks in log-current space: for each problematic hyper-rectangle, sample a dense cloud and verify expected monotonic signs (∂I/∂V_G1 ≥ 0, ∂I/∂V_d ≥ 0; ∂I/∂V_b sign by regime). Any interior violations indicate interpolation artifacts. Falsification: no violations yet regressions persist.
Mitigations if (c)/(b) hold: use log-domain interpolation with monotone constraints, or piecewise models per V_b regime boundary; or NN in V_b, linear in others. Also expand the grid density specifically along V_b where MEP-2 shows higher non-convergence (vb ≥ 0.65–1.0).

Q3. V_Nwell × Iii coupling (2553×) — discovery or SPICE artifact?

Why it may be artifact
- The body–Nwell diode parameters are uncertain by orders of magnitude: you note body_pdiode_Js default = 1e-6 A/m² may be ~4 OoM low vs Sebas’s effective Js (~2.44e4 A/m², 23:50 log). A misparameterized junction can strongly overstate body charging and therefore Iii sensitivity to V_Nwell.
- MEP-3 shows Ileak is flat vs V_Nwell at V_b=0.5 V (reverse-biased), while Iii varies 2553× at the same DC point. That is a red flag that the “coupling” is not flowing through actual diode conduction at DC, but through how the model couples the well potential into the body/impact-ionization equations. If that coupling constant is model-specific, the effect magnitude may not be physical.
- Convergence is only 94.9% (below your 95% gate) and the worst non-conv regions in MEP-2 are high V_b/high V_G2, i.e., near regime boundaries where Iii models are most brittle.

Why it might be real
- It is physically plausible that changing the Nwell potential shifts the body potential and the lateral field at the drain junction, modulating avalanche multiplication without significant DC diode current at reverse bias. The direction (Iii increasing with V_Nwell when V_Nwell > V_b) is sensible.

Single decisive measurement to falsify
- Ask for a substrate-current vs V_Nwell sweep at a fixed body bias, mimicking the “slide-21” gate/drain biases:
  - Bias: V_G1≈0.40 V, V_G2=0.30 V, V_d=2.0 V, V_b=0.5 V.
  - Sweep V_Nwell ∈ {0.5, 1.0, 2.0, 2.5, 5.0} V.
  - Measure simultaneously: I_drain, I_body (at the body contact), and I_nwell (current into the Nwell supply). The impact-ionization current Iii ≈ I_body (plus any well-injected component, cross-checked by I_nwell sign).
  - Prediction to falsify: I_body should increase monotonically with V_Nwell by at least 10× across 0.5→5.0 V at fixed (V_G1, V_G2, V_d, V_b), with negligible change in I_drain and essentially zero DC diode forward current (I_nwell ≈ 0 for V_Nwell>V_b). If measured I_body is flat (±20%) across V_Nwell, the 2553× effect is a model artifact.
If DC sensitivity is too small to resolve, do the same under a short drain-voltage pulse (slide-21 conditions) and integrate body current over the pulse; the trend vs V_Nwell should remain if the coupling is real.

Other immediate red flags to resolve before v4.4
- Stop quoting any result with clip >5% on any axis; extend the surrogate domain in V_G1 or constrain the neuron drive to operate within the grid (your own gate says so).
- Add a per-cell “in-domain occupancy” metric to the sweep reports (fraction of queries strictly inside hull on all axes); use it as a hard filter for highlights.
- Report a local surrogate sanity check around the operating point: enforce log-domain monotonicity checks per axis; publish violations.
- For the D1 claim, require a joint bootstrap over weights+Poisson and the full 10k test set with ≥32 seeds. Pre-register the non-inferiority margin vs Poisson (e.g., −0.5 pp). Without this, “matches Poisson” is unsubstantiated.
- For MEP-1, add the “semi-linear” interpolant (NN in V_b) and a dt-halving study on the catastrophic cells to disentangle path vs surface issues.

One-sentence break test
- Full-10k, 32-seed, joint-bootstrap, no-clip head-to-head at and around V_G2=0.35 with surrogate v3 and quadrilinear; if NS-RAM underperforms Poisson by more than 0.5 pp or requires >5% clip to avoid collapse, the “viable rate-coded input neuron” claim fails.
