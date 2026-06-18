# openai response (gpt-5) — 98s

Q1 — Cherry-pick audit on today’s 2 wins

- z477c V7 Hopf (419.88 ns, Vb ∈ [-0.5, +0.62] V, Id_pk 4.39 mA, 12 cycles)
Most likely fragility: knife-edge operating point that vanishes under small bias/PVT/mismatch or numerical-step changes. You only showed 12 cycles at one physical clamp; earlier success needed unphysical Vb (-2 V/+36 V). One experiment: pre-registered mode-atlas around the found point. Sweep a 2D grid (e.g., Vb and VG1 ±50 mV around the winner), at TT/27C and repeat at SS/0C and FF/85C, with dt halved vs current (and one alternate integrator). For each grid point, run 10^4 cycles and report fraction with a stable limit cycle and period within 350–500 ns, plus period CV. If the capture range is tiny or dt-sensitive, it’s a cherry-pick. Numbers to cite: 419.88 ns target vs Mario 430 ns ±2%; earlier 722 ns required unphysical Vb; only 12 cycles shown.

- z481 EP-FIX (90.83±2.60% over 4 seeds; drift +0.44 vs -11.75; best 93.72%; one seed had single-batch blowup saved by early-stop)
Most likely fragility: early-stop/test-peek dependence and seed scarcity. One experiment: fixed-hyperparam, preregistered 10-seed run on MNIST where early-stop is driven strictly by a validation set (no test peeking) and a fixed patience; report final (not peak) test accuracy, drift over last 10 epochs, and alignment-cosine distribution; include single-factor ablations of β-cos, random-sign, VG1-nudge, and early-stop. If accuracy collapses back toward the EP-FULL 75±16% or variance spikes, the “91%” is a cherry-pick. Numbers to cite: EP-FIX 90.83±2.60% vs EP-FULL 75±16%; drift +0.44 vs -11.75; seed3 “single-batch blowup, early-stop saved.”

Q2 — Reservoir-USP retirement

Kill it as a USP in v4.6. Evidence: z479 NARMA-10 rebuttal fails (NMSE 0.346 vs baseline 0.325; target 0.15 missed badly); z482 coupled FHN reservoir catastrophic (MG-17 1.008 vs ESN 0.029, seq-MNIST ~chance, synchrony 0.0016 = no phase-locking); ERvMESH killshot already restricted you to MG-only. O84 3-way oracle explicitly did not endorse reservoir/CPG; they converged on the multi-function 2T-cell USP (V6 reset 40 ns, V8 LIF, V7 Hopf 420 ns, intrinsic noise, EP-FIX via diff IFT).

Keeping “MG-only/onset-only” as a scoped reservoir claim weakens the multi-function thesis: it invites reviewers to probe the already-failed NARMA/temporal-generalization and dilutes the clean device-methods story. A few sympathetic reviewers might tolerate an appendix note (“MG forecasting works in canonical FHN, not competitive with ESN”), but keeping it in Contributions/USP space will be penalized. Recommendation: remove reservoir as a contribution; retain a brief mechanistic note in Methods/Appendix with negative results prominently cited.

Q3 — Highest-value next experiment (impact per hour; 24–48h budget)

1) (a) DS-1 full-MNIST EP-FIX (5–10 seeds, ablation, alignment-cosine)
- Highest leverage: converts a fragile 4-seed 91% into a defensible result or kills it cleanly. Directly supports the only surviving learning claim. Likely 8–16 GPU-hours; clear accept-or-reject evidence.

2) (b) Bias-programmable mode-atlas (LIF+Hopf+noise from same calibrated cell)
- Central to USP-1. Software-only, fast to produce figures. Demonstrates bias-only multi-functionality even without silicon. 6–12 hours.

3) (e) V7 structural-stability check (dt-halving + long-run + small 2D bias map at TT/SS/FF)
- De-risks the 420 ns Hopf claim and preempts reviewer attacks. 8–12 hours.

4) (c) GPU 16×16 mode-atlas variance sweep (process-variation surrogate)
- Useful, but only after (b)/(e) establish base robustness. 12–24 hours.

5) (d) Revisit NES-GD with K2 audit
- Low paper impact now; RNG energy claim is demoted (98.5 pJ/bit, 101× worse vs Cheng 2024). Do later or drop.
