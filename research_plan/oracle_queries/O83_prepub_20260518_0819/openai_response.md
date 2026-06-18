# openai response (gpt-5) — 86s

Q1 — Remaining headline risks a hostile reviewer can tear down quickly

- “One primitive, six modes on the same cell” central claim
  Minimal test: Freeze a single calibrated parameter file (z474, post-sign patch), disallow refits, vary only {Vb, Vd, VG1, timing}. Re-run all six demos and their prereg gates end-to-end.
  Likely outcome: At least 2–3 modes need hidden-parameter retunes (e.g., R_body, snap_npn_V_knee) to pass; otherwise they degrade or fail. If any mode needs a re-fit, “single primitive” collapses into “six reparameterized circuits.”

- GENERATE (TRNG): “NIST 5/5 @ 0.4 pJ/bit”
  Minimal test: Dieharder/NIST STS over ≥10^8 bits, 16 “cells” cross-correlation C; report min-entropy and extractor cost; include sense-amp energy.
  Likely outcome: Cross-cell |corr| mean ~0.1–0.2; extractor plus periphery pushes energy into multi-pJ/bit; headline energy becomes non-competitive. If ‖C−I‖_F/N > 0.3, entropy collapses.

- SUPPRESS (LMS equalizer): “170× energy”
  Minimal test: BER vs SNR sweep with explicit ADC/DAC model (8–10b, ≥10 MS/s), same target BER as digital LMS; energy/bit including periphery.
  Likely outcome: 100–1000× energy inflation; BER floor worse than digital for SNR ≥15 dB. Headline advantage evaporates.

- DETECT (PC-NAB; KWS/ECG cascade 60.8% save)
  Minimal test: PC-NAB official scoring vs HTM-Java/LSTM/IForest; cascade vs strong DSP baseline; include periphery energy.
  Likely outcome: Scores below simple baselines; energy saving vanishes with I/O cost. Claim is demoted to “works,” not “good.”

- LEARN-FROM (NES/SPSA via device noise)
  Minimal test: MNIST and FashionMNIST, matched query budgets vs Gaussian SPSA with whitening; 5 seeds; K2 noise-corr audit.
  Likely outcome: NS-RAM remains below Gaussian SPSA even with whitening (prior: 27.5% vs 40.5%); reviewer concludes device noise isn’t a useful learning resource here.

- COMPUTE-THROUGH (EP-NSRAM)
  Minimal test: Full 60k MNIST with IFT pyport, 4 seeds, ≥97% accuracy target; K1 Jacobian conditioning stats on-batch; compare to tanh-EP.
  Likely outcome: Acc <97%, conditioning pathologies in nontrivial fraction; relegated to “method works in principle, not competitive.”

- PLASTICIZE-UNDER (STDP with body eligibility)
  Minimal test: STDP window fit and retention over hours; τ_body drift across Vb; write–read fatigue.
  Likely outcome: τ_body drift >20% and retention instability; claim becomes a qualitative observation, not robust plasticity.

- “IFT unifies autograd across all six”
  Minimal test: Gradcheck (FD vs IFT) across random points for each mode; histogram of relative error; singularity rate.
  Likely outcome: Modes with switching show pockets of ill-conditioning; reviewer questions generality.


Q2 — Unfalsifiable framing detection

- The taxonomy itself (“six canonical noise operations”) is descriptive; acceptable in Introduction but unfalsifiable unless each mode has a quantitative, preregistered bar. Several current mode writeups read as “we can route noise to do X” without a pass/fail criterion.

- Modes lacking a crisp, falsifiable bar right now:
  - LEARN-FROM: “use device noise for SPSA” is trivially true for any noise source. Without a predeclared win condition (e.g., beats matched Gaussian SPSA on MNIST/FashionMNIST at equal query budget by ≥5 pp, 5/6 seeds), it’s unfalsifiable.
  - PLASTICIZE-UNDER: Showing an STDP-like curve once is descriptive. Needs quantitative retention (τ_body CV < 20% across Vb and time), stability under repeated pairing, and non-volatility over specified horizons.
  - DETECT: “PC-NAB discovery” and “cascade saves 60.8% energy” are soft without standardized baselines. Must commit to official NAB score ≥ baseline X with CI, and energy including periphery within Y× of baseline.
  - COMPUTE-THROUGH: “Smoke pass, no NaN” and “body-τ ≈ 1 ms” are descriptive. Needs a task-level bar (e.g., MNIST ≥97% with EP-NSRAM, 4 seeds; K1: ill-conditioned Jacobian <10% of batches).
  - SUPPRESS: “170× energy” without peripheral model or iso-BER target is non-falsifiable; define BER@SNR targets and total energy budget.
  - GENERATE: NIST 5/5 alone is weak; must add cross-cell independence (‖C−I‖_F/N < 0.1), min-entropy ≥0.997/bit, and extractor/periphery energy.

- Recommended explicit bars to make each falsifiable:
  - GENERATE: STS+Dieharder pass; min-entropy ≥0.997/bit; cross-cell corr metric; total energy ≤5 pJ/bit at 130 nm.
  - SUPPRESS: At SNR=20 dB, BER within 2× digital LMS; energy including ADC/DAC ≤10× digital.
  - DETECT: PC-NAB score ≥ competitive baseline (HTM/LSTM) within 5%; cascade energy including periphery ≤1.5× baseline while meeting same F1.
  - LEARN-FROM: Beats matched Gaussian SPSA by ≥5 pp on MNIST/FashionMNIST (equal queries), 5/6 seeds; K2 corr within prereg limit.
  - COMPUTE-THROUGH: MNIST ≥97% (4 seeds); K1: κ(J) < 1e10 on ≥90% of samples.
  - PLASTICIZE-UNDER: STDP window fit R^2 ≥ 0.9; τ_body drift <20% over 1 h; endurance ≥10^5 pairings without degradation.


Q3 — Single brutal experiment not yet tried that could gut the paper

Frozen-parameter, same-cell, six-mode ablation.

Design:
- Lock one calibrated cell parameter file (z474 post-sign patch). Absolutely no refits or topology tweaks between modes.
- Allow only published “programming” knobs: {Vb, Vd, VG1, timing windows}. No changes to R_body, snap_npn_V_knee, noise scales, or hidden fit params.
- Reproduce all six prereg gates: TRNG (STS+min-entropy+cross-cell corr on virtual array), SUPPRESS (BER@SNR with periphery), DETECT (PC-NAB vs baselines; cascade energy incl. periphery), LEARN-FROM (MNIST vs Gaussian SPSA, matched queries), COMPUTE-THROUGH (EP-NSRAM MNIST, K1 stats), PLASTICIZE-UNDER (STDP window + 1 h retention).
- Add one toggle per mode: noise-off vs noise-on to prove the “noise is the primitive” claim where asserted (GEN/LEARN/PLASTIC should collapse when noise=0; COMPUTE should be mostly unchanged).

Why this is the killshot:
- The unifying headline is “one physical primitive, six operations.” If any mode silently depends on a different fitted device (hidden parameter deltas, topology edits), the unification claim falls apart. A single failure here lets a reviewer dismiss the entire framing as parameter re-tuning theater.
- It is easy to ask for and decisive: one script, one param file, six outputs. Failure on even one or two modes (without refit) is enough to recommend rejection.
