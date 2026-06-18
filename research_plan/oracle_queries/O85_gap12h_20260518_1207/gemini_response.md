# gemini response (gemini-2.5-pro) — 49s

**Q1 — Cherry-pick audit on today's 2 wins**

The two wins are fragile and reek of single-point success.

1.  **z477c V7 (420ns Hopf):** The win is achieving the 419.88ns period with physical V_b [-0.5, +0.62]V. The most likely cherry-pick is **parameter sensitivity**. The log shows a sweep (`z477b`, `z477c finsweep`) landed on a single winning combination (τ=800ns, k_n=1e-4). This is a knife-edge solution. A real device has process and thermal variation. The win is meaningless if a 1% deviation in V_b or a 5°C temperature change (affecting BSIM4 parameters) kills the oscillation or pushes the period outside the ±2% Mario target.
    *   **Exposing Experiment:** A "robustness contour plot." Fix the winning parameters and run a 2D sweep of V_b vs. a key process-variation parameter (e.g., transistor Vth offset). The result will likely be a tiny island of success in a sea of failure, proving the current result is not manufacturable or robust.

2.  **z481 EP-FIX (91% MNIST):** The win is stabilizing training to 90.83±2.60% and fixing the drift. The most likely fragility is that the "fix" is just a collection of hyperparameter hacks masking fundamental instability. The log explicitly states: "**seed3 single-batch blowup, early-stop saved**." This is the smoking gun. "Early-stop" is not a fix; it's a method for cherry-picking the epoch with the lowest validation loss before an inevitable collapse. The reported mean of 90.83% is likely an average of these cherry-picked peaks, not a statement of true convergence.
    *   **Exposing Experiment:** The proposed DS-1 is correct, but must include a specific control: run all 10 seeds to completion for a fixed, large number of epochs **with early-stopping disabled**. Track the final-epoch accuracy, not the best-epoch accuracy. If the mean accuracy collapses or the variance explodes, the "fix" is invalid and the EP claim remains dead.

**Q2 — Reservoir-USP retirement**

**Formally KILL it. Do not keep it.**

The "differentiable limit-cycle reservoir" concept is not just wounded; it has been systematically executed by a firing squad of the project's own experiments. Citing the log:
1.  **ERvMESH killshot:** "CATASTROPHIC 3/3 gates FAIL" on NARMA-10.
2.  **z479 rebuttal:** FAILED, "NMSE 0.346 vs 0.325 baseline (slightly worse)." The idea was "permanently retracted."
3.  **z482 coupled FHN:** KILLED. "catastrophic" on MG-17, chance-level on seq-MNIST, and "NO phase-locking."
4.  **O84 Oracle Convergence:** "CPG/reservoir USPs not endorsed by any oracle."

Keeping it as "MG-only" is intellectually dishonest. It's a single positive data point in a field of catastrophic failures. Any competent reviewer will see the NARMA and other failures (which must be reported for honesty) and correctly identify the MG result as a corner-case anomaly, not a viable capability.

Keeping this zombie claim weakens the far more compelling and defensible USP-1 ("multi-function 2T cell"). It signals a lack of focus and an unwillingness to abandon failed hypotheses. After the day's quadruple-kill, the project's credibility rests on demonstrating rigorous, honest assessment. The most powerful way to do that is to state clearly in the v4.6 brief: "Initial explorations into reservoir computing proved unsuccessful on standard benchmarks like NARMA-10, and this direction has been retired." This builds trust and focuses the narrative on what *does* work.

**Q3 — Highest-value next experiment**

Given the constraints, the ranking is unambiguous. The goal is to build the most defensible paper possible from the wreckage.

1.  **(a) DS-1 full-MNIST EP-FIX 5-10 seeds + ablation + alignment-cosine:** This is the absolute, non-negotiable priority. The O84 oracle identified EP-FIX as a key pillar of the new USP, but the "seed3 blowup" note in the log shows its foundation is soft. Before any other work is done, you must confirm whether the 91% result is real or a statistical fluke saved by early-stopping. If this experiment fails, the new USP-1 is critically weakened. If it succeeds, it becomes the primary algorithmic result of the paper. This is a "death-sentence" experiment for the entire paper narrative. **Impact/hour: Infinite (make-or-break).**

2.  **(b) bias-programmable mode-atlas in pyport:** This is the second priority and can be run in parallel if a second machine is available. After (or while) DS-1 validates the *parts*, this experiment synthesizes the *whole*. It directly generates the central figure for the paper: a single, calibrated BSIM4 model demonstrating its transition between LIF (V8), Hopf (V7), and noise modes, purely through bias voltage changes. This is the most direct and powerful illustration of the new USP-1. **Impact/hour: High (creates the core narrative figure).**

3.  **(c) GPU 16×16 mode-atlas variance sweep:** Lower priority. This is a robustness study, which is crucial, but only *after* the primary claims are validated and demonstrated. This is a supporting experiment for the "discussion" or "supplementary materials" section, not the core thesis. It has less impact per hour than solidifying the main claims. **Impact/hour: Medium (good for rebuttals, not for the main story).**

4.  **(d) revisit-NES-GD:** Dead last. Do not run this. It's a distraction. The project has pivoted based on a brutal day of audits. Chasing old, abandoned ideas now would be a catastrophic waste of time and a sign of strategic incoherence. **Impact/hour: Negative (actively harmful distraction).**
