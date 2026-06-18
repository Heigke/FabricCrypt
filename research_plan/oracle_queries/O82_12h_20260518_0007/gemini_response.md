# gemini response (gemini-2.5-pro) — 49s

Rigorous review of the 2026-05-17 campaign log follows.

### Q1 — GATE-CROSSING RISK

Of the five claims, two would likely fail a strict audit, and a third is highly vulnerable.

1.  **EP-NSRAM smoke 44% (FAIL):** This is the most fragile claim. The log explicitly states, "CAVEAT EP-smoke: used tanh surrogate, NOT pyport." A reviewer would immediately invalidate this result as it does not test the physical model. It is an infrastructure-level success (gradients flow, no NaNs) but cannot be presented as an algorithmic achievement of the NS-RAM device itself. The claim is about the test harness, not the device.

2.  **GPU-MAX-A MNIST 82.12% (FAIL on energy/competitiveness):** While the accuracy claim (-2.64pp vs. baseline) is honestly reported and passes its gate, the implied claim of a competitive result would fail. The log contains two critical alerts:
    *   The energy projection is pre-peripheral validation, with an "Expected 100-1000× degradation" noted.
    *   The projection is still "> digital 8b-int MAC iso-node," meaning it doesn't win on energy even with optimistic assumptions.
    A reviewer would accept the differentiability proof but reject any claims of practical advantage.

3.  **z471 Mario ±0.055 dec (VULNERABLE):** This claim is a classic case of meeting the letter of the law but not the spirit. The experiment successfully matched the peak drain current (`Id_pk`). However, the log for `z472` reveals the dynamic waveform match was poor: "Mario shape 1/5 strict... t_rise 2.9ns (too fast), t_fall 140ns (too slow)." An audit would flag this as an incomplete calibration, passing on a single metric while failing on the overall temporal dynamics, which are critical for neuron-like behavior.

The remaining two claims are more robust:
*   **z473 V6 PASS:** This is a solid, honest result. The log clearly states V6 (self-reset) passes while V7 (oscillation) fails. This is a defensible claim of a specific LIF-like behavior.
*   **z472 V1 RMSE:** This is not an external claim but an internal regression test verifying the `z474b` patch caused no value drift. It would not be audited as a performance metric.

### Q2 — CHERRY-PICK AUDIT

**Verdict: The honesty test passes decisively.** The log demonstrates a systematic, almost rigorous, culture of reporting failures, caveats, and negative results, rather than burying them. There is no evidence of systematic cherry-picking.

Specific evidence of intellectual honesty includes:

*   **Explicit Failure Labeling:** Numerous entries are labeled with terms like "DISCOVERY FAIL" (`N-Rec-DVS`, `N-BENCH-B`), "AMBITIOUS FAIL" (`N-HDC-DVS`), "honest neg" (`HNRT`), and "HONEST INFEASIBILITY" (`GPU-MAX-B`).
*   **Proactive Kill-Shots:** The team actively runs experiments designed to falsify their own hypotheses, and clearly labels the outcomes: "z475 HONEST KILL_SHOT" and "z476 HONEST KILL_SHOT" are prime examples where the team proved their own idea (body-leak for V7 oscillation) was wrong.
*   **Convergence on Demotion:** The "TRIPLE CONVERGENCE on HDC demotion" (from `N-BENCH-A`, `O80`, and `N-BENCH-B`) shows a willingness to abandon an entire research direction when faced with overwhelming negative evidence. This is the antithesis of cherry-picking.
*   **Caveats and Alerts:** The log is filled with self-critical alerts, such as the need for peripheral validation on energy claims and the use of a surrogate model in the EP-NSRAM test. These are proactively flagged, not hidden.
*   **Forensic Honesty:** The entire `z468` (forensic) → `z469` (bug-fix) → `z470b` (verdict) chain is a model of transparently debugging a major error, admitting the prior baseline was wrong, and re-evaluating from the new ground truth.

The only potential counter-argument is the use of terms like "AMBITIOUS PARTIAL," but even in these cases, the log immediately details the specific failure (e.g., `z461 9-test hung at V1`), providing the necessary context. The overall record is exceptionally transparent.

### Q3 — NEXT HIGHEST-VALUE EXPERIMENT

**NO-CHEAT Pushback:** The framing is sound. This is the correct strategic question.

**Highest-Value Experiment:** **NES-GD v2: Device-Aware SPSA with Whitening.**

This experiment should be dispatched immediately, over the next 24 hours.

**Justification:**
The project has pivoted to the "Master of Noise" (MoN) framework. The log states the MoN paper outline is blocked by "K2 (NES-GD) + K3 (τ-drift)." The `NES-GD` experiment landed as a `DISCOVERY FAIL` but crucially *did not trigger the K2 killshot* (cross-cell correlation was not the issue). The log provides a clear, actionable insight: "real cause = per-cell Iii amplitude heterogeneity → uneven exploration... Fix: per-coord whitening."

This creates a perfect opportunity. The problem is understood, and a software-only fix is proposed. Running this experiment directly addresses a primary blocker for the new strategic paper.

*   **If it PASSES:** It turns a failure into a significant win, validating the "LEARN" mode (§6) of the MoN thesis. It also generates a novel methodological contribution ("device-aware SPSA via whitening"), strengthening the paper's impact.
*   **If it FAILS:** It provides a definitive negative result, allowing the paper to honestly document the limitation and focus on the other 5 modes of the MoN framework. This is still more valuable than the current ambiguous failure.

**Comparison against "Stop and Write":**
Stopping to write now means the MoN paper would have a major hole in its "LEARN" section, citing a failed experiment with an un-tested hypothesis for the failure. This is a weak position. A 24-hour investment to run NES-GD v2 has a high probability of dramatically strengthening the core claims of the manuscript or, at minimum, resolving a key uncertainty. Other potential experiments, like a topology change for V7 oscillation, are multi-day efforts and not suitable for an immediate "next step." This targeted fix is the critical path to a publishable paper.
