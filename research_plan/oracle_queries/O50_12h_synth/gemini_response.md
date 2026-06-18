# gemini response (gemini-2.5-pro) — 43s

Excellent, this is a well-structured review request. Based on the provided log context from 2026-05-12, here are the answers to your three questions.

***

### Q1 — GATE-CROSSING: Defensibility and Fragility

Of the ambitious gates crossed, one is highly defensible, while another is critically fragile.

**Most Defensible: Bayesian NS-RAM RNG (ESS 1.03× + NIST 5/5)**

This finding is the most defensible and publishable for three reasons:

1.  **Standardized External Validation:** It passed the full NIST statistical test suite (5/5). This is a widely accepted, objective benchmark for the quality of a random number generator. It's a binary pass/fail gate that the project cleared.
2.  **Strong Internal Metric:** The Effective Sample Size (ESS) of 1.03× indicates that the generator is producing high-quality, low-autocorrelation samples, slightly better than an ideal independent sampler. This metric speaks directly to its efficiency and statistical integrity.
3.  **No Contradictory Evidence or Flags:** Unlike other results, the log contains no oracle critiques, user complaints, or methodological flags against the RNG result. It stands as a clean, self-contained success.

**Most Fragile: HDC Noise Robustness (80.4% @ σ=0.05)**

This finding is the most fragile and should not be presented as a locked headline without major caveats.

1.  **Methodological Flaw (Overclaim):** The O49 oracle critique provides a damning assessment: the headline mixes results from two different configurations. It uses the energy figure (2.3 nJ/inf) from an `N=1024` cell but the noise-robust accuracy (80.4%) from an `N=2048` cell. This is a classic "apples and oranges" comparison that would be immediately rejected in peer review.
2.  **Contradicts Prior Gating:** The critique notes that the original `z293 4B2` gates for this task had "locked-FAILED." The `z302 N=2048` result was swapped in to create a "win" without disclosing the change in experimental setup. This undermines the credibility of the result and the reporting process.

The V_G1=0.6 branch fit of 0.43 dec is a valuable *diagnostic* but is not a standalone publishable finding, as it only represents a fraction of the device's operating regime while the overall model is failing.

***

### Q2 — CHERRY-PICK RISK: Reporting the V_G1=0.6 Branch Success

Reporting the "V_G1=0.6 branch at 0.43 dec PASS" is **valid stratified reporting, not cherry-picking**, but only if it is accompanied by specific, honest disclosures. Presenting it in isolation would be gross cherry-picking.

The key distinction is that the different V_G1 branches represent physically distinct operating regimes. The SA1 analysis confirmed this, noting the parasitic BJT "turns on hard at V_G1 ≥ 0.3 V" via a 1000x step in the `mbjt` parameter. Therefore, V_G1=0.2V is the subthreshold regime, while 0.4V and 0.6V are in the active, floating-body regime. It is scientifically valid to analyze these regimes separately.

To report this finding responsibly, the following disclosures are mandatory:

1.  **State the Global Failure First:** The top-line result must be that the cell-wide model **failed** its gate, with a median log-RMSE of 1.46 dec. This is the primary conclusion.
2.  **Present All Branch Data:** Show a table with the results for all three branches (V_G1=0.2, 0.4, and 0.6) from the z305 run, clearly marking the 4.56 dec and 1.76 dec results as FAILs.
3.  **Provide the Physical Justification:** Explicitly state that the branches represent different physical regimes (subthreshold vs. active BJT) and that the model's failure is not uniform. This frames the 0.6V success as a clue, not a victory.
4.  **Connect to the Topology Gap:** The V_G1=0.6 success should be used as evidence to support the SA3/SA4 conclusion: the model's parameters are reasonable within a single regime, but the *structural incompatibility* between regimes points directly to the missing physics (the VNwell→VB diode) as the root cause of the cell-wide failure.

In short, the 0.43 dec result is not a "win" to be celebrated, but a crucial piece of evidence that validates the decision to halt parameter tuning and focus on the topology rebuild.

***

### Q3 — NEXT EXPERIMENT: Highest-Value, Short-Term Action

Given the constraints (1-3 hours, materially shifts v4.4 readiness, falsifiable), the single highest-value experiment is to **run the `z305b` corrective experiment to fix the per-branch ETAB implementation bug.**

**Experiment Definition:**
*   **Action:** Rerun the z305 sweep exactly, but modify the simulation script to apply the `ETAB_M1` parameter on a per-branch basis (0.95 for V_G1=0.2, 1.7 for V_G1=0.4, 2.5 for V_G1=0.6) as specified by the SA1 canonical data, instead of the fixed global value of +1.8.
*   **Time:** The log estimates this as a ~1-hour task (z305 took ~15 mins of compute, plus setup/analysis).
*   **Hypothesis:** The catastrophic >2.5 dec degradation in the V_G1=0.2 branch's fit (from 2.06 to 4.56 dec) is primarily an artifact of the `ETAB` bug, not fundamental physics.

**Why it is the Highest-Value Experiment:**

1.  **Materially Shifts Readiness:** The current narrative is blocked by the "topology gap." However, a known, major implementation bug is confounding the analysis. Fixing it provides a clean, definitive baseline of the *true* model error. If the V_G1=0.2 error drops from 4.56 dec back to ~2 dec, it confirms our understanding and strengthens the case that the remaining error is indeed structural. It removes the risk that we are about to embark on a multi-day topology rebuild to chase an error that was partially just a typo.
2.  **Falsifiable:** The outcome is binary.
    *   **Hypothesis Confirmed:** The V_G1=0.2 error improves significantly. This cleans the slate and validates the "topology gap" as the *real* next step.
    *   **Hypothesis Falsified:** The V_G1=0.2 error remains at ~4.56 dec. This would be a crucial finding, suggesting the physics of the subthreshold regime is even more poorly understood than assumed, and the bug was not the primary cause.
3.  **Removes a "Known Unknown":** Proceeding with the topology work while knowingly leaving a major bug in the baseline analysis is poor scientific practice. This experiment is a necessary act of due diligence that de-risks the entire next phase of work. While the log dismisses it as "low value," its value lies in providing a trustworthy foundation for the much more expensive work to come.
