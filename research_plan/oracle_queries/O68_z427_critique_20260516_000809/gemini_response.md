# gemini response (gemini-2.5-pro) — 55s

To the NS-RAM team,

Here is my external referee critique as requested. I will be blunt. You have made significant progress, but the DISCOVERY gate-crossing is premature. The current model has exchanged one kind of error for another and rests on physically unsubstantiated assumptions.

---

### **Q1 — Gate-crossing risk (statistical)**

**The headline "cell-wide log-RMSE = 1.733 dec" is not honest. It is a misleading summary of a model that is qualitatively broken in a critical operating regime.**

1.  **Averaging Artifact:** Your log-RMSE is a classic example of how a single aggregate metric can obscure profound model failure. The spectacular improvements in the high-current, high-V_G1 regimes (errors reduced by >3 decades) are numerically overwhelming the 0.44 dec *regression* in the low-current, low-V_G1 regime. You have not improved the model holistically; you have re-allocated the error budget from the ON-state to the OFF-state.

2.  **Failure of Alternative Metrics:**
    *   **Max Per-Bias Residual:** The headline falls apart immediately. Your V_G1=0.2 branch shows the model overpredicting current by ~4 decades. This corresponds to a **max log error of ~4.0 dec**. By this metric, which captures worst-case performance, your model is further from the DISCOVERY gate (<2.0 dec) than your z425 starting point (3.928 dec).
    *   **Median Per-Bias Residual:** While likely better than the mean, the median would still be misleading. With 2/3 of your branches showing massive improvement, the median error across all 33 points would likely be low, but this still hides the fact that the model's sub-threshold physics is now incorrect.
    *   **Branch-Stratified RMSE:** This is the most honest representation, and you've already calculated it. The V_G1=0.2 branch has an RMSE of **2.74 dec**, which **fails the DISCOVERY gate on its own.**

3.  **Recommendation for Publication:** For a credible publication, you cannot rely on a single cell-wide RMSE. You must present:
    *   **Branch-stratified RMSEs** (as in your table). This is non-negotiable.
    *   A **scatter plot of log(I_sim) vs. log(I_meas)** for all 33 points. The V_G1=0.2 points will form a distinct cloud far from the y=x line, making the sub-threshold failure visually undeniable.
    *   The **max absolute log error**, as it quantifies the model's worst-case predictive failure.

**Conclusion on Q1:** Do not claim you have crossed the DISCOVERY gate. The correct summary is: "Structural fixes H1+H2 dramatically improve snapback and high-current fit (log-RMSE < 1.4 dec for V_G1≥0.4) but break the sub-threshold model, causing a 4-decade overprediction at V_G1=0.2."

---

### **Q2 — Cherry-pick risk (physical)**

**The H1 and H2 modifications carry a high risk of being a physically baseless, reverse-engineered curve-fit. The justification is post-hoc and requires immediate, rigorous challenge.**

1.  **H1 (1 MΩ Shunt): A "Magic Number"**
    *   **Smell Test:** A 1 MΩ resistance is not inherently absurd, but its origin is. It was *guessed*. Substrate resistance is a complex function of geometry, doping profiles, and temperature. It is not a universal constant. Without derivation from TCAD or measurement, 1 MΩ is pure numerology. You needed a resistor to pull V_Sint down, and you picked a number that worked.
    *   **Physical Implication:** This value directly controls the base current of the parasitic BJT. By guessing this value, you have effectively set the BJT gain by fiat rather than by allowing the physics of the device to determine it. This is a critical flaw.

2.  **H2 (GIDL Routing): Suspiciously Convenient**
    *   **Physical Plausibility:** This modification is more physically defensible than H1. GIDL current is generated in the body and should flow through it. Routing it to GND is a common but potentially flawed simplification in compact models.
    *   **The "Tell":** The problem is that H2 is the perfect accomplice to H1. You had a V_Sint runaway problem. H2 provides a new current injection source into Sint, and H1 provides the exact leakage path needed to bleed that current off and stabilize the node voltage. The two changes are so perfectly complementary to fixing the specific high-V_G1 error that it strongly suggests they were co-designed to solve the symptom, not derived from a first-principles analysis of the underlying cause.

3.  **Overfitting:** This is not just a risk; it is a near certainty. You had large errors on two branches. You introduced two new structural degrees of freedom (the presence of the shunt and the GIDL path) and one "hidden" parameter (the 1 MΩ value). Unsurprisingly, you fixed the two branches. The model is now tuned to the training data. The regression on the V_G1=0.2 branch is the smoking gun: your "fix" is not general and has broken the physics in a regime it wasn't tailored for.

**Conclusion on Q2:** You have engaged in targeted error cancellation, not physics-based model improvement. H2 is a plausible correction, but H1 is an unjustified parameter. The combination is a classic over-fit.

---

### **Q3 — Next falsification experiment (single highest-value)**

The goal is to break the cycle of curve-fitting. You must design an experiment that can invalidate your new hypotheses. The most urgent task is to understand the V_G1=0.2 regression, as it is the primary evidence against H1+H2 being a real physical improvement.

Here is my ranking and rationale for your proposed experiments.

**Highest-Value Experiment:**

**1. (Rank 1) Experiment #3: Switch H2 OFF only at V_G1=0.2.**
*   **Rationale:** This is the most direct and surgically precise test to diagnose the regression. It directly probes the causality: Is the new GIDL current path (H2) responsible for the 4-decade sub-threshold overprediction?
*   **Hypothesis:** At low V_G1 and low-to-moderate V_D, the standard BSIM GIDL model is known to be imperfect. By routing this potentially inaccurate GIDL current into Sint, you are injecting a spurious base current into the parasitic BJT, turning it on when it should be off.
*   **Value:** This is a zero-cost simulation. If turning off H2 at V_G1=0.2 makes the sub-threshold current drop back to measured levels, you have found your culprit. The problem is not the *routing* of GIDL, but the GIDL *model itself* being inaccurate in this regime. This cleanly separates the structural hypothesis (H2) from the parametric accuracy of the underlying BSIM equations.

**Immediate Follow-up Experiments:**

**2. (Rank 2) Experiment #1: Sweep H1 shunt resistance (10kΩ to 100MΩ) with NO refit.**
*   **Rationale:** This directly attacks the "magic number" nature of the 1 MΩ value. You should plot key figures of merit (RMSE at V_G1=0.2, RMSE at V_G1=0.6, and V_Sint at a key bias point) as a function of this resistance.
*   **Expected Outcomes & Interpretation:**
    *   A **sharp, deep minimum** in the V_G1=0.6 RMSE around 1 MΩ would confirm it's a tuned fit parameter.
    *   A **monotonic trend that plateaus** would suggest that *a* leakage path is physically necessary, but the exact value is not critical, which is more robust.
    *   Crucially, observe the effect on the V_G1=0.2 leakage. Does a higher resistance (e.g., 100 MΩ) fix the sub-threshold while preserving most of the snapback gain? This maps the trade-off space of your model.

**Crucial but Lower-Priority Experiments:**

**3. (Rank 3) Experiment #2: Add a blind held-out bias group.**
*   **Rationale:** This is the gold standard for testing overfitting. However, you should only do this *after* you have a model you believe in. Testing a model you already know is broken in the sub-threshold regime will only confirm what you already know: it doesn't generalize. First, use #3 and #1 to create a more physically sound model, then use #2 to validate it.

**4. (Rank 4) Experiment #5: TCAD-level structural probe.**
*   **Rationale:** This provides the ground truth for the H1 resistance value. It is essential for your final paper. However, it is not an *immediate* falsification experiment for the modeling team. It's a longer-lead-time task that validates a parameter, not the dynamic interaction of the model components. Get the model structure right first.

**5. (Rank 5) Experiment #4: Replace H1+H2 with a single GIDL-only fix.**
*   **Rationale:** This is a model simplification step (Occam's razor). It's a good idea, but less fundamental than diagnosing the existing failure (#3) or testing the parameter sensitivity (#1).

**In summary, your highest-value path is: (1) Diagnose the sub-threshold regression with Exp #3. (2) If GIDL is the culprit, focus there. (3) Regardless, challenge the 1 MΩ magic number with Exp #1 to understand its sensitivity and trade-offs.**
