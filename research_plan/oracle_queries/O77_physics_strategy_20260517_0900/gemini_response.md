# gemini response (gemini-2.5-pro) — 74s

Here is a 3-way oracle critique based on the provided context.

***

### Q1 — Physics completion path WITHOUT A.12 data

The central challenge is calibrating a dynamic system (snapback, reset, oscillation) with only static (DC IV) and qualitative-dynamic (a single oscillation period) data. The highest-leverage experiment is one that constrains the most free parameters of this dynamic system simultaneously. Sweeping individual parameters in isolation (as done in z453, z456) has failed.

The single highest-leverage experiment is a **Multi-Objective Bayesian Optimization (BBO) targeting DC-fit and oscillation period simultaneously.**

This is not just another parameter sweep; it's a change in methodology. The optimizer's objective function would be a weighted sum of two errors:
1.  `E_DC`: The cell-wide, fwd+bwd average log-RMSE against the 33 DC IV curves.
2.  `E_osc`: The squared error `(T_sim - 400ns)^2`, where `T_sim` is the simulated oscillation period at the bias condition shown in Mario's slide 21 (e.g., V_D=2V, VG1=0.6V, VG2=0.0V). If no oscillation occurs, `E_osc` is a large penalty.

**Justification of Dominance:**
This approach dominates others because it forces a reconciliation between the static and dynamic behaviors.
*   It dominates simple `R_body` or `snap_Is` sweeps because it co-optimizes the parameters of the NPN's holding current (`Is`, `Bf`, `Ikf`) against the reset path (`R_body`), which is the core physical conflict preventing self-reset (per z456 log).
*   It dominates re-fitting the `I_pos` PWL because the primary failure is in the regenerative NPN loop, not the trigger current.
*   It directly uses the *only* piece of dynamic information available (the 400 ns period), turning a qualitative observation into a quantitative fitting target.

**Concrete Experiment & Targets:**
*   **Script:** A new BBO harness wrapping the transient solver.
*   **Optimization Variables (plausible ranges from literature):**
    *   `NPN.Is`: Saturation current. Range: `[1e-18, 1e-15]` A. Controls the turn-on voltage and holding current.
    *   `NPN.Bf`: Forward Beta. Range: `[50, 500]`. The current `10^4` is unphysically high for a parasitic BJT and is the likely cause of the hard latch-up. (Ref: A. R. Saha, "Compact Models for Integrated Circuit Design," 2015, typical parasitic BJT betas are < 100).
    *   `NPN.Ikf`: High-current beta roll-off knee. Range: `[1e-5, 1e-3]` A. Critical for modeling the holding current realistically.
    *   `R_body_sheet`: A new parameter for a proper 5-resistor distributed body network (which must be implemented first, as P4 was a no-op). Range: `[1e3, 1e6]` Ω/sq.
*   **PASS Gate:**
    *   `E_DC` < 1.5 dec (accepting a small DC degradation to achieve dynamics).
    *   `E_osc` < (20 ns)^2, i.e., `T_sim` is between 380-420 ns.
    *   The oscillation must be self-starting and stable.

**VERDICT (Q1):** Implement a 5-R distributed body network, then run a 4-parameter BBO targeting DC RMSE < 1.5 dec and oscillation period of 400±20 ns. Confidence: 0.9.

---

### Q2 — Innate LIF closure (z458 design review)

The proposed z458 experiment (4x4 grid of `snap_Is` × `R_body`) is fundamentally flawed and will likely fail. It misdiagnoses the problem. The failure to reset, as shown in z456, is due to the NPN holding current (`~10µA`) being orders of magnitude larger than the reset current provided by `R_body` (`~0.66µA @ 1MΩ`). The `snap_Is` parameter in the `snapback_subcircuit` only provides the *initial trigger current* (`Iii_body`); it has no effect on the NPN's intrinsic behavior once Vbe > 0.6V. The experiment is tuning the matchstick, not the bonfire.

*   **Grid Resolution:** A 4x4 grid is insufficient. The transition from firing to latching is a sharp boundary (a bifurcation). An 8x8 log-spaced grid would be the minimum to map this boundary, but it would still be exploring the wrong parameter space.

*   **Most Likely Failure Mode:** **(a) Latch-up.** For any `snap_Is` value large enough to trigger the NPN, the NPN will turn on and stay on, because its holding current is determined by its own internal parameters (`Is`, `Bf`), not `snap_Is`. The system will latch unless `R_body` is made so small (e.g., <50 kΩ) that it can sink >10µA, which is likely an unphysically low resistance for the device body. The experiment will find a vast region of "no fire" and a vast region of "latch-up," with almost no "single spike" region in between.

*   **Numerical Prediction of Success:** Out of 16 cells in a 4x4 grid, I predict **0 cells** will produce a clean, single LIF spike and return to baseline. It's possible 1-2 cells on the absolute edge of the firing boundary might appear to work, but they will be numerically unstable and not represent a robust operating regime. The experiment is a waste of compute because it doesn't modify the NPN holding current, which is the root cause of the failure to reset.

**VERDICT (Q2):** The z458 proposal is a KILL_SHOT. The failure mode will be near-universal latch-up. The experiment should be redesigned to sweep `NPN.Bf` vs. `R_body` instead. Confidence: 0.95.

---

### Q3 — No-A.12 publishability

Without A.12, any claim of quantitative transient accuracy is fraudulent. The publication must pivot to framing the work as an **open-source, GPU-accelerated DC model with a detailed, honest analysis of the physical gaps preventing transient fidelity.** The novelty shifts from "we solved the NS-RAM model" to "we built a platform for solving it and precisely identified what's missing."

*   **Figures/Metrics for the Paper:**
    1.  **Figure 1: DC IV Fits.** The 33-bias log-log plot, showing fwd and bwd sweeps vs. silicon data. This is the main result.
    2.  **Table 1: Honest DC RMSE.** The final table from `HONEST_BASELINE.md`, showing fwd/bwd/avg dec for the best pipeline (z446.PT_VBIC at 1.276 dec avg).
    3.  **Figure 2: The Transient Gap.** A two-panel plot. Panel (a) shows V_B(t) from z454, demonstrating successful snapback *with the phenomenological subcircuit*. Panel (b) shows V_B(t) from z456, demonstrating the failure to self-reset. The caption must explicitly state this is the model's limitation.
    4.  **Figure 3: Qualitative Oscillation.** If the Q1 experiment is successful, show the simulated V_B(t) oscillation next to a digitized trace of Mario's slide 21. The claim is only about matching the period.
    5.  **Table 2: Physics Inventory.** A condensed version of the table from `CAMPAIGN_SYNTHESIS §3`, showing what is and isn't modeled. This demonstrates rigor and honesty.

*   **Claims to AVOID:**
    *   "The model accurately reproduces ns-snapback." (No, it doesn't).
    *   "We achieved a predictive transient model." (No, it's not validated).
    *   Any specific timing claims (e.g., "time-to-fire is 1.38 ns") must be caveated as "in our phenomenological model" and not claimed to match silicon.
    *   Do not present the `snapback_subcircuit` as a physical model; call it a "behavioral block for exploring dynamic effects."

*   **Language Softening:**
    *   Instead of "validates," use "is consistent with."
    *   Instead of "predicts," use "suggests" or "qualitatively reproduces."
    *   Frame the transient results as a "sensitivity analysis" that "identifies the parasitic NPN holding current and body resistance as the key limiters to achieving self-reset."
    *   The conclusion must explicitly list "calibration against measured transient data" as the essential next step for the community.

*   **Realistic Venue:** The work is strong but incomplete. It drops a tier from premier device modeling venues.
    *   **OUT:** IEDM, VLSI Symposium (they require validated, novel device physics).
    *   **IN:** **IEEE Transactions on Computer-Aided Design (TCAD)** or **IEEE Transactions on Electron Devices (TED)** as a full paper, or an **IEEE Electron Device Letters (EDL)** short paper. It is also a strong candidate for a full oral presentation at a circuit/systems or neuromorphic conference like **ISCAS**, **ICCAD**, or **ICONS**. The contribution to the open-source modeling community is significant enough for these venues.

**VERDICT (Q3):** Publish an honest DC-focused paper with a qualitative transient analysis, targeting a strong journal like TCAD/TED. Avoid all quantitative transient claims. The paper's contribution is the platform and the gap analysis. Confidence: 0.9.
