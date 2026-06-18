# gemini response (gemini-2.5-pro) — 65s

## O56 — 6h Critique Cycle

### Oracle 1: Fragility

The "v4.4 LOCKED" framing is fragile in its use of the word "shippable." While the 4E brief's transparency is commendable—it explicitly discloses the bimodal fit and the inert configuration flags in Section 6—the term "ship" implies a degree of engineering completeness and robustness that we know we do not have.

The fragility is this: **The v4.4 model is a point solution, not a platform.**

The two headlines (HDC and RNG) are valid but brittle achievements. They exist at specific, fixed operating points. The infrastructure gap revealed by the z313 bisection means that the compact model is not actually tunable. A customer licensing this "IP" would be unable to re-target the device for a different energy/performance trade-off, because the very knobs intended for that purpose (`R_body`, `avalanche`) are disconnected.

Therefore, the claim to be "shippable" is defensible *only if* "shipping" means "archiving a technical report on two benchmark victories." It is *not* defensible if it implies "delivering a usable, generalizable compact model for integration." The latter is what the "IP-licensable macro" framing suggests.

**Correction:** The executive summary and closing should be rephrased. Replace "We ship v4.4" with language that frames it as a milestone report: "The v4.4 development milestone delivers two validated application benchmarks from the NS-RAM device. These results establish a baseline for performance and utility, while also revealing critical infrastructure gaps in the underlying compact model that must be addressed before the IP can be considered general-purpose." This is less triumphant but more accurate and defensible.

### Oracle 2: Falsification

The strongest, most valuable claim in the v4.4 brief is the HDC accuracy headline: **83.86% on UCI-HAR**. This claim synthesizes the device physics into a system-level outcome. The most effective falsification is one that tests the robustness of this claim against the model's known weaknesses.

The z304 model's catastrophic failure at the V_G1=0.2 branch is its greatest liability. The HDC success was achieved at V_G1=0.3. This suggests the system-level performance may be critically dependent on avoiding the model's "bad" regions.

**Experiment `z319_hdc_op-point_fragility` (1-2h):**
1.  **Hypothesis:** The headline HDC accuracy is an artifact of a well-behaved operating point (V_G1=0.3V) and will collapse when forced into a region where the underlying DC model is known to be poor (V_G1=0.2V).
2.  **Procedure:** Re-run the locked HDC experiment (`N=16384`, `n=4` seeds for speed) with only one change: set the NS-RAM cell `vg1` parameter from `0.3` to `0.2`. All other aspects (dataset, hypervector dimension, classifier) remain identical.
3.  **Pre-registered Gate:**
    *   **GATE:** The mean accuracy over `n=4` seeds must be compared to the locked 83.86% baseline.
    *   **PASS-CONSERVATIVE:** `mean_acc(V_G1=0.2) > 70%`. A >13pp drop is a massive regression but still potentially usable.
    *   **FALSIFIED:** `mean_acc(V_G1=0.2) <= 70%`. This would confirm that the headline result is not generalizable across the device's operating range and is critically dependent on cherry-picking a functional region of a broken model.

This experiment directly confronts the system-level claim with the device-level reality. A FAIL verdict would severely undermine the "IP-licensable" narrative.

### Oracle 3: No-Cheat

Discipline since O55 has been **HIGH** in execution but contains a subtle **DRIFT** in synthesis.

**Evidence of Discipline (NO-CHEAT):**
The team has followed the O55 corrective actions precisely.
-   `context.md @ 12:25`: "Bootstrap z304 baseline → real CI" was logged as an action.
-   `context.md @ 12:30`: The bimodal nature of z304 was immediately identified and the log explicitly states, "DO NOT say 'z304 0.99 dec baseline' — it's bimodal."
-   `research_plan_4E_v4.4_brief.md`: Section 3 correctly reports the bimodal distribution, the V_G1 stratification, and the CI from z318.

This is excellent procedural discipline. The raw findings are being reported honestly.

**The Subtle Drift (The "Cheat"):**
The cheat is an **omission of connection**. The 4E brief presents the HDC headline (Section 4) and the bimodal DC model (Section 3) as two separate facts. It never explicitly states the crucial link: **the headline was achieved by selecting an operating point that conveniently avoids the model's known catastrophic region.**

-   `z312_hdc_n16k_summary.json` shows the HDC run used `vg1: 0.3`.
-   `z318_baseline_ci_summary.json` shows the z304 model has a `~4.7 dec` error in the `vg1: 0.2` branch.

The current framing allows the reader to believe "83.86% accuracy" is a general property of the v4.4 model, when in fact it is a property of one specific, well-behaved island in a sea of poor fit. This is not a direct lie, but it is a failure to synthesize the full, honest context. It violates the "all-cards-on-table" register promised in the brief's preamble.

**Correction:** Add one sentence of synthesis to Section 4.1 of the brief, directly after the headline table: "This accuracy is achieved at a `V_G1=0.3V` operating point, which avoids the `V_G1=0.2V` region where the underlying z304 DC model has a known catastrophic fit error (~4.7 dec)." This closes the loop and makes the framing fully defensible.
