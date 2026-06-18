# gemini response (gemini-2.5-pro) — 60s

### Oracle Response Packet O54

**Q1: OPTIMAL FIX-ORDER for next 8h of compute.**

The optimal order combines the cheapest structural fix with the most critical missing physics. Implement these four elements concurrently in the `pyport_v4` (P3) build:

1.  **BSIM `rbodymod=1` (N3):** This is the highest priority. `T4` identifies it as a first-order BSIM card error, not new physics. It is a zero-cost code change that corrects the DC envelope, which is a prerequisite for fitting transient behavior.
2.  **VNwell→VB Diode (Correct Polarity):** This is a known, one-line bug fix. The log confirms it is essential for producing the correct hysteresis direction by allowing the body to bleed charge when forward-biased.
3.  **Multi-τ Traps (N1):** `z311` proved this mechanism is necessary for the magnitude and shape of the transient hysteresis loop. It is the primary driver of the multi-rate behavior seen on slide-21.
4.  **SRH Generation-Recombination (N2):** This is the key mechanism (`T4`) for the ramp-rate-dependent *trigger voltage shift*, which is a separate phenomenon from the trap-driven loop shape.

This front-loaded combination maximizes the chance of passing the P3 gate (DC < 0.7 dec) and provides a credible physical basis for the P5 transient validation.

**Q2: For each phase P3 (pyport_v4) and P4 (KWS attack), what is the cheapest experiment that would FALSIFY it within 2h wall time?**

*   **P3 (pyport_v4):** Falsify the `rbodymod=1` (N3) hypothesis immediately. Re-run the DC fit on only the single worst-performing branch (`V_G1=0.2V`, which has >2 dec error) with `rbodymod=1` enabled. If this change does not produce a >1.0 dec improvement on that single curve within minutes, the "first-order error" diagnosis is wrong, and the path to passing the P3 gate is falsified.
*   **P4 (KWS attack):** Falsify the "encoding is the problem" hypothesis. Implement the simplest new encoding (e.g., rank-coded MFCC) and train for only 5-10 epochs on a 10% random subset of the Speech Commands dataset. If accuracy does not lift measurably above the 8.3% chance baseline, the bottleneck is likely the SNN architecture itself, not the input encoding. This would falsify the current "attack" plan.

**Q3: Should we ABANDON KWS as a v4.4 application target?**

**Persist, but only to clear the low-bar gate.** The `4D_critique` correctly identifies the KWS-at-chance result as the project's single most damaging internal contradiction. Abandoning it now would be a tacit admission that the device fails its advertised primary use case.

The goal is not product readiness, but credibility. Clearing the P4 gate (Acc > 25%) demonstrates a non-random result, closing the falsification loop and turning a "failure" into a "diagnosed performance boundary."

*   **Physical Argument to Persist:** The hysteretic, stateful dynamics of NS-RAM are a plausible, if unproven, substrate for temporal audio processing.
*   **Physical Argument for Caution:** The device's characteristic time constants (µs–s from traps) may be a poor match for the phoneme-level timescales in speech.

If the P4 gate cannot be passed, then abandon KWS for v4.4, but report the >25% attempt as an explicit, honest negative result.

**Q4: Are we still drifting from NO-CHEAT discipline?**

**No. The log shows strong adherence.** The team is following the pre-registered plan with high discipline.

*   **Specific evidence:** The `z310` diode bug was diagnosed, but the fix was explicitly **deferred** until the `z312` cluster job completed, respecting resource constraints and avoiding hasty work. The final log entry confirms that the v4.4 brief (`4E`) is **HELD**, respecting the gate condition from the `4D_critique`. The `z312` results were reported as a full matrix, not cherry-picked. These actions directly align with the "Honest FAIL allowed" and "gate locked" principles.
