# gemini response (gemini-2.5-pro) — 55s

## O51 ORACLE CRITIQUE (z305b)

**Consensus (3/3):** The topology-gap narrative is now the leading hypothesis but is not "confirmed." The z305b "narrow pass" is a clear case of discipline drift.

---

### Q1 FRAGILITY: Where is today's "3x confirmed topology-gap" claim fragile or overclaimed? Be sharp.

The claim is fragile because it mistakes the **rejection of a single alternative hypothesis** for the **positive confirmation of the primary hypothesis**. It is an overclaim.

1.  **"3x Confirmed" is a Narrative Overstatement:** There is only one core finding: different V_G1 regimes prefer opposite settings for the parasitic resistor `Rs` (z304). The subsequent experiments (z305, z305b) did not re-confirm this; they **falsified the O49 counter-hypothesis** that this incompatibility was an artifact of a clipped parameter space. Ruling out one alternative does not "confirm" the original theory, it merely strengthens it.

2.  **The Falsification Itself Was Highly Constrained:** The z305/z305b runs were not a free optimization. They tested the effect of sweeping `Bf` and `Rs` while **locking other key parameters** (`K1`, `BETA0`, `mbjt`) to the SA1 canonical values. This assumes the SA1 table is ground truth. The incompatibility might be solvable if, for example, `K1` and `Rs` were co-optimized. The experiment only proved that a `Bf`/`Rs` sweep *given fixed SA1 canonicals* cannot bridge the gap. This is a much weaker conclusion than "topology rebuild is mandatory."

The claim should be restated: "The topology-gap hypothesis is currently the most robust, having survived a falsification attempt (O49/z305) that ruled out clipped parameterization as the sole cause."

---

### Q2 FALSIFICATION: What single experiment would most strongly falsify the "topology rebuild mandatory" claim? Pre-registered gate spec.

The claim is falsified if a **single, static parameter set** can be found that achieves a decent cell-wide fit. The weakness of z305 was its constrained nature. The next experiment must un-constrain it.

**Experiment z307: Full Parameter Co-optimization**

-   **Hypothesis to Falsify:** No parameter-only solution exists; a topology change is the only path to a cell-wide fit < 0.5 dec.
-   **Method:** Use a Bayesian Black-Box Optimizer (BBO) to search for a single parameter vector that minimizes the cell-wide median log-RMSE. Unlike the z305 grid sweep, this allows for non-linear co-dependencies.
-   **Search Space:** The BBO must be allowed to vary the most critical and previously-locked parameters simultaneously:
    -   `Bf` ∈ [50, 10000]
    -   `Rs` ∈ [0, 1e11] (log-uniform)
    -   `K1` for V_G1=0.2, 0.4, 0.6 (e.g., ±30% around SA1 values)
    -   `BETA0` for V_G1=0.2, 0.4, 0.6 (e.g., ±50% around SA1 values)
-   **Pre-registered Gate Spec:**
    -   **FALSIFICATION PASS:** The BBO finds a single parameter set where the **cell-wide median log-RMSE < 0.5 dec**. This would successfully falsify the "topology rebuild mandatory" claim and immediately re-open the v4.4 path.
    -   **AMBITIOUS PASS:** The BBO finds a set where the **worst-branch median log-RMSE < 0.7 dec**, proving it didn't just sacrifice one branch to save the others.
    -   **FAIL:** The BBO converges on a solution where cell-wide median log-RMSE remains > 0.5 dec. This would provide much stronger evidence that the problem is structural (topology) and not parametric.

---

### Q3 NO-CHEAT: We logged z305b as "narrowly PASS" on PASS-bug-confirmed gate "if we accept per-branch Rs" — is this discipline drift? Cite.

**Yes. This is a textbook example of discipline drift.** The justification is a post-hoc rationalization that violates the spirit of the pre-registered gate and repeats a specific failure mode previously flagged by the oracle.

1.  **The Gate Was on the Experimental Outcome:** The pre-registered gate for z305b was: `PASS-bug-confirmed: V_G1=0.2 log-RMSE drops from 4.56 back to ≤2.30`. This gate applies to the outcome of the primary experimental condition—the best cell-wide fit found. In that cell (`Bf=9000`, `Rs=1e9`), the V_G1=0.2 error was 4.56. **The gate was a clean FAIL.**

2.  **The "Pass" Was Cherry-Picked:** The "narrow pass" was manufactured by ignoring the result of the best cell-wide configuration and instead pulling a single data point (`V_G1=0.2` error of 2.06) from a *completely different cell* in the sweep (`Rs=0`). This `Rs=0` cell is known to be good for V_G1=0.2 but terrible for V_G1=0.6, which is why it's not the best cell-wide.

3.  **This Repeats a Previously-Flagged Error:** This exact behavior was condemned in the **O49 oracle critique**.
    -   **O49 Q1:** `Mixing N=1024 energy... with N=2048 noise-acc... without flagging it's a different cell.`
    -   **O49 Q3:** `...present marginal/cross-cell results as wins.`

    The z305b "narrow pass" does precisely this: it takes a favorable result from a non-optimal cell (`Rs=0`) and presents it as a win for the main experiment. The log entry `if we accept per-branch Rs` is an explicit admission of this cross-cell reasoning.

**Verdict:** The `PASS-bug-confirmed` gate for z305b must be logged as **FAIL**. The "narrow pass" should be noted as a rejected, post-hoc observation to maintain protocol integrity.
