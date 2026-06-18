# gemini response (gemini-2.5-pro) — 112s

Excellent and thorough query. This is a classic, fascinating problem in compact modeling where device physics, numerical methods, and model parameterization intersect. Based on the provided context, here is a detailed analysis.

### Executive Summary

The primary mechanism for the snapback is a **positive feedback loop involving impact ionization (`Iii`) charging the floating body**. Your PyTorch port correctly implements the necessary physics, but the simulation fails to trigger the snapback because the **impact ionization current is quantitatively suppressed by several orders of magnitude** due to the high value of the `beta0` parameter (`~19`) in the provided model cards.

The snapback is initiated when the gain of this feedback loop exceeds 1. The loop gain is determined by how much `Iii` increases for a given increase in body voltage `Vb`. This gain is exponentially sensitive to `beta0`. LTSpice is likely producing the snapback because its internal calculation, or some other interacting parameter, results in a higher effective `Iii` than your faithful port does with the given high `beta0`.

The arclength solver is not at fault; it correctly traces the monotonic curve that the physics model provides. There is no fold for it to detect.

---

### 1. (a) The Physical Mechanism for Snapback

The snapback is a manifestation of a fold bifurcation caused by a positive feedback loop centered on the floating body potential, `Vb`.

**The Loop:**

1.  **Initiation (Priming the Pump):** At low `Vd`, the deep N-well diode (`I_well_body`), which is forward-biased (`vnwell`=2V, `Vb`~0V), injects a small but non-trivial current into the body. This current raises `Vb` slightly and establishes the initial operating point.
2.  **Gain Stage (Impact Ionization):** As `Vd` increases, the drain-to-source voltage of M1 (`Vds_M1`) increases. Once `Vds_M1` exceeds the saturation voltage `Vdseff_M1`, the high-field region near the drain generates electron-hole pairs via impact ionization. The holes are swept into the P-type body, creating a charging current `Iii`.
3.  **Positive Feedback:** This `Iii` current further increases `Vb`. A rising `Vb` has two critical effects:
    *   It reduces the threshold voltage `Vth` of M1 due to the forward body effect (`Vbs_M1 = Vb - Vsint` becomes less negative).
    *   A lower `Vth` causes the channel current `Ids_M1` to increase.
    *   Since `Iii` is proportional to `Ids_M1`, the increase in `Ids_M1` leads to an even larger `Iii`.
4.  **Runaway (Snapback):** When the loop gain becomes greater than one—meaning a small increase in `Vb` produces an `Iii` increase large enough to overcome all discharging currents and cause an even larger increase in `Vb`—the system becomes unstable. `Vb` rapidly snaps up to a higher stable voltage (~0.7-0.8V), where it is clamped by the now strongly forward-biased body-source diode and the parasitic NPN's base-emitter junction.
5.  **High Current State:** In this new state, `Vth` is low, `Ids_M1` is high, and the parasitic NPN is active (`Vbe = Vb` > 0.7V), conducting a significant collector current `Ic_Q1` from drain to ground. The total terminal current `Id = Ids_M1 + Ic_Q1 + ...` jumps by decades.

**Why Your Port Fails to Snap Back:**

The magnitude of the impact ionization current is governed by the BSIM4 equation (manual §6.1):
`Iii = (Ids_pre_scbe / Leff) * (alpha0 + alpha1*Leff) * (Vds - Vdseff) * exp(-beta0 / (Vds - Vdseff))`

The critical term is `exp(-beta0 / (Vds - Vdseff))`.
*   Your model cards (`M1_130DNWFB.txt`, `2Tcell_BSIM_param_DC.csv`) specify `beta0` values between **10.75 and 20**.
*   In the snapback region (`Vd` ~ 1.2V), `Vds - Vdseff` is typically in the range of 0.2V to 0.5V.
*   For `beta0 = 19` and `Vds - Vdseff = 0.3V`, the exponential term is `exp(-19 / 0.3) = exp(-63.3) ≈ 10⁻²⁸`.

This exponential term **completely suppresses the impact ionization current**, keeping it in the femtoampere range or lower. The feedback loop gain never approaches 1, `Vb` never runs away, and the I-V curve remains monotonic. LTSpice must be calculating a significantly higher `Iii`, implying either a different effective `beta0` or a different `Vds - Vdseff`. Given the faithfulness of your `compute_dc.py` port, the `beta0` term is the most likely discrepancy.

### 2. (b) Loop-Gain Instrumentation

The condition for the fold bifurcation is that the Jacobian of the KCL residual vector with respect to the state variables becomes singular. For the body node, the KCL is `R_B(Vb, Vsint) = I_charge(Vb, ...) - I_discharge(Vb, ...) = 0`. The instability is primarily driven by the body voltage dynamics.

The quantity to monitor is the partial derivative of the body-node KCL residual with respect to the body voltage: **`∂R_B / ∂Vb`**.

*   In a stable regime, `∂R_B / ∂Vb` is negative. This indicates negative feedback: if `Vb` increases, the net current into the body `R_B` becomes negative, pushing `Vb` back down. This is because the discharging currents (diodes, NPN base) increase faster with `Vb` than the charging currents.
*   At the snapback point, **`∂R_B / ∂Vb` crosses zero and becomes positive**. This indicates positive feedback: an increase in `Vb` now leads to a net positive current, pushing `Vb` even higher.

**Your implementation already calculates this term.** The `[1, 1]` element of your finite-difference Jacobian in `_jacobian_finite_diff` is exactly `∂R_B / ∂Vb`.

**Actionable Instrumentation:**
Log the value of `J[..., 1, 1]` at each converged Newton step during the `arclength` sweep. Plot it against `Vd`. You will see it is always negative in your current simulation. For a simulation that snaps back, you would see it approach and cross zero at the fold.

### 3. (c) Solver-Side False Negative

It is highly unlikely that the arclength solver is at fault.

*   The solver's job is to trace the solution manifold defined by the physics equations `R(x) = 0`. If the physics equations produce a monotonic I-V curve (because the loop gain is < 1), there is no "S-shape" or fold in the manifold to trace or detect.
*   Your solver correctly reports `n_folds = 0` because the tangent vector's Vd-component (`t[2]`) never changes sign, which is the correct behavior for a monotonic curve.
*   While it's theoretically possible for a large `ds` to jump over a very narrow fold, your adaptive step-size logic based on tangent rotation (`cos_rot`) is designed to prevent this by shrinking `ds` near sharp turns. The root cause is the lack of a turn in the first place.

You can falsify this by running the decisive experiment below. If that experiment produces a snapback, you will see your arclength solver report `n_folds = 1` (or 2, for an S-shape), confirming it works correctly when a fold is present.

### 4. (d) Single Decisive Experiment

The core hypothesis is that `Iii` is being suppressed by an overly large `beta0`. The most direct way to test this is to manually lower `beta0` and observe the result.

**Experiment Protocol:**

1.  **Modify PyTorch Simulation:** In your `z91g_two_model_validation.py` script, activate the `NSRAM_BETA0_TEST` override you've already built. Run a single curve that should exhibit strong snapback (e.g., VG1=0.4V, VG2=0.0V).
    *   Set the environment variable: `NSRAM_BETA0_TEST=2.0 python z91g_two_model_validation.py`
    *   This will force `beta0 = 2.0` for both M1 and M2 inside `compute_iimpact`.

2.  **Expected PyTorch Outcome:**
    *   The simulation will now produce a sharp, multi-decade jump in `Id` vs. `Vd`.
    *   The `arclength` solver will report `arclen_n_folds > 0` in the output dictionary.
    *   A plot of the predicted `Id` will qualitatively match the shape of the measured data (a low-current floor followed by a sharp vertical rise).
    *   If you instrument `J[1,1]` as suggested in (b), you will see it cross zero.

3.  **Confirm with LTSpice Probes (Oracle Access):**
    *   In your LTSpice deck (`2tnsram_simple.asc`), run a DC sweep for the same bias point (VG1=0.4, VG2=0.0).
    *   Plot the following traces against `V(D)` (your Vd sweep):
        1.  `I(M1.b)`: This is the BSIM4 substrate current, which is dominated by `Iii`.
        2.  `V(B)`: The floating body potential.
        3.  `I(Q1.c)`: The collector current of the parasitic NPN.
    *   **Expected LTSpice Outcome:** You will observe that at the exact `Vd` where the `Id` snaps, `I(M1.b)` will abruptly increase by many orders of magnitude, and `V(B)` will jump from a low voltage (< 0.5V) to ~0.7-0.8V. This confirms that `Iii` is the trigger and the body potential is the state variable that becomes unstable.

This experiment directly manipulates the hypothesized cause (`beta0`) in your model and compares the internal state variables (`Iii`, `Vb`) with the "ground truth" from SPICE, providing a definitive confirmation of the mechanism.

---
**Ranking of Plausible Mechanisms:**

1.  **`Iii` Gain Suppression (Highest Confidence):** This is the most direct explanation. The exponential sensitivity of `Iii` to `beta0` is a well-known feature of the model, and the values in your cards are large enough to quench the effect entirely.
2.  **N-Well Diode as an Enabler (High Confidence):** While not the gain element itself, the `I_well_body` current is crucial for pre-charging `Vb` to a level where the `Iii` feedback can take over. The `mbjt` scaling in the CSV, which turns this off for VG1=0.2V, strongly correlates with the weaker/absent snapback in those measured curves.
3.  **Other Mechanisms (Low Confidence):** PSCBE, GIDL, and junction breakdown are less likely to be the *primary trigger* for this classic `Iii`-driven snapback, although they contribute to the overall KCL and can modify the exact voltage at which the snapback occurs. They are part of the `I_discharge` term that the `Iii` gain must overcome.
