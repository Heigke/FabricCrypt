# gemini response (gemini-2.5-pro) — 114s

Excellent and highly detailed query. You've done a thorough job of exploring the parametric space and have correctly identified that the issue is likely structural or physical, not just a matter of tuning.

Here is a detailed analysis based on the provided artifacts.

### (a) Sanity-Check on Target: Does Sebas's Fit Reproduce the Jump?

**Yes, absolutely.**

Looking at your provided `sebas_iv_fits_main.png`, specifically the rightmost panel (VG1 = 0.25 to 0.45 V), the solid lines (Sebas's LTSpice simulation) clearly exhibit the S-shaped negative differential resistance (NDR) and hysteretic snapback. They bend up sharply at high Vd, closely tracking the measurement data (symbols).

This is a critical finding. It confirms:
1.  The high-current jump is **not** a measurement artifact.
2.  A SPICE-level DC simulation with the correct models and topology **can and does** reproduce this behavior.
3.  Your model is missing a fundamental piece of physics or a key topological connection that is present in the LTSpice deck.

### (b) Best-Bet Structural Hypothesis & (d) Single Discriminating Experiment

After a careful review of your provided `2tnsram_simple.asc` netlist, I have identified a **critical topological discrepancy** between the LTSpice schematic and your Python model implementation. This is almost certainly the primary cause of the missing snapback.

#### Primary Hypothesis: M2 Body/Substrate Connection

Your model assumes that the body of M2 is connected to the floating body of M1. From `nsram_cell_2T.py`:
-   Docstring: "Two NMOS (M1 short, M2 long) share floating body B."
-   `_residuals` function: The call `_eval_mosfet(..., Vb=Vb, ...)` for M2 explicitly connects its body to the floating node `Vb`.

However, the LTSpice schematic (`_normalised/2tnsram_simple_asc.txt`) tells a different story. M1 and M2 are both `nmos4` symbols. In LTSpice, for a discrete 4-terminal NMOS, an unconnected substrate/body terminal defaults to the global ground (node `0`). The schematic does not show an explicit connection from the body of M2 to the body of M1 (node `B`). Therefore, the correct topology is:

-   **M1.Body:** Floating (Node `B`, connected to Q1.Base).
-   **M2.Body:** **Grounded** (Node `0`).

**Consequences of this topological error:**
1.  **Incorrect Body KCL:** Your `R_B` equation incorrectly includes all of M2's body currents (Iii, GIDL/GISL, Igb, body diodes). These currents should not flow into the floating node `B`; they should flow to/from ground. This drastically alters the charging/discharging dynamics of the floating body.
2.  **Incorrect M2 Operation:** Your model calculates a body effect for M2 (`Vbs_M2 = Vb - 0`), which is incorrect. With a grounded body, M2 has no body effect (`Vbs_M2 = 0`).

This is the single most likely reason your feedback loop is not triggering. The body node `B` in your model has far more leakage paths (all of M2's diodes and leakages) than it should, preventing `Vb` from rising enough to turn on the parasitic BJT.

#### Single Discriminating Experiment (Answer to d)

The single most discriminating experiment is to correct the M2 body topology in your code.

**Action:** Modify your `nsram_cell_2T.py` `_residuals` function.

1.  **Change the M2 evaluation call:**
    In the call to `_eval_mosfet` for M2, pass `Vb=zero` instead of `Vb=Vb`.

    ```python
    # In _residuals function, around line 300
    # ...
    # M2: D=Vsint, G=VG2, S=0, B=GND
    m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero, Vb=zero, # <-- CHANGE HERE
                      junctions=j_M2, overrides=P_M2)
    # ...
    ```

2.  **Correct the Body KCL (`R_B`) equation:**
    Remove all terms related to M2 from the `R_B` summation, as they now flow to ground, not the floating body.

    ```python
    # In _residuals function, around line 370
    # ...
    # ---- Body KCL: currents INTO B ------------------------------------ #
    R_B = (
        m1["Iii"] # + m2["Iii"]  <-- REMOVE
        + m1["Igidl"] + m1["Igisl"] # + m2["Igidl"] + m2["Igisl"] <-- REMOVE
        + m1["Igb"] # + m2["Igb"] <-- REMOVE
        - m1["Ibs"] - m1["Ibd"]
        # - m2["Ibs"] - m2["Ibd"] <-- REMOVE
        - Ib_Q1
        + I_well_body
    )
    # ...
    ```

Rerun your `z91g_two_model_validation.py` script with these changes.

**Expected Outcome:** With M2's leakage paths removed from the floating body's KCL, the body node `B` will be much more sensitive to charging currents from M1's impact ionization (`Iii_M1`) and the well-body diode (`I_well_body`). This should allow `Vb` to rise to ~0.7V at a much lower `Vd`, strongly turning on the parasitic BJT (Q1) and producing the high-current snapback jump.

#### Secondary Hypothesis: Self-Heating (SHMOD)

If the topology correction alone is insufficient, the next most likely candidate is **self-heating**. You correctly identified this as a missing piece of physics.

-   **Mechanism:** At high Vd and high current, power dissipation (`P = Vd * Id`) in M1 heats the device. This temperature rise exponentially increases the BJT's saturation current (`Is`), causing it to turn on more strongly and conduct significantly more collector current. This creates a powerful electro-thermal positive feedback loop, which is a classic mechanism for snapback in BJTs and floating-body MOSFETs.
-   **Evidence:** The `M1_130DNWFB.txt` and `M2_130bulkNSRAM.txt` files do not specify `RTH0` or `CTH0`, but BSIM4 has default values. It's a standard physical effect that would be included in a commercial simulator like LTSpice.
-   **Status:** Your port does not implement it. This remains the most significant *physical model* gap.

### (c) Measurement Artifact?

As determined in (a), this is highly unlikely. Sebas's own SPICE simulation successfully reproduces the effect, indicating it is a real, DC-simulatable device behavior.

### Summary and Path Forward

1.  **Immediate Action:** Implement the M2 body connection fix described above. This is a clear bug in your model's topology relative to the ground-truth schematic. I am highly confident this will unlock the snapback behavior.
2.  **Next Step (if needed):** If the snapback appears but the shape or trigger voltage is still off, the next step would be to implement a first-order self-heating model (BSIM4's `SHMOD=1`). This would involve calculating power dissipation, estimating a temperature rise `ΔT = P * RTH0`, and updating temperature-dependent parameters like `Is` within your Newton loop.
3.  **Lower-Priority Items:** The other hypotheses (`RBODYMOD`, `mbjt` semantics) are less likely to be the primary cause of a missing 6-decade current jump. Address the fundamental topology error first.
