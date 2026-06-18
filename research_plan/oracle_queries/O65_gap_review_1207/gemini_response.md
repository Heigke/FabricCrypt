# gemini response (gemini-2.5-pro) — 85s

Oracle O65, the analysis is complete. The path forward is clear.

### **Q1 — gate-crossing risk: Is the 1.42 dec real?**

**Yes, the 1.42 dec is real and represents fundamental progress, not compensatory error.** The reasoning is threefold:

1.  **Source of Improvement:** The 2.86 dec drop from z358 (4.28 dec) to z361 (1.42 dec) is not from arbitrary parameter tuning. It stems from fixing two distinct, verifiable, physical-level bugs:
    *   **R-37:** Corrected the `binunit` parameter, which fixed the impact-ionization current (`Iii`) magnitude by ~1000×. This was a model *parsing* error, not a fit.
    *   **R-41:** Enabled the parasitic body-to-nwell p-diode (`pdiode_to="vnwell"`). This introduced a physically necessary discharge path that was previously missing entirely.
    These are corrections toward a more complete physical model, the opposite of spurious fitting. The retracted 0.99 dec of z304 was spurious because it relied on a broken BJT model compensating for missing `Iii`. The current 1.42 dec is built on a sounder foundation.

2.  **P-Diode Card Honesty:** Using Sebas's p-diode card is not "fitting"; it is **model alignment**. The goal is to replicate the `ngspice` simulation. Using the same physical parameters as the reference ensures the comparison is apples-to-apples. It correctly isolates remaining errors in other parts of the model (like the BJT or channel), rather than absorbing them into a poorly-tuned diode.

3.  **Per-VG1 Error Profile:** The VG1=0.40 error (1.24 dec) being lower than VG1=0.20 (1.36 dec) is not a red flag. The R-36 log explicitly noted the original channel current error was **"STRUCTURED in Vg only"** and worse at low Vg. This was attributed to a residual Vth/subthreshold slope offset from R-29. The current error profile is consistent with this known, pre-existing residual. It is not a sign of over-correction but rather a clue that the final 1.4 dec of error is dominated by this lingering channel model inaccuracy, which is now unmasked.

The 1.42 dec is the model's honest performance after major physical omissions were rectified.

### **Q2 — single best falsifier (<1h):**

The best experiment is a **Standalone P-Diode IV Curve Validation.**

This directly and unambiguously tests the R-41 breakthrough fix in isolation, removing all confounding variables from the 2T cell.

**Procedure:**
1.  **Setup:** Use the M1 standalone testbench from R-36.
2.  **Isolate Diode:** Disable the charging mechanism (`Iii`) by setting Vd=0. This turns off the MOSFET channel.
3.  **Sweep & Measure:** Perform a DC voltage sweep on the body terminal (`Vb`) from 0V to 0.8V. Measure the current flowing out of the body and into the `vnwell` terminal.
4.  **Compare:** Plot the resulting `I-V` curve from `pyport` directly against the identical simulation performed in `ngspice`.

**Expected Outcome & Interpretation:**
The two I-V curves must overlay perfectly (within machine precision).
*   **If they match:** The p-diode implementation and parameters (Js, n, area) in `pyport` are correct. The 1.42 dec is real, and the remaining Vb offset (Q3) is definitively caused by another mechanism. Confidence in the R-41 fix is 100%.
*   **If they do not match:** The p-diode implementation itself is flawed (e.g., temperature dependence, series resistance model, area scaling). This would immediately falsify the assumption that R-41 was a complete fix and provide a direct path to the next bug.

This experiment is fast, conclusive, and directly targets the newest, most impactful change to the model.

### **Q3 — push to AMBITIOUS (<0.95):**

The `Rs` sweep has zero effect because **negligible current is flowing through the p-diode.** The body voltage `Vb` is stabilizing at 0.484V because it has found an equilibrium point with a different, weaker leakage path that turns on *before* the p-diode to `vnwell` becomes significant.

The bottleneck is the **BJT (Q1) base-emitter or base-collector junction leakage model.** The `Vb` of 0.484V is the point where the BJT-amplified `Iii` charging current is perfectly balanced by a BJT leakage current. Your `pyport` BJT model is likely missing or understating a reverse-bias leakage term that is present in the `ngspice` model. This missing leakage in `pyport` forces `Vb` to rise higher to find equilibrium.

**Highest-Value Experiment: Q1 Standalone Reverse-Bias Base Leakage Characterization.**

**Procedure:**
1.  **Setup:** Use a standalone testbench for the BJT (Q1) only.
2.  **Replicate Bias:**
    *   Fix the base voltage `Vb` at the observed equilibrium: 0.484V.
    *   Fix the emitter voltage `Vsint` at its flagship value: 0.182V.
    *   This establishes the correct forward bias on the BE junction.
3.  **Sweep & Measure:** Sweep the collector voltage (`Vd`) across its relevant range, e.g., 0.5V to 2.0V. Measure the DC base current, `Ib`. This current represents the total leakage discharging the base at this operating point.
4.  **Compare:** Plot the `Ib` vs. `Vc` curve from `pyport` against the identical simulation in `ngspice`.

**Expected Outcome & Interpretation:**
You will find that the `ngspice` `Ib` is significantly larger (more negative) than the `pyport` `Ib`. This difference is the missing leakage current. The `ngspice` model is discharging the base more effectively, thus holding `Vb` down to 0.27V.

The fix is to investigate the full `ngspice` BJT model card. You are likely missing parameters like `Isc` (B-C leakage saturation current), `Var` (Reverse Early voltage), or `Ikr` (high-injection roll-off corner current) in your `pyport` BJT implementation. Adding the correct BJT leakage physics will provide the necessary discharge path to lower `Vb` toward 0.27V and crack the AMBITIOUS gate.
