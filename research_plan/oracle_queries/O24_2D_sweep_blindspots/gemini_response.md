# gemini response (gemini-2.5-pro) — 38s

Here is a ranked audit of untested knobs and a prognosis for your 2T NS-RAM cell model.

### 1. Ranked List of Potential Improvements

1.  **Early Effect (VAF): Critical.** For a lateral NPN with a low-doping, wide base, base-width modulation is a first-order effect, not a secondary one. The default VAF (forward Early voltage) is likely for a vertical BJT and is far too high. This is the most probable source of significant remaining error.
2.  **Cross-Validation (ngspice): Urgent Process Step.** Before any further tuning, confirm the 0.795 dec optimum in ngspice. A discrepancy could indicate a PyTorch-specific model implementation detail or a solver artifact, rendering further PyTorch-side tuning moot.
3.  **Knee Current (IKF): High Impact.** High-injection gain roll-off is very likely at mA-level currents. The parasitic BJT's gain (Bf) will decrease as collector current (Ic) approaches IKF. This directly impacts the device's saturation behavior, which is critical for RAM cell operation.
4.  **Residual Analysis (by row/bias): High Impact Analysis.** The fact that 5 rows consistently show high error points to a *structural* mismatch at specific bias points (e.g., high Vgs, low Vds). Analyze if these are the points where the parasitic NPN is most active. This analysis will guide whether VAF (output conductance) or IKF (saturation) is the dominant missing effect.
5.  **Leakage Current (Ise, NE): Medium Impact.** These parameters govern the non-ideal base-emitter diode current, primarily affecting the turn-on characteristic and low-Vbe leakage. This could clean up errors in the transition region but is unlikely to be the dominant error source in the fully on state.

### 2. Sweep Ranges and Residual Signatures

| Rank | Parameter(s) | Sweep Range | Expected Residual Signature if it Matters |
| :--- | :--- | :--- | :--- |
| 1 | **VAF** | **1V to 50V** (sweep low). Default is often >100V. | Error will decrease specifically in the BJT saturation region (high Vd, where output conductance `d(Ic)/d(Vce)` is high). The model currently likely underestimates current at high Vd. |
| 3 | **IKF** | **0.1mA to 10mA**. Center the sweep on your cell's peak current. | Error will change at high current levels (high Vg2, high Vd). If the model over-predicts current here, it's because it's missing the gain roll-off. |
| 5 | **Ise, NE** | **Ise**: 1e-12 to 1e-9 A<br>**NE**: 1.5 to 2.5 | Error will improve at low Vbe, near the BJT turn-on threshold. This will appear as a better fit in the sub-threshold or weak-inversion region of the cell. |

*Note: BSIM4 secondary effects (IGCMOD, etc.) and solver tweaks (GMIN) are very low-probability candidates for large improvements and should be pursued only after exhausting the BJT parameters.*

### 3. Key Parameter Coupling

**Yes, VAF is strongly coupled with Bf.** The Early effect modifies collector current via the `(1 + Vce/VAF)` term. This term multiplies the base current, which is scaled by Bf. A 1D sweep of VAF at a fixed Bf is misleading.

**Action:** Perform a **2D sweep of Bf × VAF**. The true optimum will lie in this plane, just as the previous one lay in the Bf × Is plane. A plausible region is Bf=[1e4, 5e4] and VAF=[2V, 20V].

### 4. Prognosis: Can you improve by >0.2-0.3 dec?

**Yes, very likely.** An error of 0.795 dec is still a factor of ~6 mismatch. Given the extreme sensitivity of lateral NPNs to the Early effect—a knob you have not yet tuned—it is highly probable that a Bf×VAF sweep will uncover a new optimum significantly below 0.7 dec. The discovery of the Bf×Is pocket proves your methodology is sound but that 1D sweeps are insufficient for this coupled model.

Finding the correct VAF/IKF could plausibly bring the median error down to **~0.4-0.5 dec**, which is closer to an irreducible model-vs-silicon floor for a compact model.
