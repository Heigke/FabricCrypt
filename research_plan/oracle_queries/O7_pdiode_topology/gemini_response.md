# gemini response (gemini-2.5-pro) — 58s

Excellent. This is a classic "predictive modeling" task based on engineering context. Here is a pre-prediction of the missing parasitic pdiode's properties and a ready-to-integrate code patch.

### 1. Diode Connection Topology (Question 1)

The diode most likely connects the floating P-body of M1 to the surrounding N-type semiconductor that provides isolation. Here are the candidates, ranked by confidence:

1.  **body ↔ substrate (GND)** - **(95% Confidence)**
    *   **Reasoning:** This represents the primary isolation junction for the floating body. In a bulk CMOS process, an NMOS transistor (with a P-type body) is typically built inside an N-well to isolate it from the P-type substrate. This N-well is usually tied to the most positive supply (Vdd) or, for noise-sensitive analog/mixed-signal circuits, a quiet ground (GND/VSS). Given the context of a memory cell, a connection to GND is highly probable. This junction has the largest area (the "bottom plate" of the transistor body) and directly matches Sebas's description of modeling the "capacitive response of the floating body." The `pdiode` model (P-body to N-well) is the correct physical structure.

2.  **body ↔ DNW (vnwell=+2V)** - **(4% Confidence)**
    *   **Reasoning:** This is physically plausible as a P-body to N-well junction. However, the model *already has* a `I_well_body` term for a DNW-to-body diode. It is highly unlikely Sebas would add a second, massive diode to the same node pair. It's more probable that this new diode represents a different physical junction that was previously omitted.

3.  **body ↔ Sint** - **(1% Confidence)**
    *   **Reasoning:** This junction is the body-to-source junction of M1. This is an intrinsic part of the MOSFET and is already modeled by the BSIM4 `Ibs` (body-source diode current) term. Adding it externally would be redundant and incorrect.

**Conclusion:** The diode connects **node B (body) to node GND (substrate/N-well tie)**. The P-anode is the body (B) and the N-cathode is ground.

### 2. Predicted SPICE Card Parameters (Question 2)

Based on the provided context and typical 130 nm process values:

*   **Area**: `A = 22e-12` (m²)
*   **Js (Saturation Current Density)**: The existing DNW diode has `Js ≈ 3.4e-7 A/m²`. This is a good baseline. A typical range is 1e-7 to 5e-6 A/m². Let's predict **`Js` ≈ 1e-6 A/m²**.
*   **n (Emission Coefficient / Ideality Factor)**: For parasitic bulk junctions, this is typically slightly greater than 1. Predict **`n` ≈ 1.2**.
*   **Cj0 (Zero-Bias Junction Capacitance per Area)**: Sebas gave a 5–10 fF range for the 22 µm² area.
    *   `Cj0_per_area = (5 to 10 fF) / 22 µm² = 0.23 to 0.45 fF/µm²`.
    *   This is a very standard value. We'll predict a value in the middle: **`Cj0_per_area` ≈ 0.35 fF/µm²**, which gives a total capacitance of `Cj0 = 7.7 fF`.
*   **Vj (Junction Potential / Built-in Voltage)**: Standard for silicon p-n junctions. Predict **`Vj` ≈ 0.7 V**.
*   **M (Grading Coefficient)**: Depends on the junction doping profile. For a bulk junction, it's typically between linearly graded (0.33) and abrupt (0.5). Predict **`M` ≈ 0.4**.

### 3. Quantitative Impact at Steady State (Question 3)

We will calculate the extra body drainage current for the predicted **body↔GND** diode at the given steady state: `Vb = 0.487 V`.

*   **Diode Voltage:** `V_diode = Vb - Vgnd = 0.487 V - 0 V = 0.487 V`.
*   **Thermal Voltage:** `Vt = kT/q ≈ 25.85 mV` at 300K.
*   **Total Saturation Current (Is):** `Is = Js * Area = (1e-6 A/m²) * (22e-12 m²) = 2.2e-17 A`.
*   **Diode Current Formula:** `I_diode = Is * (exp(V_diode / (n * Vt)) - 1)`.

**Calculation:**
1.  `n * Vt = 1.2 * 0.02585 V = 0.03102 V`.
2.  `V_diode / (n * Vt) = 0.487 / 0.03102 ≈ 15.70`.
3.  `exp(15.70) ≈ 6.58 x 10^6`.
4.  `I_diode ≈ 2.2e-17 A * (6.58e6) ≈ 1.45e-10 A = **145 pA**`.

This is a **drainage current** flowing **out of the body** into the substrate. In the context of the `R_B` residual equation (sum of currents INTO the body), this term would be subtracted: `R_B = ... - 145 pA`. This is a significant leakage path that was previously unmodeled.

### 4. Environment-Variable-Controlled Code Patch (Question 4)

Here is a single, self-contained patch for `nsram_cell_2T.py`. It introduces a helper for the diode model and uses an environment variable `NSRAM_PDIODE_TOPOLOGY` to select the connection.

```python
# nsram/nsram/bsram_port/nsram_cell_2T.py

import os
import torch
# ... other imports

# --- Start of Patch ---

# Helper function for the new parasitic pdiode model
def _pdiode_model(Vp, Vn, Vt, area_m2, js_A_per_m2, n):
    """
    Calculates the current for the parasitic pdiode.
    Current is defined as flowing from anode (Vp) to cathode (Vn).
    """
    # Clamp the forward bias voltage to prevent numerical overflow in exp()
    # A clamp at ~35*n*Vt corresponds to ~e^35, which is safe.
    v_diode = Vp - Vn
    v_diode_clamped = torch.clamp(v_diode, max=(0.95 * n * 35 * Vt))
    
    Is = area_m2 * js_A_per_m2
    current = Is * (torch.exp(v_diode_clamped / (n * Vt)) - 1.0)
    return current

# --- End of Patch ---


class NsramCell2T:
    # ... existing class definition ...

    def _residuals(self, Vd, VG1, VG2, Vsint, Vb):
        # ... existing calculations for m1, m2, Q1, etc. ...

        # --- Start of Patch ---
        
        # Read topology from environment variable, default to NONE (original behavior)
        pdiode_topology = os.getenv("NSRAM_PDIODE_TOPOLOGY", "NONE").upper()
        
        I_pdiode = torch.zeros_like(Vb) # Initialize to zero
        
        if pdiode_topology != "NONE":
            # Default parameters based on pre-prediction
            PDIODE_DEFAULTS = {
                "area_m2": 22e-12,      # 22 um^2
                "js_A_per_m2": 1.0e-6,  # Predicted Js
                "n": 1.2,               # Predicted ideality factor
            }

            if pdiode_topology == "GND":
                # Predicted case: Body (P) to Ground (N)
                I_pdiode = _pdiode_model(
                    Vp=Vb, Vn=torch.zeros_like(Vb), Vt=self.Vt, **PDIODE_DEFAULTS
                )
            elif pdiode_topology == "SINT":
                # Candidate 2: Body (P) to Sint (N)
                I_pdiode = _pdiode_model(
                    Vp=Vb, Vn=Vsint, Vt=self.Vt, **PDIODE_DEFAULTS
                )
            elif pdiode_topology == "VNWELL_EXTRA":
                # Candidate 3: An additional Body (P) to vnwell (N) diode
                vnwell_v = torch.full_like(Vb, self.vnwell_bias)
                I_pdiode = _pdiode_model(
                    Vp=Vb, Vn=vnwell_v, Vt=self.Vt, **PDIODE_DEFAULTS
                )
        
        # --- End of Patch ---

        # Calculate R_B (currents INTO body B)
        R_B = (
            + m1["Iii"]
            + m1["Igidl"] + m1["Igisl"] + m1["Igb"]
            - m1["Ibs"] - m1["Ibd"]
            - self.base_current_q1(Vb)  # Ib_Q1
            + self.well_body_current(Vb) # I_well_body
            - I_pdiode # New term: current flows OUT of body, so subtract
        )

        # Calculate R_Sint (currents INTO Sint)
        R_Sint = (
            - m1["Ids"] # Current out of Sint (as M1 source)
            + m2["Ids"] # Current into Sint (as M2 drain)
            # Note: C1 (CBpar) is handled in dynamic analysis, not DC residuals
        )

        return R_Sint, R_B

```

**How to use this patch:**

1.  **Integrate:** Copy the `_pdiode_model` helper function and the patched logic inside `_residuals` into your file.
2.  **Test:** To activate the new diode model, set the environment variable before running your script:
    ```bash
    # This will enable the most likely topology (body-to-ground)
    export NSRAM_PDIODE_TOPOLOGY="GND"
    python your_simulation_script.py
    ```
3.  **Verify:** Once the SPICE card arrives, you can replace the `PDIODE_DEFAULTS` with the exact values from the card. The topology will almost certainly be `GND`.
