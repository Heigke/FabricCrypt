# gemini response (gemini-2.5-pro) — 110s

Excellent and detailed diagnostic work. The data you've provided—especially the near-constant Vth offset combined with the reasonably matching DIBL and subthreshold slope—is a powerful pointer. It strongly suggests a bug in a Vds-independent, zero-body-bias term within the Vth calculation.

Here is a detailed analysis addressing your questions.

### Executive Summary: The Prime Suspect

My prime suspect is an error in the calculation of the surface potential, **`phi`**, within `temp.py`. A secondary, but also plausible, suspect is a structural misinterpretation of how the pocket implant (LDE) corrections are applied to the Vth equation in `dc.py`. The `phi` error is more likely to produce a simple, constant offset of the magnitude you're seeing.

---

### (a) Prime Suspect Analysis

Let's evaluate your list of potential culprits for a **constant ~60 mV LOW shift** in Vth.

1.  **Φs (Surface Potential) - PRIME SUSPECT:**
    This is the most likely source of the error. In `temp.py`, line 241, the calculation is:
    ```python
    # temp.py:241
    phi = ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30)) + model["phin"] + 0.4
    ```
    The term `ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30))` correctly calculates the bulk potential `2 * phif`. The `model["phin"]` term is also correct. However, the hardcoded `+ 0.4` is highly suspect. The physical equation does not include this raw additive constant. This `0.4` is likely a misinterpretation of a typical value for `2*phif` itself, which is then being erroneously *added* to the calculated value.
    *   **Impact:** This introduces a large, constant error in `phi_pre`. Since `phi_pre` is a foundational input to nearly every subsequent Vth term (`sqrtPhi_pre`, `V0`, `Vth_NarrowW`, etc.), this error propagates everywhere, resulting in a significant, Vds-independent offset. A 0.4V error in `phi` would not directly translate to a 0.4V error in Vth, but through the various square-root and ratio dependencies, a ~60 mV offset is very plausible.
    *   **First Check:** **Remove the `+ 0.4` from that line in `temp.py` and re-run the `z91l` script.** I predict this will correct a large portion, if not all, of the 60 mV offset.

2.  **`vfb` (Flat-band Voltage):**
    Less likely to be the direct cause. In your `dc.py` port, the Vth equation starts from `vth0`, not `vfb`. `vfb` is primarily used in `temp.py` to calculate `vtfbphi2`, which feeds into the `Coxeff` model. While an incorrect `Coxeff` can shift the Id-Vgs curve (by altering the effective `beta`), it's a secondary effect and less likely to produce such a clean, constant Vth offset.

3.  **`K1ox`/`K2ox` Scaling:**
    Your port's logic in `temp.py` for `toxm` defaulting to `toxe` is correct. This would make `k1ox = k1` and `k2ox = k2`, which is the standard behavior when `toxm` is not specified. This is unlikely to be the bug.

4.  **Narrow-width K3 term:**
    Your estimate of ~79 mV is correct, making this a high-leverage term. The implementation in `dc.py` appears correct:
    ```python
    # dc.py:340
    Vth_NarrowW = toxe * phi_pre / (Weff + w0)
    Vth = ... + (k3 + k3b * Vbseff) * Vth_NarrowW
    ```
    This matches the manual. While an error in `Weff` or `w0` could cause a shift, the formula itself is present and correctly structured. It's a suspect, but secondary to the obvious anomaly in the `phi` calculation.

5.  **LDE (`lpe0`, `nlx`):**
    This is a strong secondary suspect, not because of a sign error, but because of its complexity and potential for structural misinterpretation. See the detailed answer in (b). The effect of `lpe0` is indeed a Vth *increase* (RSCE), not a pulldown. The term is present in your code. The absence of `nlx` is a deviation from the full BSIM4 model but may be correct for this specific card if ngspice also ignores it when unspecified.

**Conclusion for (a):** The prime suspect is the `phi` calculation in `temp.py`. The hardcoded `+ 0.4` is almost certainly an error.

---

### (b) Where in our code does LDE enter?

Yes, the port attempts to implement the pocket implant corrections. Here's the breakdown:

*   **`lpe0`:** This parameter is correctly loaded and used in `dc.py` to calculate `Tlpe1`, which is then **added** to `Vth`.
    ```python
    # dc.py:338
    T0_lpe = safe_sqrt(1.0 + lpe0 / Leff)
    Tlpe1 = (k1ox * (T0_lpe - 1.0) * sqrtPhi_pre 
             + (kt1 + kt1l / Leff + kt2 * Vbseff) * TempRatio)
    ...
    # dc.py:370
    Vth = ... + Tlpe1 ...
    ```
    The term `k1ox * (sqrt(1 + lpe0/Leff) - 1) * sqrtPhi_pre` models the Reverse Short Channel Effect (RSCE) from pocket implants. For `lpe0 > 0`, this term is positive and correctly *increases* Vth. The sign is correct.

*   **`lpeb`:** This parameter is also used, modifying the main body effect term.
    ```python
    # dc.py:363
    Lpe_Vb = safe_sqrt(1.0 + lpeb / Leff)
    ...
    # dc.py:365
    Vth = ... + (k1ox * sqrtPhis - k1 * sqrtPhi_pre) * Lpe_Vb ...
    ```
    This term also contributes to the body-bias dependence of the pocket effect.

*   **`nlx`:** A search confirms your suspicion: **`nlx` is not used anywhere in `dc.py` or `temp.py`**. It is defined as a parameter in `model_card_data.py` but its effect on Vth is not implemented. The full BSIM4 equation includes a term like `-K1 * (nlx/Leff) * sqrt(phi)`. The absence of this term would cause Vth to be *higher* than it should be, which is the opposite of what you observe. Therefore, the missing `nlx` is not the cause of your Vth being too low.

**Conclusion for (b):** The port implements the `lpe0` and `lpeb` effects, but not `nlx`. The implemented `lpe0` term correctly models RSCE (a Vth increase), so a sign error here is not the cause of the low Vth. The implementation looks fragmented compared to the single Vth equation in the manual, which increases the risk of error, but the missing `nlx` term would cause an error in the opposite direction.

---

### (c) ngspice-side hidden default?

This is a classic and frustrating part of SPICE modeling. Simulators can have subtle differences.

The most plausible "hidden" difference is how `vfb` is handled when not specified by the user.
*   **BSIM4 C-source (`b4set.c`):** If `vfb` is not given, it is calculated internally using `vth0`, `phi`, `k1`, etc. The formula is approximately `vfb = vth0 - phi - k1 * sqrt(phi)`. This makes `vfb` consistent with the other parameters.
*   **Your Port:** Your `model_card.py` uses the default `vfb = -1.0` from `model_card_data.py` if it's not in the SPICE file. This value is then used in `temp.py` to calculate `vtfbphi2`, which affects `Coxeff`.

If ngspice calculates `vfb` internally and your port uses a fixed default of -1.0, then `vtfbphi2` will be different. This alters `Coxeff`, which changes the effective gate capacitance and thus the entire Id-Vgs curve, which can manifest as a Vth shift. While not a direct additive term to Vth, it's a plausible source of systemic offset.

However, given the `+ 0.4` anomaly in your `phi` calculation, that remains a much stronger and more direct suspect than this secondary effect.

---

### (d) Single Localizing Experiment

The best experiment is to instrument the Vth calculation to see the contribution of each term. You are on the exact right track.

**Recommended Instrumentation Patch:**

Modify `dc.py` inside `compute_dc` right after the final `Vth` is assembled (around line 372). Add a print statement that fires only once (e.g., by checking if Vds is a scalar or just printing the first element of the tensor).

Here is the concrete code to add. Place it after line 372 in `dc.py`:

```python
    # --- Vth assembly (dc.py:365-372) ---
    Vth = (type_n * vth0
           + (k1ox * sqrtPhis - k1 * sqrtPhi_pre) * Lpe_Vb
           - k2ox * Vbseff
           - Delt_vth
           - T2_narrow
           + (k3 + k3b * Vbseff) * Vth_NarrowW
           + Tlpe1
           - DIBL_Sft)

    # +++ START DIAGNOSTIC PATCH +++
    if Vds.numel() > 0 and Vgs.numel() > 0: # and Vds.item() == 0.5: # Optional: trigger on specific Vds
        # Print inputs
        print("\n--- Vth Diagnostic Breakdown ---")
        print(f"  Inputs: Vgs={Vgs.item():.3f} Vds={Vds.item():.3f} Vbs={Vbs.item():.3f}")
        print(f"  phi_pre (from temp.py) = {phi_pre.item():.6f} V")
        print(f"  vth0 (from card)       = {vth0.item():.6f} V")
        print(f"  Leff={Leff.item():.3e} m, Weff={Weff.item():.3e} m")
        
        # Print term-by-term contributions to Vth
        print("\n  Vth Additive Terms (in Volts):")
        term_vth0_base = type_n * vth0
        term_body_effect_base = (k1ox * sqrtPhis - k1 * sqrtPhi_pre) * Lpe_Vb - k2ox * Vbseff
        term_sce_dvt = -Delt_vth
        term_nwe_dvt = -T2_narrow
        term_nwe_k3 = (k3 + k3b * Vbseff) * Vth_NarrowW
        term_lde_temp = Tlpe1
        term_dibl = -DIBL_Sft
        
        print(f"  1. Base vth0             : {term_vth0_base.item():+.6f}")
        print(f"  2. Body Effect (K1,K2)   : {term_body_effect_base.item():+.6f}")
        print(f"  3. SCE (DVT0/1/2)        : {term_sce_dvt.item():+.6f}")
        print(f"  4. NWE (DVT0W/1W/2W)     : {term_nwe_dvt.item():+.6f}")
        print(f"  5. NWE (K3)              : {term_nwe_k3.item():+.6f}")
        print(f"  6. LDE/Temp (lpe0,kt1)   : {term_lde_temp.item():+.6f}")
        print(f"  7. DIBL (eta0,dsub)      : {term_dibl.item():+.6f}")
        print("  ------------------------------------")
        
        # Sum of terms should equal final Vth
        computed_sum = (term_vth0_base + term_body_effect_base + term_sce_dvt + 
                        term_nwe_dvt + term_nwe_k3 + term_lde_temp + term_dibl)
        print(f"  Sum of Terms           : {computed_sum.item():+.6f}")
        print(f"  Final Vth              : {Vth.item():+.6f}\n")
        # This will print once per forward pass. To avoid flooding, you might want to
        # add a condition to only print for a specific Vds, as commented above.
    # +++ END DIAGNOSTIC PATCH +++
```

Run your `z91l_vth_dibl.py` script with this patch. It will print a breakdown for each point in the Vgs sweep. Compare the value of `phi_pre` and each of the 7 terms to a hand calculation or a known-good reference. The BSIM4 manual from the Berkeley device group is the ultimate reference. There isn't a standard "reference plot" for term breakdowns, as it's highly dependent on the specific model card and geometry. Your best bet is to manually calculate each term using the equations from the manual and your card's parameters.

This instrumentation will immediately show if `phi_pre` is off by ~0.4V and how that error cascades into the other terms, or if another term (like the K3 narrow-width effect) has an incorrect magnitude.
