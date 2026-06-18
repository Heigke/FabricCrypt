# gemini response (gemini-2.5-pro) — 150s

Excellent, this is a classic and subtle BSIM4 implementation bug. The provided data and code are sufficient to pinpoint the error. The 1-decade subthreshold current error is a major fidelity gap, and this fix should close it almost entirely.

Here is the analysis and the proposed solution.

### Diagnosis

The 12% error in the subthreshold slope (`n`) and the corresponding high subthreshold current is caused by using the wrong short-channel effect attenuation factor (`Theta0`) in the `n`-factor calculation.

The code calculates two distinct `Theta` terms, but reuses one of them incorrectly:
1.  `Theta0`: Calculated in `dc.py:280` based on `dvt0` and `dvt1`. This term correctly models the Vth roll-off due to short-channel effects (`Delt_vth`).
2.  `theta0vb0`: Calculated in `dc.py:328` based on `dsub`. This term correctly models the Drain-Induced Barrier Lowering (DIBL) effect.

The subthreshold ideality factor `n` is physically determined by the ratio of capacitances, including a short-channel correction term `Cdsc`. According to the BSIM4 manual (and the physics of the device), this correction term should be modulated by the DIBL effect.

The current implementation at `dc.py:354` incorrectly uses the Vth roll-off `Theta0` to modulate `Cdsc`, instead of the DIBL-related `theta0vb0`.

```python
# dc.py:351-354
tmp2 = nfactor * tmp1
tmp3 = cdsc + cdscb * Vbseff + cdscd * Vds
# BUG: This line uses 'Theta0' (from dvt1)
tmp4 = (tmp2 + tmp3 * Theta0 + cit) / coxe 
```

This is a subtle but critical error. While both `Theta0` and `theta0vb0` are small exponential terms that decay with `Leff`, they are derived from different physical parameters (`dvt1` vs. `dsub`) and have different values. Using the wrong one results in an incorrect value for `n`, which in turn creates the incorrect subthreshold slope and the large current error seen in the plots.

The fact that the internal `Vth` variable matches ngspice (`z91m` data) while the external constant-current `Vth` does not (`z91l` data) is the classic signature of a correct `Vth` calculation but an incorrect subthreshold current behavior.

### Answers to Your Questions

**1. Where do you bet the 12% lives?**

The 12% error lives in the **`Theta0` definition** used for the `n` factor calculation. Specifically, it's a variable confusion bug:

*   **Correct:** The `n` factor's `cdsc` term should be modulated by the DIBL short-channel effect, which is calculated as `theta0vb0` from the `dsub` parameter.
*   **Incorrect (Current Code):** The code uses `Theta0`, which is calculated from the `dvt1` parameter for Vth roll-off.

The other candidates are less likely:
*   `Xdep`: The `phi` fix was correct, and `Xdep`'s dependence on `Vbs` is standard. An error here would also affect `Vth`, which is known to be correct.
*   `coxe`: This is a fundamental parameter; an error here would affect the entire I-V curve, not just the subthreshold slope.
*   `mstar mixing`: `mstar` affects the transition to strong inversion, but the core subthreshold slope is set by `n * Vtm`. The error is in `n` itself.
*   `Vbseff smoothing`: At `Vbs=0`, this has a negligible effect.

**2. Which BSIM4 reference variable does pyport's `n` formula use for `Theta0`?**

The pyport `n` formula at `dc.py:354` uses the variable `Theta0`. This variable is defined at `dc.py:280` and is derived from the `dvt1` parameter. This is the term intended for the short-channel Vth roll-off correction (`Delt_vth`). **This is the wrong variable for the `n` calculation.** It should be using `theta0vb0`, which is derived from the `dsub` parameter.

**3. One-line A/B test:**

To isolate and confirm this is the source, temporarily set `cdsc = 0` in the `M2_130bulkNSRAM.txt` model card and re-run the simulation.

```
# In M2_130bulkNSRAM.txt
# Change this line:
+cdsc = 2.4e-4                 cdscb = 0                     cdscd = 0
# To this:
+cdsc = 0                      cdscb = 0                     cdscd = 0
```
This will nullify the `tmp3 * Theta0` term in the `n` calculation. If this is the source of the error, the subthreshold slope discrepancy between pyport and ngspice should be significantly reduced (though not eliminated, as `nfactor` still contributes).

**4. If a fix is obvious from the code: what file:line change closes the 12%?**

Yes, the fix is obvious and is a one-line change.

**File:** `dc.py`
**Line:** 354
**Change:** Replace the variable `Theta0` with `theta0vb0`.

```python
# --- BEFORE (BUG) ---
# dc.py:354
tmp4 = (tmp2 + tmp3 * Theta0 + cit) / coxe

# --- AFTER (FIX) ---
# dc.py:354
tmp4 = (tmp2 + tmp3 * theta0vb0 + cit) / coxe
```

This change correctly applies the DIBL-related short-channel effect to the `Cdsc` term in the subthreshold ideality factor calculation, aligning the model with the underlying device physics described in the BSIM4 documentation. This should correct the subthreshold slope and largely eliminate the 1-decade current error in the deep subthreshold region.
