# gemini response (gemini-2.5-pro) — 149s

Excellent debugging work. You've correctly isolated a subtle and contradictory phenomenon and ruled out the most common culprits. The answer lies in a logical contradiction within your own analysis, which is captured in a code comment.

The bug is not in the `Vth` or `voff` calculations you've scrutinized, but in the "fix" you implemented for the subthreshold slope factor `n`. Your reasoning was flawed, causing you to implement the exact opposite of the required change.

Let's dissect the key comment from `compute_dc_vth.py`:

> `A.5.f (2026-05-02): ... Reusing the Vth-DVT Theta0 (T0_th = dvt1·Leff/lt1) makes the exponent too large for long L, suppressing the cdsc contribution and pushing pyport n ~12% high vs ngspice`

This statement contains a fundamental contradiction. Let's break it down:

1.  **"exponent too large"**: This means the argument to the `exp()` function, `T0_th = dvt1 * Leff / lt1`, is a large positive number.
2.  **Effect on `Theta0`**: The full term is `exp(-T0_th)`. If `T0_th` is large and positive, then `exp(-T0_th)` is a very small number, approaching zero.
3.  **"suppressing the cdsc contribution"**: The `cdsc` term in the `n` calculation is multiplied by `Theta0_n`. If `Theta0_n` is very small (as per step 2), then the `cdsc` contribution is indeed suppressed. This part is consistent with step 1.
4.  **"pushing pyport n ~12% high"**: This is the contradiction. The formula for `n` is roughly `n ≈ 1 + (nfactor*Cdep/Coxe) + (cdsc*Theta0_n/Coxe)`. If the `cdsc` contribution is suppressed (i.e., `Theta0_n` is near zero), then `n` will be *smaller*, not larger.

The plot confirms that your pyport's subthreshold slope (S=76.2 mV/dec) is higher than ngspice's (S=72.5 mV/dec). Since `S ∝ n * Vtm`, this means your pyport's `n` is indeed too **high**.

Therefore, the correct description of the original problem was:
*   `n` was too high.
*   This means the `cdsc` contribution was too **large**.
*   This means `Theta0_n` was too **large**.
*   This means the exponent argument `T0_th_n` was too **small**.

### The Incorrect "Fix"

To fix an `n` that is too high, you needed to make `Theta0_n` smaller by **increasing** its exponent argument (`T0_th_n`).

Instead, your implemented fix does the opposite:

```python
# compute_dc_vth.py:368
T0_th_n = 0.5 * dvt1 * Leff / lt1.clamp_min(1e-30)
```

You multiplied the exponent argument by `0.5`, making it even **smaller**. This makes `Theta0_n` even **larger**, which in turn makes `n` even **higher**. You have exacerbated the slope error, not fixed it. The fact that the error may have appeared to shrink from 12% to 5% is likely due to the interaction with the other major bug you fixed (`phi` calculation), which shifted the operating point and changed the relative error.

### The Solution

The `0.5` factor is incorrect. It is based on a flawed analysis of the problem. In most standard BSIM4 versions, the `Theta0` attenuation factor is calculated identically for both the `Vth` DVT effect and the `n` factor's `cdsc` term.

**Remove the `0.5` factor from the `T0_th_n` calculation.**

```python
# CORRECTED code in compute_dc_vth.py

# ...
# A.5.f (2026-05-02): use a SEPARATE Theta0 for the n branch.
# The half-factor was based on a flawed analysis and has been removed.
T0_th_n = dvt1 * Leff / lt1.clamp_min(1e-30)  # REMOVED 0.5 *
Theta0_n = _exp_threshold_branch(T0_th_n)
tmp4 = (tmp2 + tmp3 * Theta0_n + cit) / coxe
# ...
```

### Why This Fixes Everything

1.  **Slope Correction**: Removing the `0.5` factor will double the exponent argument `T0_th_n`. This will make `Theta0_n` significantly smaller, correctly reducing the `cdsc` contribution and lowering `n`. This should bring your subthreshold slope much closer to the ngspice ground truth.

2.  **Offset Correction (The surprising part)**: The sub-VT current is not just a function of `Vth`, but of the entire `Vgsteff` bridge, which depends heavily on `n`. The denominator of `Vgsteff` is `T9v = mstar + n * T3v`. By using an incorrect, artificially high value for `n`, you were distorting the entire `Vgsteff` calculation. Correcting `n` will change the denominator's behavior, which in turn changes `Vgsteff` and the resulting drain current. It is highly probable that fixing the slope factor `n` will also resolve a significant portion of the 5-11x vertical offset you are seeing. The "60-80 mV-equivalent error in voffcbn" is likely an alias of the error in `n`.
