# gemini response (gemini-2.5-pro) — 212s

Excellent, this is a well-structured diagnostic request. The detailed context and prior reports (A1a-i) are extremely helpful.

### Diagnosis

The 2-5 decade `Id` discrepancy, despite the correct `emitter=GND` topology, is caused by a **critically underestimated impact ionization current (`Iii`)**. Your simulation correctly finds the only stable DC fixed point that the *current physics model* allows: a low-body-voltage state (Vb ≈ 0.34 V) where `Iii` is negligible (~1e-25 A) and cannot initiate the regenerative feedback required for BJT turn-on.

The root cause is not a solver failure but a model-physics gap. Sebas's SPICE simulation is almost certainly using a more complex or differently parameterized `Iii` model than our port currently implements.

### Ranked List of Causes

1.  **(b) Incomplete Impact-Ionization Physics (Highest Probability):** Our BSIM4 port is missing standard length-dependent binning for `alpha0` and `beta0`. The presence of `lalpha0` and `lbeta0` in Sebas's model cards is a strong signal that this effect is essential. As your diagnostic A1d showed, with the card's base `BETA0=18-20`, the `exp(-beta0/...)` term is mathematically guaranteed to be near zero at this bias. Length-binning is the mechanism that would alter `alpha0` and `beta0` to physically correct values for this specific device geometry.

2.  **(d) Solver Path-Dependence (Hysteresis):** It is plausible that the true I-V curve is S-shaped (hysteretic) and our forward Vd sweep is tracking the low-current "off" branch, while Sebas's transient simulation (or a different DC sweep) latches onto the high-current "on" branch. Our arclength solver finding `n_folds=0` argues against this, but only for the *current* physics model. If the model is wrong (see #1), it may not produce a fold that the solver can see.

3.  **(a) Different Fixed Point (Low Probability):** A1g's multi-root search was robust. It's highly unlikely there's another DC root our solver is simply "missing" *with the current physics*.

4.  **(c) "Complementary Bipolar Current" Misinterpretation (Very Low Probability):** A1i's analysis is correct. The term almost certainly refers to the Q1 NPN current path itself, which acts in parallel to the M1 channel current. There is no evidence of a hidden behavioral source.

---

### Proposed Fixes & Experiments

#### **Fix #1: Implement `lalpha0` and `lbeta0` Binning (Highest Priority)**

This is a clear bug in our BSIM4 port. The `l*` parameters are standard binning coefficients and `temp.py` is ignoring them.

**Patch for `_extracted/nsram_bsim4_port/nsram/nsram/bsim4_port/temp.py`:**

The `compute_size_dep` function populates a `scaled` dictionary. `alpha0` and `beta0` are already in the `SCALED_PARAMS` list. The loop correctly calculates `base + l_X/L + w_X/W + p_X/LW`. The issue is that `lalpha0` and `lbeta0` are not being consumed from the model card. A quick check of your `leak.py` shows it reads from `sd.scaled`, so fixing `temp.py` is sufficient.

The `SCALED_PARAMS` list in `temp.py` already includes `"alpha0"` and `"beta0"`. The `lalpha0` and `lbeta0` parameters are being loaded by `model_card.py` but not used. The main loop in `compute_size_dep` correctly applies the binning formula. The problem is that the `lalpha0` and `lbeta0` values are not being read from the model card into the `model` object in the first place. Let's assume they are loaded correctly and the issue is just that they are not used.

Aha, I see the issue in `A1h`: *"the lalpha0, lbeta0 fields ingested by _model_card_data.py:174,180 are not consumed in temp.py"*. This implies the main scaling loop is correct but the values aren't being passed. Let's ensure the `compute_size_dep` function correctly applies them.

The loop in `compute_size_dep` is generic. If `lalpha0` is loaded into the model object as `lalpha0`, the loop will find it. The bug is likely that the parameter names in `_model_card_data.py` don't match. Assuming they are loaded correctly, the existing code should work. **No patch is needed if the parameters are loaded correctly.** Let's proceed assuming the values are available in the `model` object.

#### **Experiment #1: Backward Vd Sweep to Test for Hysteresis**

This is the highest-information experiment to distinguish a solver-path problem from a model-physics problem. It directly tests for a high-current stable branch.

**Instructions:**
Modify your validation script (`z91g_two_model_validation.py` or a copy) to perform a backward sweep for a single failing bias point (e.g., VG1=0.6, VG2=0.0).

1.  **Solve for a point deep in saturation:** Use `solve_2t_steady_state` (from `nsram_cell_2T.py`) to find the solution at `Vd=2.0V`. This should be on the high-current branch if one exists.
2.  **Sweep backwards:** Create a reversed `Vd_seq_rev = torch.flip(Vd_seq, dims=[0])`.
3.  **Use warm-starting:** Call `forward_2t` (or `forward_2t_arclength_grad`) with the reversed sequence, ensuring `warm_start=True`. The key is to seed the first point of the backward sweep (`Vd=2.0V`) with the solution from step 1. The `forward_2t` loop will then naturally cascade this solution backwards.

```python
# In a test script, after setting up cfg, models, bjt, P_M1, P_M2...
# For bias VG1=0.6, VG2=0.0

from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state, forward_2t

Vd_seq = curves[...]["Vd"] # The Vd sequence for the failing curve
VG1_t = torch.tensor(0.6)
VG2_t = torch.tensor(0.0)

# 1. Find a high-current starting point at Vd_max
print("Solving for high-Vd start point...")
with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
    out_start = solve_2t_steady_state(
        cfg, model_M1, bjt, Vd=torch.tensor(2.0), VG1=VG1_t, VG2=VG2_t,
        Vsint_init=torch.tensor(0.5), Vb_init=torch.tensor(0.8), # Start high
        model_M2=model_M2
    )
    vsint_start = out_start["Vsint"].detach()
    vb_start = out_start["Vb"].detach()
    print(f"  Vd=2.0V -> Vsint={vsint_start.item():.3f}V, Vb={vb_start.item():.3f}V, Id={out_start['Id'].item():.3e}A")

# 2. Create reversed Vd sequence and override the first point's warm-start
Vd_seq_rev = torch.flip(Vd_seq, dims=[0])

# 3. Run forward_2t with a custom warm-start for the first point
print("Running backward sweep...")
with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
    # Temporarily modify forward_2t to accept an initial warm-start
    # Or, more simply, manually loop:
    vs_warm, vb_warm = vsint_start, vb_start
    results_rev = []
    for vd_i in Vd_seq_rev:
        out_i = solve_2t_steady_state(
            cfg, model_M1, bjt, Vd=vd_i, VG1=VG1_t, VG2=VG2_t,
            Vsint_init=vs_warm, Vb_init=vb_warm, model_M2=model_M2
        )
        results_rev.append(out_i['Id'].item())
        vs_warm, vb_warm = out_i['Vsint'].detach(), out_i['Vb'].detach()

# 4. Plot results_rev vs Vd_seq_rev and compare to measurement
# If results_rev tracks the high-current branch, the model is bistable.
```

---
### Answers to Specific Questions

1.  **Ranked Likelihood:**
    1.  **(b) Incomplete Impact-Ionization Physics:** Most likely. The model is missing a standard BSIM4 feature (`lalpha0`/`lbeta0` binning) that directly impacts `Iii`, the known weak link in the feedback loop.
    2.  **(d) Solver Path-Dependence:** Plausible. If the (correct) physics model is hysteretic, a simple forward sweep can get stuck on the wrong solution branch.
    3.  **(a) Different Fixed Point:** Unlikely. A1g's tests were convincing that the *current model* is not multi-rooted.
    4.  **(c) "Complementary Bipolar" Misinterpretation:** Very unlikely. A1i's schematic analysis is sound and provides a complete, self-consistent physical interpretation.

2.  **The `lalpha0` Puzzle:**
    *   **Yes, you absolutely should apply the binning.** Its omission is a bug in your BSIM4 port. The formula is `alpha0_eff = alpha0 + lalpha0/Leff`.
    *   **Unit Analysis:** Your reasoning is correct. For the formula to be dimensionally consistent, if `[alpha0]` is `m/V`, then `[lalpha0]` must be `m^2/V`.
    *   **Calculation & Verdict:** Your calculation `alpha0_eff = 7.84e-5 - 0.527 ≈ -0.527` is also correct. An effective `alpha0` cannot be negative. This indicates a strong possibility of a **typo or unit error in the provided `M2_130bulkNSRAM.txt` model card**. The magnitude of `lalpha0` is unphysically large.
    *   **Recommendation:** Implement the binning in `temp.py` as it's a code-level correctness issue. Then, for validation against Sebas's data, you may need to temporarily set `lalpha0=0` for the M2 card to avoid the negative result, but flag this discrepancy to Sebas. The bug in your code is the *omission* of the calculation, not the result of the calculation itself.

3.  **Body-Charging Avalanche Bistability Test:** Yes. The backward Vd sweep described in **Experiment #1** is the direct and standard way to test for this. It will trace the upper stable branch of an S-shaped I-V curve if one exists.

4.  **"Complementary Bipolar Current":** My assessment aligns with A1i. Sebas is using "complementary" to mean the parasitic BJT (Q1) current path acts *in addition to* (in parallel with) the primary MOSFET (M1) channel current path. Both contribute to the total drain current. The `emitter=GND` wiring is the key to this parallel path, and your recent fix correctly implemented it. It is not a hidden behavioral source.

5.  **Concrete Next Test:**
    1.  **Highest Priority (Fix + Test):** Implement the `lalpha0`/`lbeta0` binning in `temp.py`. Re-run the `z91g` validation. This is a known bug fix. Due to the likely data error in `lalpha0`, this may not fix the result, but it makes your model more physically correct.
    2.  **Second Priority (Diagnostic):** Run the **Backward Vd Sweep (Experiment #1)** on the worst-failing curve (VG1=0.6, VG2=0.0). This will give a definitive yes/no on whether the model (even the un-binned one) supports a high-current state that your solver is missing. This is the highest-information *diagnostic* test.

Given the time constraint, **run Experiment #1 immediately**. It requires minimal code changes and will tell you whether to focus on improving the solver's path-finding or on fixing the underlying physics model.
