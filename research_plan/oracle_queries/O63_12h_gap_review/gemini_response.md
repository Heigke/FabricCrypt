# gemini response (gemini-2.5-pro) — 40s

### Oracle 1

**Q1 — gate-crossing risk**: z304's 0.99 dec is a spurious local optimum and a dangerous baseline. The evidence is dispositive: R-15 and R-18 found a 6 order-of-magnitude discrepancy in the BJT collector current (`Ic_Q1`) between the pyport model and ngspice ground truth. The BJT was functionally dead in pyport. The 0.99 dec fit was achieved by the optimizer compensating for this fundamental physics error with other parameters, creating a mathematically convenient but physically meaningless solution.

The R-21 log entry confirms this: "...alpha0/Bf/VAF/lat_BV were tuned for the BROKEN BJT." Fixing the BJT Vbe wiring (R-20) correctly revealed the model's true state, causing the error to jump to 4.16 dec. This is not a regression; it is the removal of a bug that was being masked by another set of errors (the parameters).

The goal must immediately change from "match z304" to "find the first sub-1.0 dec basin with the corrected BJT topology." Chasing the 0.99 dec value is chasing a ghost created by error cancellation. The project is now at a new, physically sound starting point, and the 3.4-4.2 dec error is the true baseline to improve upon.

**Q2 — cherry-pick risk in R-22 floor**: The 3.43 dec floor is not a parameter-tuning problem; it is the signature of a structural model deficiency. The BBO's failure across 76 evaluations, with the top 5 results clustered in a tight 3.42-3.55 range, proves the model's current mathematical form is incapable of fitting the data, regardless of parameter values. The landscape is a high-altitude plateau because a required physical mechanism is either missing or incorrectly implemented.

The prime suspect for this structural blocker is the `eta_sigmoid` term. The R-18 log identifies it as a "hardcoded choke" that "kills 1-2 dec pre-knee" and was responsible for the pre-knee slope having the wrong sign. This term was likely an empirical patch to suppress leakage or otherwise tame the model when the BJT was broken. Now, with the BJT correctly implemented and firing hard (the "activated-overshoot" of R-21), this artificial choke is likely preventing the model from correctly representing the device's turn-on characteristic. It is a remnant of the old, broken physics regime and is now actively harmful. Disabling it is the most logical next step to break through the 3.43 dec floor.

**Q3 — highest-value next experiment**: The single most informative experiment is a **targeted disabling of the `eta_sigmoid` term, followed immediately by a short (e.g., 20-eval) BBO run on the same 6 parameters from R-22.**

This is superior to all other options for the following reasons:
1.  **It is a direct test of the primary hypothesis from Q2.** The BJT fix created an "activated-overshoot." The log identifies `eta_sigmoid` as an artificial "choke." This experiment directly tests if removing the choke resolves the overshoot.
2.  **It is fast and decisive.** Unlike a full `cfg-diff` or a 2^7 DOE sweep, this is a single toggle. The result will be available in minutes.
3.  **It provides a clear, falsifiable outcome.**
    *   **If the BBO floor drops significantly below 3.4 dec:** The hypothesis is proven correct. `eta_sigmoid` was the second structural blocker. The path to sub-1.0 dec is now open via parameter tuning.
    *   **If the BBO floor remains at ~3.4 dec:** The hypothesis is falsified. The blocker is another structural element. This negative result is also valuable, as it immediately promotes the systematic `cfg-diff` audit (R-23) as the next priority.

This experiment avoids wasting time on solver-centric tests (like ngspice warm-starts), which are irrelevant when the BBO has already proven the model's underlying equations are the problem. It is a surgical strike on the most likely remaining structural bug.
