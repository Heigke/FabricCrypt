# z465 honest analysis

## What we did
4D Bayesian optimisation over (snap_Is, R_body, β, C_body) against
7 Mario targets extracted from slide 08 (O47 deck) of his Lecce talk.
Fitness = weighted sum of relative errors + DC RMSE penalty (kicks in past 2 dec).
70 BBO iterations, gp_minimize (skopt), seed = SNAP_HOT canonical.

## Caveats

- Mario slide 08 spike train is a **simulation overlay**, not measured silicon. 
  Calibrating to it = agreement with Sebas's published SPICE, NOT real device.
- Rise time (26 ns target) is near pixel resolution of the screenshot (±6 ns).
- DC RMSE is evaluated on a 6-curve subset during BBO and 12-curve subset at best (
  full V1 cell-wide is 25 biases; here we trade some fidelity for BBO speed).
- BBO objective is single-process per call (scipy BDF). Parallel evaluations would
  require multiple processes; not done here.

## Per-target findings

- No catastrophic per-target failure

## Gate verdict: **INFRA_ONLY**
- n_within_30 (hi-res) = 2/7
- DC RMSE (full subset) = 1.373 dec
- best fit_total = 0.5572
