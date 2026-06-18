Status: NS-RAM pyport rebuilt with proper flag wiring (5/5 unit tests PASS,
infrastructure verified working). Applied complete recipe from Sebas's mail
+ Zoom material (Cb=7fF, Adiode=22μm², ALPHA0 const, K1 LUT, drop avalanche).

PROBLEM: v5b with Sebas's own published Bf=50 gives cell-wide 3.01 dec.
Baseline z304 with Bf=3000 (a "found" value, not Sebas's) gave 0.99 dec.

Worst: V_G1=0.6 branch went from 0.43 dec (z304) → 1.18 dec (v5b) — adding
"physical" elements REGRESSED the previously-best branch.

Js sweep: 5 values from 1e-6 to 2.44e4 give BITWISE IDENTICAL fits → diode
current path is NOT actively conducting in our v5 _residuals.

Q1: Given that v5b with Sebas's params is 3× worse than z304 with Bf=3000,
is the most likely cause:
  (A) Model is STRUCTURALLY wrong (e.g., wrong node topology, wrong polarity,
      missing device, wrong sign convention)
  (B) Model is right but we're in wrong parameter region; need BBO search
      across wider hyperparameter space
  (C) z304's 0.99 was a "spurious local optimum" matching wrong physics

Cite specific evidence.

Q2: Js variation gives ZERO effect on cell-wide median. What does this tell
us about which current path is dominant?

Q3: V_G1=0.6 regression (0.43 → 1.18 dec) when adding more "physical"
elements: what mechanism could cause adding correct physics to make a
previously-good fit WORSE?

Q4: Cheapest 2-hour experiment that distinguishes structural-vs-parametric
problem? Pre-registered gate.

Be sharp. ≤500 words per oracle.
