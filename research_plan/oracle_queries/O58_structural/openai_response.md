# openai response (gpt-5) — 135s

Q1. Most likely cause? C > B >> A
- Evidence for C (spurious local optimum in z304):
  - z304’s best VG1=0.6 fit needs Bf=9000 (JSON: 0.704 dec). That Bf is nonphysical vs Sebas’s Bf≈50 and “BJT = complementary firing current,” not a calibrated device. Overfitting with an unphysical gain is classic “fit the wrong physics” behavior.
  - z304’s cell-wide “0.99 dec” isn’t reproduced in the branch-resolved aggregation you posted (best compromise ≈2.40 dec), and its best branches rely on degrees-of-freedom v5b intentionally removed (K1(VG2), ALPHA0 polynomial, active avalanche). v5b constrains those to Sebas’s recipe (K1(VG1)-only, ALPHA0 const, avalanche dropped).
  - When v5b is forced to z304-like extremes (Bf=3000), error gets worse (JSON: 4.25 dec), not better. If z304’s 0.99 were genuine physics, reusing its “magic” Bf should have helped.
- Evidence against A (structural/topology error as primary):
  - R-3 shows the previously inert p-diode wiring was fixed (body_pdiode_Rs now present and consumed). Avalanche and TAT toggles are live (tat_on degrades exactly as expected). 5/5 unit tests pass. No symptom of wrong polarity/topology (e.g., sign catastrophes) — just high residuals.
- Why B (wrong parameter region) is secondary:
  - You haven’t yet swept the new recipe’s key levers (K1_LUT_scale, mbjt_step_threshold, BETA0 scale). Those can plausibly recover a lot. But the magnitude of the z304→v5b jump and the reliance on nonphysical Bf point first to C.

Q2. Js invariance ⇒ body-diode conduction is not the dominant path
- Five decades of body_pdiode_Js produce bitwise-identical residuals. With body_pdiode_Rs = 1e10/1e9/1e8 Ω per branch, the diode branch is effectively isolated for DC. Combined with pdiode bv≈11 V and Sebas’s note that the diode is for capacitance (Cb) rather than firing, the dominant DC current is channel + BSIM impact-ionization feeding the complementary BJT path. The diode’s DC current path is negligible in your operating window.

Q3. Why adding “correct physics” worsened VG1=0.6 (0.43 → 1.18 dec)?
- Reduced overfitting capacity: z304 used K1(VG2) and ALPHA0(VG1,VG2); v5b locks K1 to VG1-only and ALPHA0 constant. That removes nonphysical degrees-of-freedom that previously soaked up mismatch at VG1=0.6.
- Reframing the BJT as a complementary current source with realistic Bf≈50 removes the unphysical gain (Bf≫1000) that had been compensating model deficiencies.
- Dropping avalanche (per Sebas) removes a surrogate knob that z304 likely used to mimic impact-ionization/body dynamics. Until BETA0/K1/mbjt-step are re-tuned under the constrained recipe, error rises.

Q4. Cheapest 2-hour discriminator (structural vs param) — pre-registered
Design: path-liveness ablation at the three “overlay” corners [(0.6,0.35), (0.4,0.25), (0.2,0.0)] on the median cell.
Runs: baseline v5b recipe; then toggle one mechanism at a time:
- A1: iii_to_body_factor = 0 (kill BSIM impact-ionization to body)
- A2: mbjt = 0 (force complementary BJT off)
- A3: body_pdiode_to = "off" (remove diode branch entirely)
- A4: tat_on = True (sanity; should worsen)
Gate (structural-fault FAIL): for a given corner, if both A1 and A2 change median_log_rmse by <0.1 dec vs baseline, the corresponding path is inert in KCL → structural problem.
PASS (parametric): at least one of A1 or A2 increases error by ≥0.5 dec at VG1=0.6; proceed to R-6-lite sweep on K1_LUT_scale × mbjt_step_threshold × BETA0_scale.

This 4×3 grid (≤12 fits) with current logging fits easily in 2 hours and gives a crisp structural vs parameter verdict.
