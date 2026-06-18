# grok response (grok-4-latest) — 53s

Q1 — FRAGILITY: The HDC noise robustness (80.4% at σ=0.05 via N=2048) is overclaimed as "noise-immune"; methodological gap is no direct comparison to unscaled N=1024 baseline (which drops to 77.5% at same σ per z302), hiding assumption that 2x energy cost (4.5 nJ/inf vs 2.3 nJ) is irrelevant for practical deployment.

Q2 — SINGLE-EXPERIMENT FALSIFICATION: z304 per-branch refit (results in z304_sebas_refit_summary.json); rerun exactly as original (sweep per-V_G1 Bf=[50,500,9000], alpha0=[1e-5,1e-4,1e-3,0.01], Rs=[0,1e9,1e10] on Sebas 33 IV curves, aggregate med/signed log-RMSE); if overturned to find compatible cell-wide Bf/Rs (e.g., single optima with median <1.0 dec across branches), it invalidates SA3 topology gaps, SA4 incompatibility, 4E hold decision, and v4.4 path blockers.

Q3 — NO-CHEAT DISCIPLINE: Yes, drifted from pre-registered SA4 gate (median log-RMSE <0.5 dec conservative); reported "borderline PASS" at 0.70 dec for V_G1=0.6. Cite: "## 2026-05-12 18:42 — SA4 z304 final per-branch refit" table and verdict.
