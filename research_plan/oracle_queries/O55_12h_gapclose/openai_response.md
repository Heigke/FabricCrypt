# openai response (gpt-5) — 77s

Q1 — GATE CROSSING (which is shippable; any overclaim?)
- Most defensible headline: HDC N=16384 at ~84% UCI-HAR, 35 nJ/inf. Ship only after n=10 seeds land; until then phrase as “headline candidate; 10-seed lock pending.” Also restate noise as “no worse than σ=0” (drop “noise-benefiting” until CI excludes zero).
- T2 Vd>2 V dataset: fully shippable as a new artifact (“143 digitized points ≥2 V from slides 15+21, with per-curve 25–35% uncertainty”). It’s an enabling result, not a performance headline.
- M2 traps 6.2-dec lift: not shippable as a headline. It’s a qualitative mechanism validation; the stub overshoots measured hysteresis by ~14× at 0.17 V/s. Reframe as “mechanism confirmed qualitatively; magnitude not yet matched; not in production model.”
- Overclaims to correct: locked language at n=4 and “noise-benefiting” claim; causal language on diode before data.

Q2 — CHERRY-PICK?
- Yes, reporting “z304 baseline 0.99 dec” as a single number without CI is weak and risks perceived cherry-pick, even if unintentional.
- Fix in ≤1 h:
  1) Determinism check: rerun z304 exactly once; if byte-identical residuals, state “deterministic; CI via bootstrap across IV rows.”
  2) Row bootstrap: 1000 resamples of the 33 IV rows; report median and 95% CI of median_log_RMSE.
  3) Sensitivity: one rerun with tighter/looser solver tolerances to show robustness.
- Then compare z313 vs z304 using CIs or paired per-row deltas. Until then, cite “z304 ≈1.0 dec (n=1 point estimate), CI pending.”

Q3 — Highest-value 1–3 h experiment
Pick C: Snapback peak law sweep and scorer.
- Why: Independent of the currently broken cfg plumbing; gives a crisp, falsifiable physics score you can keep using as the code changes.
- What to do (1–2 h):
  1) Using current pyport (z304), sweep V_G2 points that match slide 15 (e.g., 0.05–0.45) at V_G1=0.3 and the same ramp proxy you’ve been using.
  2) Extract V_peak per curve (local maximum pre-knee), fit linear V_peak vs V_G2; report slope and RMSE vs law 2.73 − 0.625·V_G2.
  3) Add rate probe using your existing 10/100/1000 µs ramps; verify monotone V_peak increase with slew (~+0.15 V/dec threshold).
  4) Log both as gates in the harness so future model edits auto-score them.
- While seeds run (D, opportunistic): queue the KWS linear baseline (P4 falsifier) and finish the z312 10-seed lock; both are cheap and unblock comms.
