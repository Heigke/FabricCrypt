# openai response (gpt-5) — 109s

Q1 — gate-crossing risk (is z304’s 0.99 dec “real”?)
Verdict: Very likely spurious. The 0.99 median was achieved with a reverse‑biased BJT (Vbe<0) and a hard pre‑knee choke (eta_sigmoid), which together forced the model to fit by misallocating current among MOS leakage/diodes rather than engaging the NPN feedback. Three independent lines of evidence:

- R-15 showed 6 decades low Ic and wrong Ib sign under the pre‑fix wiring; yet z304 scored 0.99. That can only happen if the fit leaned on non‑BJT paths to mimic snapback, i.e., compensation not physics.
- After fixing the BJT (R-21/22), a broad 6D refit plateaued at 3.42–3.55 dec over 76 evals. If z304’s basin were physically consistent, small retunes around the corrected topology would recover <1. They didn’t.
- R-14/R-18 found the pre‑knee dId/dV sign wrong and a hardcoded Iii choke (eta_sigmoid) removing 1–2 decades pre‑knee—precisely the regime where the “good” z304 match would need correct physics.

Falsification tests (run any 1 to overturn the verdict):
- Handover test: For each of the 33 biases, take ngspice OP (Vsint*, Vb*) and evaluate pyport(z304 params) currents with corrected BJT. If median dec <1.2, the 0.99 wasn’t spurious; if ≥2, it was.
- Current-partition probe at flagship bias: Compare z304’s Ic_Q1, Ids_M1, Iwell/diodes vs ngspice. If Ic_Q1 is ≥4–6 decades low or wrong sign while total I matches, the match is compensatory.
- Minimal-perturb refit: Lock all z304 structural flags; change only Vbe/Vbc to the corrected forms and refit a0/Bf/Va/Is. If still ≥2 dec, the original 0.99 depended on the bug.

Operational implication: Stop chasing “match z304”; the target is a new <1 dec basin with the corrected BJT and physically enabled feedback.

Q2 — R-22 plateau: which structural flag is missing/mis-set?
Primary culprit: eta_sigmoid. Evidence from R-18 shows it hard‑gates Iii with sigmoid(10*(Vd−1)), removing 1–2 decades pre‑knee and flipping the pre‑knee slope sign. With the BJT now correctly wired (stronger positive feedback), this choke prevents the base from charging in the right Vd range and pins the floor near 3.4 dec regardless of scalar BJT gains. Action: set eta_sigmoid=False (or move its knee down by ≥0.3–0.5 V and/or reduce gain from 10→2–3).

Secondary structural suspects (ranked by expected impact):
- use_lateral_collector=True: Without it, lat_BV is neutered or misapplied; snapback knee and drop amplitude won’t shape correctly. Re‑enable and keep lat_BV in the fit set.
- use_well_diode=True with body_pdiode_to=vb (not gnd): R-10 showed gnd ties can “fake” Vb but cause overshoot; with the BJT fixed, restore the diode to the base node so charge recirculates correctly.
- vnwell_Rs finite (not 0/∞): Needed to get the correct Vb dynamics and hysteresis width; sweep order(s) of magnitude if currently clamped.
- use_local_base=True: If supported, it can stabilize the snapback plateau by decoupling the fast local base from the global well node; smaller effect than the above.

Minimal discriminating AB plan (no re‑fit): starting from z338 best params, flip these flags in order and re‑score all 33: (1) eta_sigmoid off, (2) + use_lateral_collector on, (3) + use_well_diode on & body_pdiode_to=vb. Any ≥1.5 dec gain indicates the plateau was structural, not parametric.

Q3 — single highest‑value experiment (next 1h) + ranking
Pick: ngspice handover (OP handover: ngspice solves Vsint*, Vb*; pyport computes currents). This cleanly separates solver/basin issues from physics. Do it in two lines:
- Line A: current z338 structure (as in R-22 best) with eta_sigmoid on.
- Line B: identical, but eta_sigmoid off.

Interpretation:
- If Line A ≥2.5 dec and Line B still ≥2.0 dec, even with perfect (Vsint,Vb) states, the architecture/flags are missing key physics (enable lateral collector, well diode routing, etc.). You cannot cross sub‑1 without structural change.
- If Line B drops near or below 1.2 dec, the choke was the blocker; proceed to a quick 2D retune (a0,Bf) with eta_sigmoid off to try crossing <1.
- If both are <1, the residual/solver was the dominant issue; prioritize 2D Newton with Vb free and warm‑starts from ngspice OP.

Why this is best: It yields a yes/no on “can the current equations reproduce the silicon with the right internal state?” within 1 hour, and it simultaneously tests the top structural suspect (eta_sigmoid) without committing to a full DOE.

Ranking of your listed candidates (by info/time ROI):
1) ngspice handover (chosen).
2) cfg-diff (R-23) with targeted AB toggles (eta_sigmoid off, +use_lateral_collector).
3) forward-2T warm‑start from ngspice basin.
4) Vsint‑residual sensitivity audit.
5) Full 2^7 DOE sweep (too slow, non‑diagnostic for 1h).
