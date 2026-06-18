# Oracle critique cycle — NS-RAM project, 2026-05-12 evening

Context: 01_LOG.md tail (250 lines) shows today's activity:
- DA2 finding: Mario-zenodo SPICE deck is OUTDATED (user confirmed),
  generic PTM 130nm thin-ox, NOT our Sebas-measured 130nm thick-ox imec
- SA1: Sebas canonical 58-param catalogued from JSON+cards (no conflicts)
- SA3: 3 missing pyport topology elements identified (VNwell→VB diode,
  VB-VG2 cap, VB as output node)
- SA4: per-V_G1-branch refit shows DC log-RMSE V_G1=0.6 → 0.70 dec
  (borderline PASS), V_G1=0.2 → 2.06 dec (FAIL); branches want
  INCOMPATIBLE Bf values
- HDC headline: 80.23% n=20 UCI-HAR, σ-robustness via N=2048 → 80.4%
- Bayesian RNG: ESS 1.03× pseudo + NIST 5/5
- 4D verdict: GATE not SHIP; KWS at chance ship-blocker
- 4E decision: HOLD compile — findings insufficient for v4.4 brief

Please answer all 3 questions SHARPLY and SPECIFICALLY:

Q1 — FRAGILITY: Where is today's strongest result most fragile or
overclaimed? Be concrete: cite a metric, a methodological gap, a
hidden assumption. Don't be polite.

Q2 — SINGLE-EXPERIMENT FALSIFICATION: Among everything we did today,
what is the ONE experiment whose result, if rerun and overturned,
would invalidate the most other claims? Define the rerun exactly.

Q3 — NO-CHEAT DISCIPLINE: Did we drift from pre-register / honest
gate / full-heatmap discipline in any logged action? Cite the line.
If we didn't, say so.
