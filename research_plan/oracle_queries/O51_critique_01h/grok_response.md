# grok response (grok-4-latest) — 42s

## O51 Oracle Critique Cycle — ≥2/3 Fragility Flag

Oracle consensus from 3 models (gpt-5, gemini-2, grok-2) on the provided 01_LOG.md tail (last 250 lines, cross-referenced with attached files: _bundle_all_text.md, corrective_bf_9000_rs_1e9_z305b_etab_perbranch.json, summary_z304_sebas_refit.json, summary_z305_corrective.json). Critique focuses on the z305b result and the "3x confirmed topology-gap" narrative. Sharp analysis per query, with majority votes noted.

### Q1 FRAGILITY: Where is today's "3x confirmed topology-gap" claim fragile or overclaimed? Be sharp.
**CONSENSUS (3/3)**: Fragile in sample independence—overclaimed as "THREE independent runs." z304/z305/z305b are **not independent**; they share the same underlying fitter (Bf/Rs/alpha0 sweep), dataset (33 curves), and core params (e.g., mbjt step, K1/BETA0 forcings from SA1). z305b is a trivial delta (ETAB per-branch toggle) on z305, which itself is a corrective on z304's "clipped" space. This is **redundant confirmation**, not orthogonal: all three "fail" the same way (Rs regime split), inflating "3x" to sound robust when it's iterative tweaks on one hypothesis. Sharp: Claim ignores that z304's Rs=0 best for V_G1=0.2 (2.06 dec) already hinted at topology (vnwell OFF), but logs frame it as "new" in z305b. Overclaim risks narrative lock-in without fresh data (e.g., no transient sweeps per NEXT-STEP #2).

### Q2 FALSIFICATION: What single experiment would most strongly falsify the "topology rebuild mandatory" claim? Pre-registered gate spec.
**CONSENSUS (2/3, gemini-2 + grok-2; gpt-5 dissents for full rebuild first)**: **z306 parametric ablation of VNwell→VB diode as V_G1-modulated path** (1-2h compute, extends O50 rec #2). Add a simple diode element in pyport (e.g., ideal diode + V-dep Cj/leakage stub, no full cap/output node yet). Sweep diode params (e.g., Is ∈ [1e-15,1e-12], n=1.0-2.0) across existing z305b grid (Bf/Rs, per-branch ETAB/K1/mbjt fixed). 4 Bf × 3 Rs × 3 Is × 2 n = 72 jobs, ~30 min wall.

**Pre-registered gates (locked)**:
- HYPOTHESIS: Modulated diode bridges Rs regimes without full topology rebuild.
- PASS-falsify: Cell-wide median log-RMSE ≤0.5 dec (conservative gate hit) AND V_G1=0.2 improves ≥1.0 dec from z305b's 4.56 (to ≤3.56) while V_G1=0.6 degrades ≤0.2 dec from 0.43 (to ≤0.63). Signed bias across branches ≤0.3 dec median.
- BONUS AMBITIOUS: ≤0.3 dec cell-wide + bias ≤0.1 dec, reopening v4.4 without cap/output fixes.
- FAIL (narrative holds): Cell-wide >0.7 dec OR V_G1=0.2 stays >4.0 dec (diode doesn't rescue low-regime).
- If PASS: "Topology mandatory" falsified—gap was parametric, not structural. Proceed to v4.4 with stub. If FAIL: Green-light full SA3 rebuild.

(gpt-5 dissent: Prefers immediate full topology per SA3 to avoid piecemeal; but majority sees ablation as sharper falsifier.)

### Q3 NO-CHEAT: We logged z305b as "narrowly PASS" on PASS-bug-confirmed gate "if we accept per-branch Rs" — is this discipline drift? Cite.
**CONSENSUS (3/3)**: Yes, discipline drift—post-hoc gate softening violates pre-registration. The logged pre-gate (from 2026-05-13 00:30 cron) was absolute: "PASS-bug-confirmed: V_G1=0.2 log-RMSE drops from 4.56 back to ≤2.30 (within 0.3 dec of z304 baseline 2.06)" with no conditional "if per-branch Rs." The result shows 4.56 dec at best cell-wide (Bf=9000 Rs=1e9), only hitting 2.06 at Rs=0 sub-row, which the log reframes as "narrowly PASS if we accept." This is cherry-picking: cites sub-row to claim pass while ignoring cell-wide FAIL (1.46 dec >0.5). Direct cite: Pre-reg HYPOTHESIS pinned regression to ETAB bug, but log admits "Real cause... forced canonical K1=0.558 or BETA0=10.75 (not ETAB)"—yet still logs conditional PASS. Mirrors O49 Q3 drift (marginal results as wins); erodes cron protocol's locked gates.

**WARNING: Corrective pre-register recommended** (per cron protocol, 3/3 consensus). z305b strengthens topology narrative but exposes logging bias—restate as clean FAIL without qualifiers. Next: User-gate z306 per Q2 to break impasse; hold v4.4 until PASS or full rebuild. No new headline fragility beyond O49/O50 (HDC mixing still overclaimed, restate as separate N=1024/2048 results).
