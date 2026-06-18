# grok response (grok-4-latest) — 59s

### Q1
(A) Model is STRUCTURALLY wrong.

Evidence:
- Js sweep (1e-6 to 2.44e4 A/m²) yields bitwise identical fits (z321_progress.json, all configs at each Bf level show same medians, e.g., 3.011 dec for Bf=50 across all Js). This proves diode path inert/not wired to contribute to residuals (R3_pyport_audit.md confirms missing body_pdiode_Rs → no series resistance limits diode current; path bypassed in v5).
- v5b adds "physical" elements (e.g., Sebas's ALPHA0=7.842e-5 const, K1 LUT, mbjt step, Adiode=22μm², drop avalanche per R1b_zoom_DEEP.md) but regresses V_G1=0.6 from 0.43 dec (z304) to 1.18 dec, and cell-wide from 0.99 to 3.01 dec. z304's Bf=3000 is non-physical ("found" artifact, not Sebas's published Bf=50), masking topology flaws like orphan flags (e.g., TAT vtss/xtss never consumed per R3) and missing body cap in DC residuals.
- Parametric mismatch (B) unlikely: Sebas's params are direct from fits-xlsx (23/33 biases, R1b), yet worsen fit vs. z304's tuned non-Sebas values. Spurious optimum (C) possible for z304 but contradicted by v5b's consistent degradation across branches, pointing to wrong node topology (e.g., avalanche kept despite Sebas drop, per mail.txt 2026-04-17).

(148 words)

### Q2
Diode current path is negligible/sub-threshold; channel impact ionization (BSIM4 ALPHA0/BETA0) or parasitic BJT dominates.

Evidence: Zero effect from Js sweep implies diode forward/reverse current (I_body_pdiode in _residuals) not flowing or clamped (e.g., Vb << Vt, or infinite Rs equivalent due to missing body_pdiode_Rs per R3_pyport_audit.md). In operating regime (Vd=0-2V, V_G1=0.2-0.6V), body voltage floats low, keeping diode off; snapback/firing relies on III or NPN amplification (Bf-scaled in z321, where higher Bf worsens medians: 3.01 at Bf=50 → 4.25 at Bf=3000). Confirms per R1b: NPN as "complementary firing source" artifact, not real device—diode only for cap/τ in transients (image-2.png ramp rates).

(128 words)

### Q3
Local optimum trap: z304's "good" fit (0.43 dec at V_G1=0.6) relied on non-physical params (Bf=3000 >> Sebas's 50) compensating for missing/inert elements; adding physics (e.g., realistic Adiode=22μm², constant ALPHA0, mbjt step) perturbs the landscape, escaping the spurious minimum but landing in a worse one without re-optimization.

Mechanism: z304 overfits by inflating Bf to boost NPN gain, masking diode inactivity (Js inert per Q2). v5b integrates "correct" elements (R1b recipe: drop avalanche, add LDE stress, true Cb=7fF) but inherits z304's narrow search space—regression exposes topology flaws (e.g., orphan TAT/Cj0 params per R3). Re-run BBO needed to find new global min; regression = evidence of prior wrong-physics compensation.

(137 words)

### Q4
Sweep Bf (50-3000 log-space, 5 pts) × Js (1e-6 to 2.44e4, 5 pts) on daedalus CPU using v5b full recipe (ALPHA0 const, K1 LUT, mbjt step, Adiode=22μm², drop avalanche, Cb=7fF, add body_pdiode_Rs). Pre-reg: if any combo <1.0 dec cell-wide → parametric (B); if all ≥3.0 dec like v5b baseline → structural (A). 25 configs × ~5min/cell × 25 cells = ~2h. Distinguishes: parametric would recover z304-like fit via search; structural keeps diode inert (zero Js sensitivity).

(102 words)
