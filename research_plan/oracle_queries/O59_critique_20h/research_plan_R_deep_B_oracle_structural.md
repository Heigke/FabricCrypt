# R_deep_B — Oracle Synthesis: Structural vs Parametric? (O58_structural)

**Date**: 2026-05-13
**Packet**: `research_plan/oracle_queries/O58_structural/`
**Providers**: openai (gpt-5, 135s), gemini (2.5-pro, 68s), grok (4-latest, 59s)
**Wall**: ~4.4 min

---

## Per-question consensus / dissent

### Q1 — Structural vs parametric vs spurious?

| Oracle | Primary | Secondary |
|---|---|---|
| **gpt-5** | **C (spurious)** | B > A |
| **gemini** | **A (structural)** | C strong secondary |
| **grok** | **A (structural)** | C possible, B unlikely |

**Consensus**: it is **NOT pure (B) parametric**. All three reject "just sweep harder." 2/3 vote structural (A); gpt-5 inverts and says z304 was a spurious local optimum on wrong physics — but it agrees the v5b body-diode path is dead in KCL (which is itself a structural fact). So **the unanimous operational call is: there is at least one dead current path in v5b**, regardless of whether you call that "structural" or "parameter region with a dead branch".

**Dissent**: gpt-5 weights C > A because z304's Bf=9000 best-branch is nonphysical and v5b passes unit tests (no sign catastrophes). Gemini+grok weight A higher because the audit (`R3_pyport_audit.md`) explicitly flagged missing `body_pdiode_Rs` and the Js-invariance is bit-exact.

**Combined verdict**: Both are simultaneously true. z304 was a spurious optimum (fit-by-overfitting via nonphysical Bf and avalanche crutch) AND v5b has at least one structurally inert path (body diode). Removing the z304 crutches before fixing the v5b dead branch produced the regression.

### Q2 — Js invariance → which path dominates?

**Unanimous**: body p-n diode DC path is **inactive / negligible**. The dominant current paths are (in agreement):
1. Channel `Ids` (BSIM4)
2. Impact ionization `I_iii` (BSIM4 ALPHA0/BETA0 → body)
3. Parasitic BJT `I_bjt` (complementary firing source)

Diode role per Sebas: capacitive (Cb, transient time-constant), **not DC firing**. Adiode=22μm² and Cb=7fF make sense only in transient context.

Mechanism for inactivity (gemini + grok converge): missing `body_pdiode_Rs` series resistance means the diode branch is either clamped by the network or never reaches forward conduction in (Vd ∈ [0,2], V_G1 ∈ [0.2,0.6]) — body voltage floats low, diode stays off.

### Q3 — Why did "adding correct physics" regress V_G1=0.6?

**Unanimous mechanism**: *removing compensating errors*.

- z304 had three "crutches" giving it surplus DOF: (i) K1(VG2) instead of K1(VG1), (ii) ALPHA0 polynomial in (VG1,VG2), (iii) active avalanche/Chynoweth path, (iv) nonphysical Bf ≫ 50.
- v5b correctly removes (i)(ii)(iii) per Sebas's recipe and reframes BJT with Bf≈50.
- But v5b did NOT yet rewire the body voltage correctly (diode path dead). So the model lost its crutches before its real replacement mechanism became active → regression, especially at V_G1=0.6 where the avalanche crutch had been doing the most work.

### Q4 — Cheapest 2h discriminating experiment?

All three propose **path-liveness ablation**, differing in implementation:

| Oracle | Design |
|---|---|
| gpt-5 | Toggle 3 mechanisms (`iii_to_body_factor=0`, `mbjt=0`, `body_pdiode_to="off"`) at 3 bias corners. Gate: if A1 and A2 both move <0.1 dec → structural fault. |
| gemini | Single-cell `use_well_diode=True` with vnwell_Rs=1e8Ω vs current control. Gate: >1% rmse change → structural confirmed. |
| grok | 5×5 Bf×Js sweep on full v5b + add body_pdiode_Rs. Gate: any combo <1.0 dec → parametric; all ≥3.0 → structural. |

---

## Verdict

**Structural with parametric-amplification**. The v5b model has at least one dead KCL branch (body p-n diode, Js-invariant by bit-exact test). The previous z304 "success" was a spurious local optimum riding on now-removed crutches (overfit Bf, K1(VG2), avalanche). You cannot resolve this with BBO until the body-voltage path is electrically live.

**Order of operations**:
1. Make the body branch live (add `body_pdiode_Rs` OR re-enable the existing `vnwell_Rs` path).
2. Verify Js sweep now produces non-identical residuals (positive control on structural fix).
3. Then BBO over (Bf, K1_LUT_scale, mbjt_step_threshold, BETA0_scale).

## Recommended cheapest 2h experiment (synthesized)

**Two-stage liveness ablation, ≤2h on daedalus**:

**Stage 1 (≤30 min) — Liveness positive control**
- Single cell V_G1=0.6, V_G2=0.0, recipe = v5b.
- Variant A (control): current v5b.
- Variant B: enable `use_well_diode=True` with `vnwell_Rs=1e8 Ω` (gemini's path — uses already-wired infrastructure).
- Variant C: kill BSIM impact-ionization (`iii_to_body_factor=0`).
- Variant D: kill BJT (`mbjt=0`).
- **PASS structural confirmed if**: B differs from A by ≥1% RMSE AND (C or D) shifts RMSE by ≥0.5 dec.
- **FAIL (parametric only) if**: A=B bit-exact and C,D both move <0.1 dec → no path is live; deeper structural problem than body diode.

**Stage 2 (≤90 min) — Mini BBO conditional on Stage-1 result**
- If structural confirmed: fix `body_pdiode_Rs` properly, then run 5×5 (Bf, K1_LUT_scale) on 3 representative cells (9 fits × ~10min ≈ 90 min).
- Pre-registered success: any combo <1.5 dec at V_G1=0.6.

If Stage 2 still fails to recover ≤1.5 dec, the structural flaw extends beyond the body diode (likely BJT polarity / iii→body sign / Vb node consumption).

---

## Files
- `research_plan/oracle_queries/O58_structural/prompt.md`
- `research_plan/oracle_queries/O58_structural/openai_response.md`
- `research_plan/oracle_queries/O58_structural/gemini_response.md`
- `research_plan/oracle_queries/O58_structural/grok_response.md`
