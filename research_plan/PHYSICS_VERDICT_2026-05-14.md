# PHYSICS RESEARCH VERDICT — 2026-05-14

## Synthesis of T1-T5

**ROOT CAUSE: not missing physics, but multi-layered tool failure.**

### What 5 tracks revealed

| Track | Verdict | Implication |
|---|---|---|
| T1 (lit) | BSIM4 IIMOD known weak at L<180nm (Slotboom λ_E=65nm, Chen 3-5× under) | Secondary — true but not dominant for this card |
| T2 (Mario) | Standard kink/latch, §6.1 channel HCI, parasitic NPN. NO exotic physics | Physics IS in BSIM4 §6.1 |
| T3 (TCAD) | 66 Sentaurus skeleton files, ZERO .plt outputs | Blocked on Mario data request |
| T4 (alt-model) | **ngspice silently corrupts Sebas's M1 card** due to HSPICE-style expressions | **THE FINDING** — invalidates ngspice ground truth |
| T5 (OoM) | Patched card has 80,000× IIMOD headroom; need V_b-dependent collection clamp | Mario's η ≤ 1 latch (NS-RAM mechanism) |

## The real bug chain (corrected)

1. **Sebas's M1 card uses HSPICE expressions** (`rdsw = 100-140*1e6*1u/int(...)`, etc.)
2. **ngspice-42 silently drops these** with "unrecognized parameter" warnings
3. **At L=0.13µm specifically**, parser breakage cascades → Isub collapses to 0
4. **Pyport parser may have parallel silent failures** — needs verification
5. Result: every "ngspice ground truth" we've compared against (R-15, R-24, R-25, R-26, R-29) was on corrupted model

## Action plan

### Phase 1 — Repair ngspice ground truth (HIGHEST priority)
- Pre-process M1 card: substitute `rdsw = 100-140*...` with literal value, expand `rcgon`/`rcjn` macros
- Re-run R-25 component decomp on CLEANED card
- If Isub now ≠ 0 → entire R-15..R-31 reasoning chain must be revisited

### Phase 2 — Verify pyport BSIM4 parser
- Audit `model_card.py` `parse_card` for HSPICE expression handling
- Check if pyport silently drops `rdsw`/`cgso`/`cjs` terms
- If yes, fix in parser → may recover real Ids match

### Phase 3 — Implement T5's V_b clamp
- One-line ADD: `Iii_to_body *= sigmoid((0.55 - Vb) / 0.030)`
- Per Mario's η ≤ 1 walk-back from M3b
- Re-run 33-curve refit, expect significant improvement

### Phase 4 — Honest revisit of "z304 spurious" verdict
- If cleaned card produces different Ids at z304's tuned params
- The "0.99 dec was BJT-compensation" conclusion may have been on broken baseline
- Need full re-evaluation

## Gates (pre-registered for Phase 1-3 retest)

- INFRA: cleaned ngspice deck gives non-zero Isub at L=0.13µm
- PASS: cell-wide median < 0.95 dec after Phase 1+3 (cleaned card + V_b clamp)
- AMBITIOUS: cell-wide median < 0.50 dec
- NO-CHEAT: re-run on FULL 33 curves, not subset

## Risks

- T4's diagnosis only verified at flagship bias; need to confirm L-dependence at other points
- Pyport may have already implicitly handled HSPICE syntax (need parser audit)
- T5's V_b clamp interacts with iii_gain sigmoid we removed in R-23
