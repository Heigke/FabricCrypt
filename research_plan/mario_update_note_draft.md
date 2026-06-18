# Mario update note — body-state model enhancement (DRAFT) — ⚠️ STALE / SUPERSEDED

**🚨 STATUS 2026-05-09: DO NOT SEND.** Per O35 3-oracle consensus
this draft is stale — written 2026-05-07 BEFORE z218–z233 completed
the body-state work it announces as "in progress." It also contains
the now-falsified "lumped vs q2d branch divergence" framing that
z232 corrected. Use **mario_update_note_v2_draft.md** instead, which
reflects the actual current state (body-state surrogate complete,
NARMA-10 NRMSE 0.612 ± 0.030, R-track triangulated, cross-task
transfer negative at p = 8e-17).

Original draft preserved below for audit trail only.

---

**Status**: Draft for user review. Synthesised from O32 oracle consensus
(openai + gemini + grok all unanimous: tell Mario now, framing = solution-
forward, not alarmist).

**Send target**: Mario Lanza (KAUST). Cc: Sebastian Pazos.

**Suggested subject**:
> NS-RAM modeling update — body-state enhancement for long-memory tasks

---

## Email body

> Mario,
>
> Quick update — no fire, just keeping you informed as we push the
> brief's reservoir-computing line into harder edge-AI tasks.
>
> **What we found**: as we extended testing to long-memory benchmarks
> (NARMA-10, memory capacity), our current Python surrogate gives
> sub-2-step memory at N=200, where a working ESN-class network gives
> 100-200. Mackey-Glass results from the brief hold (it's a short-
> memory task and our model captures it). But edge-AI workloads like
> keyword spotting need the longer memory.
>
> **Why**: our surrogate is an instantaneous lookup of `(V_G1, V_G2,
> V_d) → I_d`. It omits the body capacitance — which on real silicon
> is where the cell's temporal computation actually lives (V_b
> integrates impact-ionization current, parasitic NPN turns on at
> threshold). In short, we modelled everything except the most
> physically interesting part of the cell.
>
> **What we're doing**: building a 4D transient surrogate
> `(V_G1, V_G2, V_d, V_b) → (I_d, I_ii)` with explicit V_b dynamics.
> Estimated 2-3 days. Initial parameters from 130 nm literature
> (C_b ~ 5 fF, τ_body in the 50 ns – 10 µs range), refined when
> Sebas's pulsed-V_d data lands.
>
> **What I need from you / Sebas**: nothing chip-design-side. The
> existing tape-out recommendation (ER\_SPARSE topology, calibrated
> at the lateral-parasitic point) is unchanged — it's calibrated
> against measured DC silicon, not against the surrogate's
> reservoir behavior. The body-state model only matters for what
> *applications* we can run on the cell, not for the cell parameters
> themselves.
>
> The two characterisation runs from the brief's M3b deliverable
> (I_c/I_b at saturation + pulsed-V_d / TLP for τ extraction) are
> now even more important — they let us pin C_b and τ for the
> transient surrogate. I'll send Sebas the request packet today
> if you and he are OK with it.
>
> **My ask**: if you have any chip-design decisions that depend
> on long-horizon memory benchmarks, please defer until we have
> the transient-surrogate update (~72 h). Otherwise no action
> needed; brief stands.
>
> Best,
> Eric

---

## What is intentionally NOT in this note

- No retraction of brief's results. The Mackey-Glass NRMSE = 0.747
  finding holds.
- No claim that current chip is wrong. The cell-design recommendation
  doesn't depend on body-state modelling — it's based on DC fits to
  silicon, which we have.
- No drama. Solution-forward framing per all 3 oracles.

## Pre-send checklist

- [ ] User reviews tone (matches Sebas/Mario relationship style)
- [ ] Confirm the Sebas request packet is sent same day or before
- [ ] Decide whether to attach the per-row residual diagnostic figure
- [ ] Optional: include the MC-vs-N plot from z217 as appendix
      (visual evidence of the gap)

---

*Drafted 2026-05-07 from O32 oracle consensus.*
