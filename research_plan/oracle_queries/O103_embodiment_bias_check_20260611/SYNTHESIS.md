# O103 synthesis — 4-oracle adversarial bias-check on H7 "embodiment" (2026-06-11)

Packet framed adversarially (told oracles to police our success-bias AND their own).
Verdicts on P("deserves the word *functional* embodiment"):
- GPT-5: **0.15** · Gemini-2.5-pro: **0.15** · Grok-4: **0.08** · Deepseek-reasoner: **<0.05**
- Consensus ≈ **0.10**. Unanimous, and converging on the SAME specific mechanisms (not vague).

## What all four independently said is WRONG (the refutation)
1. **`dep_zero ≈ 0.94` is the smoking gun.** Zeroing the substrate leaves coherence intact
   (PPL 19.96 vs base 19.85). So the body carries NO information the model uses for language.
   The "dependency" is a **kill-switch** keyed to "input untampered", not functional use.
2. **knock/shuffle 1000s× = tamper-detection → sabotage.** A cheap integrity check (DC band,
   low-order autocorrelation, cross-channel lag) + high-gain FiLM meltdown. Not embodiment.
3. **Cross-die break = normalization/operating-point artifact** (per-die median/MAD pushes a
   foreign signal off-manifold). v12 daedalus-coherent-on-both confirms it learned a generic
   "real-live-dynamics key", NOT a die fingerprint. "If z-scoring flips it, it's about z-scoring."
4. **Interoception (ch5 power-rate) is a tautology.** compute → power → counter is definitional.
   Predicting your own power draw ≠ "feeling"; it's fitting a known map. ("bathroom scale goes
   down when I exhale.")
5. **v14 = manufacturing fragility.** PPL 104 + dep_zero 0.19 at step 3400 = dependent but
   incoherent. Tightening the loop destroys the language task.
6. **The 6 "gaps" pass by construction** — every embodiment signal is encoded directly in a loss
   term (L_xdie, L_grad, L_self, L_fresh). The model complies with metrics named like a body-sense;
   it does not discover one.

## What SURVIVES as real (narrow, all four agree)
- A reproducible **temporal-dynamics detector**: a frozen 135M LM + FiLM stays base-coherent only
  on untampered LIVE windows and melts under temporal-shuffle/knockoff (shuffle breaks it ⇒ it
  keys on temporal structure, not just marginals). Stable across seeds.
- **Graded "hardware-gated style"**: r(ch4 dynamics → output entropy)=+0.91, collapses under
  shuffle. Genuine continuous behavioral steering by a live physical signal.
- **Self-effect sweep** = legitimate hardware characterization (which channels respond to compute).
- The **online 500 Hz hardware→generative-model harness** is a non-trivial systems scaffold.

## Decisive falsifiers we have NOT run (converging)
- **Transplanted authentic body** (Gemini/GPT-5): pipe a LIVE signal from a 3rd identical gfx1151
  die under unrelated load into the ikaros model. Stays coherent ⇒ it's an authenticity check, not
  self. Breaks ⇒ genuine self-model. Separates die-CLASS authenticity from causal SELF.
- **Yoked-swap / time-lagged closed loop** (GPT-5): per-step randomize feedback = self vs
  yoked-other (same time, other die) vs lagged-self vs AR-sham; pre-registered; shared norm.
- **Recording-only training** (Deepseek/Grok): train on a deterministic REPLAY of real signals.
  Same dependency gaps ⇒ live feedback is NOT essential ⇒ falsified.

## The bar for REAL functional embodiment (Deepseek, sharpest)
Need **zero-substrate PPL ≫ base WHILE real PPL < 1.5× base** — i.e. the model must NEED the body
to write well, not just detect tampering. Current result is the opposite (zero ≈ base). Until that
flips, "constitutively dependent on this silicon" is unsupported.

## Honest meta-note on the instrument
I PROMPTED them adversarially, so ~0.10 is a conditioned lower bound. But four independent models
converging on the SAME falsifiable mechanisms (dep_zero, normalization, tautological interoception)
— not vague doubt — is a strong, credible signal. The charitable "it's style/identity not perplexity"
reading is undercut because the style coupling is itself trained-in and the break is a tamper-detector.
Take it seriously: the v13/v14 objective rewards "large gap", which a kill-switch satisfies. That is
the wrong objective for embodiment.
