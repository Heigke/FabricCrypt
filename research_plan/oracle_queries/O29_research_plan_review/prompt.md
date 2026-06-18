# O29 — Critique the post-NRF research plan

## Context

Brief sent to NRF 2026-05-06 (oracle-vetted, 0.654-dec headline,
ER_SPARSE tape-out recommendation). The attached `plan.md` is a
draft of how to spend the next ~4-12 weeks of post-deadline work,
covering 8 phases (A-H) and a cron strategy for autonomous
execution.

You (gpt-5 + gemini) wrote O28 saying:
- 0.654-dec brief is defensible
- low-Id root is physical, alt-root is parasitic-NPN latch
- Plan A path: ngspice cross-check + branch-protected Newton +
  body-leak regularizer
- Don't pivot to two-NPN now

## Questions (be terse, < 600 words total)

1. **Priority order**: Is "Week 1: B.1 → F7 → A.12 → G.2" the right
   sequence, or should something else go first? Specifically: does
   B.1 ngspice cross-check on 3 biases really gate everything else,
   or can B.2/B.3 (Newton hardening) start in parallel?

2. **Decision tree at B.5**: I set thresholds at gain=0.02 and 0.05
   dec. Are these the right cutoffs given the 0.654-dec baseline,
   or too lax / too strict?

3. **Missing items**: What's in the plan that shouldn't be? What's
   NOT in the plan that should be? Particularly worried about
   blind spots in:
   - Hard-benchmark suite (F.4) — is XOR/NARMA/MC the right set?
   - M9 fan-out structure — currently in brief as future, not in plan
   - Pavlovian conditioning result — should it appear in dissemination?

4. **Risk assessment**: What's the single biggest risk to executing
   this plan over 4-12 weeks? E.g., Sebas data delay, multi-root
   pathology not actually fixable, paper-deadline mismatch?

5. **Cron strategy critique**: I proposed
   - 30-min cron during 08:00-22:00 work hours
   - Daily 02:00 deep-work synthesis
   - Weekly Mon 09:00 plan review
   Is this overkill / underkill / wrong cadence? Anything I'm
   missing about how to keep an autonomous research loop productive
   over weeks?

Goal: a final plan you (oracles) can stamp "send-it" on, with a
ranked todo list I can act from.
