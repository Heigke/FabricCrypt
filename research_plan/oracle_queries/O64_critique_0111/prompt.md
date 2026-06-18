# O64 — Autonomous critique cycle (6h)

## Context

Last 6h of 01_LOG.md attached as context.md. NS-RAM 2T pyport vs Sebas 33 IV.
Arc R-13..R-29: BJT topology fix (R-20 applied) → BBO plateau 3.43 dec →
cfg-diff (no toggle helps) → ngspice handover (4.12 dec) → component decomp
(Ids_M1 3 dec under ngspice) → card patch (lalpha0 length cancellation found,
+3 dec ngspice gain) → pyport refit only -0.17 dec gain → patch flow verified
(Iii ×10 confirmed) → R-29 BSIM4 channel audit dispatched.

## Three CRITICAL questions

**Q1 — Fragility / overclaim**: Where is this arc fragile or overclaimed?
Specifically scrutinize:
- "Pyport architecture VINDICATED" (R-25) — based on pyport==ngspice ≤0.24 dec
  match. Is this a meaningful comparison if BOTH are wrong vs silicon by 5 dec?
- "REWRITE REQUIRED" (R-24) → "Sebas card bug" (R-25) reversal in 30 min. Are we
  drifting toward easier-to-fix narratives?
- The chain Iii×10 → +0.17 dec (z343) while ngspice patched → +3 dec — is the
  R-26 "lalpha0 is root cause" claim solid?

**Q2 — Single best falsifier**: One concrete experiment to falsify the strongest
current claim that "fixing the Ids_M1 3-dec gap will recover 3-5 dec on cell-wide
fit". Concrete, runnable in <1h.

**Q3 — NO-CHEAT drift**: Cite specific 01_LOG.md lines/runs where:
- Gate was logged post-hoc (not pre-registered)
- "Honest FAIL" was glossed as "expected"
- Bias subset cherry-picked (z338 BBO used 9-bias subset, full 33 never validated)

≤450 words per oracle. Be aggressively skeptical.
