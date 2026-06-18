# NS-RAM Autonomous Campaign Plan — 2026-05-16

Path B from CAMPAIGN_SYNTHESIS_2026-05-16.md: accept ~1.2 dec functional model, drive to defensible publication in ~2 weeks.

## Phases (autonomously executed)

### P1 — Honest fwd+bwd re-run of all major pipelines [URGENT, 4-6h]
Re-run z430/z432/z443/z446/z449/z454 with BOTH sweep directions, write corrected DC table. Closes 6/9 cherry-picks.
- P1a (ikaros): z430, z432, z443 (older pipelines, GPU)
- P1b (zgx): z446, z449, z454 (newer, after code sync)
- Output: `research_plan/HONEST_BASELINE_2026-05-16.md`

### P2 — BSIM3/4 silent type-mismatch fix [2 days, ikaros]
Audit nsram/bsim4_port/ for places where BSIM3-card values were silently forced into BSIM4 paths or vice versa. From synthesis: this is a known open item.

### P4 — rbodymod=1 (Rb body resistance) implementation [2 days, ikaros]
Open since 2026-05-13 in MASTER_FIX_PLAN. Likely cause of fwd=1.31 vs bwd=2.86 asymmetry. Wire BSIM4 §6.7 Rbody equations.

### P5 — Holdout split methodology [6h, ikaros]
Reserve 7 of 33 curves as test set (random seed=0). Refit on 26 train. Report test RMSE separately.

### P6 — Brief v4.5 with honest numbers [3 days writing]
Rewrite Mario brief, kill all cherry-picked headline numbers, include holdout RMSE.

### O-loop — Oracle consultation every 6h
3-way oracle on current honest data. Auto-flag drift/disagreement.

## Autonomous execution

- Each phase ends with summary.json + gate verdict
- Cron monitors completion, dispatches next phase automatically
- Oracle critique fires every 6h on latest delta
- No user input required until P6 complete

## Gates (pre-registered)

- P1 PASS = all 6 pipelines re-run with both directions, table written
- P2 PASS = type-mismatch test added to test suite, all pass
- P4 PASS = rbodymod=1 lands without breaking forward fit > 0.1 dec
- P5 PASS = test-set RMSE within 0.3 dec of train (no overfit)
- P6 PASS = brief v4.5 reviewed by 3-oracle panel, ≤1/3 fragility flags

## KILL_SHOT (campaign abort triggers)
- P1 reveals avg > 2.0 dec everywhere → no functional model claim possible, pivot to Verilog-A
- P4 worsens fwd fit > 0.5 dec → rbodymod implementation wrong, escalate
- 3/3 oracles flag publishability as fundamentally compromised
