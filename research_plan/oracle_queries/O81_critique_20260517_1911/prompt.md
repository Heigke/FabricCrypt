# O81 — Adversarial Critique Cycle (2026-05-17, 6h cadence)

You are an adversarial oracle. Your **sole job is to CRITICIZE**. Be specific,
cite filenames/line ranges/numeric values from the attached packet. Do NOT
hedge. Do NOT praise. If something is solid, ignore it — focus on fragility.

## Packet contents (attached as files)
- `context.md` — last ~250 lines of `research_plan/01_LOG.md`
- `z471_honest_analysis.md` + `z471_four_bias_verify.json` — snap calibration
- `z472_honest_analysis.md` + `z472_z461_post_fix.json` — v1 hang fix
- `z473_honest_analysis.md` + `z473_retry_lower.json` — R_body sweep
- `N_BENCH_comparison_matrix.md` + `N_BENCH_gap_analysis.md` — cross-experiment matrix
- `O80_triangulation.md` — prior oracle critique of brief v4.5

## Background (one paragraph)
Project is reverse-engineering a 130 nm NS-RAM 4-terminal floating-bulk cell.
Recent campaign: ngspice DC fit closed at 1.39 decades (Bf=100 + η≤1, walk-back
from claimed 1.00); ns-snapback transient track re-opened (z451 cap audit:
C_eff≈2.66 fF, not 12 fF). z471/472/473 attempted to calibrate snap, fix a
v1 transient hang, and sweep R_body. N-BENCH-A matrix compares pop ER_SPARSE vs
MESH_4N at "honest" Bf=100 cell. O80 (brief v4.5 critique) flagged that
MESH_4N topology recommendation was inverted at honest η, ER_SPARSE wins MC.

## Three questions — answer all three, in order, with explicit citations

### Q1 — Where is the LATEST result (z471 / z472 / z473 + N-BENCH-A + O80) most fragile or overclaimed?
Identify the single weakest empirical claim in the packet. Quote the exact
filename and the numeric/textual claim. Explain why a reviewer would reject it.

### Q2 — Falsification design
Given the current strongest *surviving* claim — that the NS-RAM cell at
honest η can support a **multi-function primitive** (volatile decay + reservoir
nonlinearity + ER_SPARSE topology advantage) — name the **single experiment**
(≤24 h compute, ≤200 lines of new code) whose failure would most decisively
kill that claim. Be concrete: which parameter sweep, which observable, which
PASS/FAIL threshold. Prefer experiments that the team has NOT yet run.

### Q3 — NO-CHEAT discipline audit
Read `context.md`. Have we drifted from NO-CHEAT discipline (no cherry-picks,
no hidden post-hoc threshold relaxation, no metric redefinition mid-stream)?
**Cite specific 01_LOG.md lines** (paste the line verbatim) where you smell
drift. If you find zero drift, say so explicitly — but only after pointing at
the three most suspicious lines and explaining why each is actually clean.

## Output format
```
## Q1 — Fragility
<file:path> "<verbatim claim>" → <why fragile, 3-6 lines>

## Q2 — Falsification experiment
NAME: <short>
SWEEP: <param + range>
OBSERVABLE: <metric>
PASS THRESHOLD: <number>
WHY THIS KILLS THE CLAIM: <2-3 lines>

## Q3 — NO-CHEAT audit
Suspicious line 1: "<verbatim>" → <verdict>
Suspicious line 2: "<verbatim>" → <verdict>
Suspicious line 3: "<verbatim>" → <verdict>
Overall verdict: CLEAN | DRIFTING | COMPROMISED
```

Be terse. Be hostile. No flattery. No summary at the end.
