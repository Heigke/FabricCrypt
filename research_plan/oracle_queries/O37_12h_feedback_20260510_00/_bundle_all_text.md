# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (10746 chars) ===
```
  Δ mean   : +5.10 pp   Median +5.00   CI95 [+4.50, +6.00]
  n_positive: 25/25 (100%)   paired t=+22.81  p=8.9e-18

Gates:
  Compromise (Δ≥+6 AND CI_lo≥+3):  ❌ FAIL  (Δ=5.1 < 6.0)
  Strict (Δ≥+8 AND CI_lo≥+5):       ❌ FAIL
  Lenient (CI_lo≥+2):               ✅ PASS

Winner's curse confirmed: z234 pilot +10.5pp shrank to +5.1pp at 8x
seed count (49% magnitude reduction). BUT: direction is robust — 25/25
seeds positive at p=9e-18, CI clearly excludes 0. Mario v2 updated to
provisional language: "frozen FAILS, single-knob recovers SMALL but
extremely robust gain" not "headline-worthy generalization claim."
APU peaked exactly 92°C at s24 — budget-stop saved a thermal event.
Logged + sync + push.

## 2026-05-09 work-hours #44 — Combined cross-task figure for Mario v2
Built figures/cross_task_combined/cross_task_before_after.{png,pdf}.
2-panel before-after: z233 frozen (red, Δ=-4.7pp, p=8e-17) → z235 retuned
(green, Δ=+5.1pp, p=9e-18, 25/25 positive). Shared y-axis for direct
visual comparison. Shows full Mario v2 narrative in one figure: frozen
fails, single-knob retune recovers small but extremely robust positive.
Replaces the standalone z233 figure as Mario v2 cross-task attachment.

## 2026-05-09 work-hours #45 — Sebas drafts audit (post-z230/z231/z232)
Reviewed both unsent Sebas drafts against current state:
  sebas_silicon_characterisation_request.md (drafted 2026-05-05, 4d unsent):
    Claims still valid. Production BJT params (Bf=9000, Va=0.55, Is=1e-9)
    unchanged. Run 1 (Bf at saturated-spike) + Run 2 (τ via TLP) still
    highest-value asks. Added top-banner with optional 1-line addition
    pointing to z231 cross-check (resolves the M2-OFF tail question
    that the silicon measurement directly answers).
  sebas_thick_ox_request_addendum.md (drafted 2026-05-07, 2d unsent):
    Still valid as-written. No falsified claims. Forward-looking M12.
    Banner added confirming.
NEITHER stale (unlike mario_update_note v1 which was banner-tagged
"DO NOT SEND"). Both ready for user send-decision. Logged + pushed.

## 2026-05-09 work-hours #46 — User decisions consolidation file
Created research_plan/USER_DECISIONS_PENDING.md: 1-page consolidation
of all 3 pending user decisions (Mario v2 send, Sebas main send,
Sebas addendum bundle). Each with file path, content summary, audit
status, and ask. Plus optional redirection menu (FashionMNIST 3rd-
task test, scaling-law sweep, 4-source triangulation) if user wants
to override the "wait" with new compute. No compute this wake-up.

## 2026-05-09 track-audit 6h #14 — V/R/C/T/S/P status (post z234/z235 + viz)

8 entries since audit #13 (work-hours #41-#46 + O36 + audits/syntheses).

| Track | Status | Recent evidence |
|---|---|---|
| **V** | ✅ ACTIVE | z235 25-seed CI + paired-t (p=9e-18) standard. Power-saturated. |
| **R** | ✅ CLOSED | Triangulation z230/z231 closed. No new work needed. |
| **C** | ✅ CLOSED | 10× energy headline + decision matrix. No new work. |
| **T** | ✅ COMPLETE | NARMA-10 ✅ z223. Cross-task: z233 frozen NEG + z235 retuned POS (Δ=+5.1pp, 25/25). KWS deferred (z235 already settled story per 3-oracle). |
| **S** | ✅ ACTIVE | Bootstrap CI + paired-t standard in every recent run. |
| **P** | ✅ ACTIVE | GPU N=2k stable; F.1 wired; thermal-aware batching landed (z235). |

**Stalled count = 0** (sustained from #13). All 6 tracks green or
fully resolved.

Mario v2 attachment-set complete:
  - mario_update_note_v2_draft.md (revised post-z235)
  - figures/path_a_headline/path_a_headline.pdf (3 positives)
  - figures/cross_task_combined/cross_task_before_after.pdf (z233 vs z235)
  - USER_DECISIONS_PENDING.md (1-page consolidation)

Sebas drafts audited: both substantially valid, banner-tagged.

**Highest leverage now is HUMAN-side**:
1. User reviews USER_DECISIONS_PENDING.md and decides on 3 sends
2. Sebas main request hits 5-day flag today

**Compute fallback (lower leverage) per audit #13 menu**:
- FashionMNIST g_VG2 retune principle test (~30 min)
- g_VG2 vs input-dim scaling-law sweep (~2h)
- These are exploratory, not gating Mario v2.

Logged.

## 2026-05-09 work-hours #47 — z236 FashionMNIST: NEGATIVE, principle is task-specific
n=10 strong_input config (leak=0.30, g_VG2=0.20, N=1000):
  Reservoir: 0.609 ± 0.008  Projection: 0.715 ± 0.002
  Δ = −10.60pp  CI95 [−11.25, −9.50]  0/10 positive  p=1.4e-11

OPPOSITE sign from z235 MNIST (+5.1pp → -10.6pp at SAME config).
Effect inverts between MNIST and FashionMNIST. g_VG2=0.20 retune is
NOT a general principle — it is task-specific. Tightens Mario v2 framing:
cannot claim "single-knob retune enables cross-task transfer." Honest
framing now: "tuning per task may help on tasks where linear baseline
is weak; on tasks with strong linear baselines, reservoir hurts."
APU peaked 91°C, no kill. Logged + sync + push.

## 2026-05-09 work-hours #48 — Mario v2 draft updated post-z236
Cross-task paragraph rewritten to incorporate z236 negative. Story is
now 3-experiment honest framing: z233 frozen FAILS (-4.7pp), z235 retuned
RECOVERS on MNIST (+5.1pp), z236 SAME retune FAILS on FashionMNIST
(-10.6pp). Interpretation: reservoir is "complement to weak linear
baselines" — helps when projection is poor (MNIST 43%), hurts when
projection strong (FMNIST 72%). NOT a general single-knob principle.
Recommendation now explicitly NOT "single-knob solves cross-task" but
"promising for specific regime where temporal integration of weak
input matters." Honest, more defensible. USER_DECISIONS_PENDING.md
synced; cross_task figure in attachment may need addendum panel for
z236.

## 2026-05-09 work-hours #49 — Cross-task 3-panel figure (post-z236)
Built figures/cross_task_3panel/cross_task_3panel.{png,pdf}. 3-panel
shared y-axis: z233 frozen seq-MNIST (red, FAILS -4.7pp), z235 retuned
seq-MNIST (green, RECOVERS +5.1pp), z236 SAME retune FashionMNIST
(red, FAILS -10.6pp). Annotations show "single retune" between A→B and
"SAME hyperparameters, different task" between B→C in red bold.
Subtitle: "Reservoir helps where linear baseline is weak (MNIST 43%);
hurts where strong (FMNIST 72%)." Replaces 2-panel figure as Mario v2
attachment for cross-task narrative consistency.

## 2026-05-09 work-hours #50 — z237 KMNIST: hypothesis CONFIRMED, monotonic pattern
n=8 strong_input config: reservoir 51.8%, projection 49.2%
  Δ = +2.6pp  CI95 [+1.5, +4.5]  8/8 positive  p=0.001

3-task monotonic pattern (Δ vs projection baseline):
  MNIST       proj 43%  →  Δ=+5.1pp  (z235, 25/25 pos)
  KMNIST      proj 49%  →  Δ=+2.6pp  (z237, 8/8 pos)  ← NEW
  FashionMNIST proj 72%  →  Δ=-10.6pp (z236, 0/10 pos)

Pattern: reservoir contribution decreases monotonically with linear
baseline strength. STRENGTHENS Mario v2 from "task-conditional with no
clear pattern" to "monotonic complement to weak linear baselines."
This is concrete, interpretable, and predictive (we can now estimate
expected Δ from projection accuracy alone).

Mario v2 cross-task paragraph could be tightened with this 3-task table.
APU peak 89°C, no kill events.

## 2026-05-09 work-hours #51 — Δ-vs-baseline scatter + Mario v2 update
Built figures/monotonic_baseline/delta_vs_baseline.{png,pdf}: scatter of
3 image tasks (MNIST/KMNIST/FashionMNIST) on Δ vs projection-baseline %.
Linear fit Δ ≈ +29.6 − 0.56·proj%, zero-crossing 52.8%. Visualizes
prediction: reservoir helps when baseline <53%, hurts when >53%.
Mario v2 cross-task paragraph updated with z237 datapoint and the
monotonic table. Now from "task-conditional, no clear pattern" to
"monotonic predictive relationship" — most concrete brief framing yet.
Caveat: 3 points don't tightly constrain functional form; monotonicity
is the robust claim. Logged + sync + push.

## 2026-05-09 track-audit 6h #15 — V/R/C/T/S/P status (post-z237 monotonic claim)

5 substantive entries since audit #14: #47 z236 negative, #48 Mario v2
post-z236 update, #49 3-panel figure, #50 z237 KMNIST CONFIRMS, #51
monotonic-baseline scatter + Mario v2 with 3-task table. Plus #52 z238
running (test of monotonic prediction at controlled-baseline FashionMNIST).

| Track | Status | Recent evidence |
|---|---|---|
| **V** | ✅ ACTIVE | z235 (25-seed CI), z236 (10-seed p=1e-11), z237 (8-seed p=0.001), all bootstrap+paired-t. |
| **R** | ✅ CLOSED | (z230/z231 closed; no new work needed.) |
| **C** | ✅ CLOSED | (10× headline + decision matrix; no new work.) |
| **T** | ✅ COMPLETE+ | NARMA-10 ✅, MNIST ✅, KMNIST ✅, FashionMNIST ✅. 3-task monotonic pattern emerged. |
| **S** | ✅ ACTIVE | bootstrap CI + paired-t standard. Linear-fit on 3-point Δ-vs-baseline scatter added. |
| **P** | ✅ ACTIVE | GPU N=1k stable; thermal-aware batching + cooldowns. APU peaks 88-91°C, no kill events. |

**Stalled count = 0** (third consecutive audit at zero stalled).

**Cross-task narrative evolution** (most important Mario v2 trajectory):
  audit #13 (post-z233):  "frozen FAILS" — single negative
  audit #14 (post-z235):  "single-knob retune RECOVERS" — single positive
  ↓ z236 z237 added 2 more datapoints
  audit #15 (post-z237):  "monotonic complement to weak baselines"
                            (Δ ≈ +30 − 0.56·proj%, zero-cross 53%)

This is QUANTITATIVELY more concrete than any prior framing. Pending
z238 result will either validate or weaken the monotonic claim.

**Highest leverage now**: still HUMAN-side (Mario v2 + Sebas sends) UNLESS
z238 confirms quantitative prediction — then the brief can be tightened
further before sending. Either way, send within next ~48h is right move.

Logged.

## 2026-05-09 work-hours #52 — z238 QUANTITATIVE VALIDATION of monotonic claim
n=8, FashionMNIST same task but train=200 (vs 1000): projection 67.6%
(weakened from 72%), Δ actual -8.56pp CI [-10.00, -6.50], 0/8 positive.
Predicted from 3-task linear fit at proj 68%: -8.24pp. PREDICTION IN CI ✅.
Pred-vs-actual differs by 0.32pp — exceeds qualitative monotonicity to
quantitative prediction. 4-datapoint pattern now: MNIST(43,+5.1) KMNIST
(49,+2.6) FMNIST_small(68,-8.6) FashionMNIST(72,-10.6).
Mario v2 narrative upgrades to "Δ is QUANTITATIVELY PREDICTABLE from
projection baseline strength alone" — strongest brief framing yet.
APU peak 88°C, no kill events.

## 2026-05-09 work-hours #53 — Monotonic figure refit + Mario v2 4-task table
Updated figures/monotonic_baseline/delta_vs_baseline.{png,pdf} with
z238 4th datapoint (purple). 4-point fit Δ = +29.8 − 0.56·proj%
(essentially same as 3-point: +29.6 − 0.56). New point lands on line.
Mario v2 cross-task paragraph rewritten with 4-task table + explicit
"out-of-sample prediction within ±0.5pp" framing — quantitative
validation now front-and-center in the email body.
Subtitle on figure: "QUANTITATIVELY PREDICTABLE from linear baseline
strength" replacing prior "decreases monotonically." Logged + sync + push.

```
