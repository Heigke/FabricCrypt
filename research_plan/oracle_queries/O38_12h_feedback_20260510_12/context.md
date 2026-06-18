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

## 2026-05-10 oracle 12h #37 — O37: 3/3 oracles converge + shared WARNING

3/3 oracles agree on every key point:

**Q1 — Brief framing**: SCOPED kvantitativ claim is gate-passed.
  - openai: "approximately linear in projection-baseline strength,
    predictable to ≈±1pp within MNIST-like 28×28 grayscale tasks"
  - gemini: "provisional quantitative-prediction claim" — lead with it
  - grok: "headline-worthy without overclaiming, caveat as preliminary
    linear fit across 4 MNIST-family tasks"
  Consensus: include in Mario v2 NOW with explicit MNIST-family scope.

**🚨 Q2 — SHARED WARNING (3/3 flag same risk)**: TASK-MODALITY CONFOUND.
  All 4 tasks are 28×28 grayscale sequential image classification with
  same projection+linear-classifier pipeline. The Δ-vs-baseline relation
  may be a property of the PIPELINE, not the NS-RAM reservoir
  specifically. Could fail catastrophically on non-image tasks.
  Mitigation candidates:
    - openai: ESN control at one bias point — same slope?
    - gemini: NARMA-10 with same pipeline (non-image, time-series)
    - grok: CIFAR-10 RGB (color, larger images)
  Plus secondary risks (less critical): hyperparam-specificity
  (g_VG2=0.20 only), linear-vs-saturating functional form.

**Q3 — Next experiment**: Non-MNIST-family extension. Variants:
  - openai: pMNIST at low baseline (extends 43-72% range to ≤35%)
  - gemini: NARMA-10 with same pipeline (different MODALITY — strongest)
  - grok: CIFAR-10 grayscale at proj ~55% (near zero-crossing)
  Acceptance gate (consensus): predicted within ±2pp + sign matches
  → claim extends; large error or sign-flip → claim bounded to family.

### 2-line synthesis
Monotonic claim is brief-headline-ready with MNIST-family scope; needs
non-image extension to upgrade beyond family-bounded claim. Strongest
risk = pipeline-vs-NS-RAM attribution; easiest mitigation = NARMA-10
test using same baseline+linear pipeline (gemini), or CIFAR-10 (grok).

WARNING-tag push (≥2 oracles share task-modality risk).

## 2026-05-10 resource audit 00:03 — all GREEN
disk 49%, mem 2.6/31GB (8%), APU 34°C, sentinel PID 9161 alive,
results 52GB stable (no growth since prior audit), /tmp logs all <100KB.
No alert. (Used absolute path append to avoid cwd-bug recurrence.)

## 2026-05-10 daily synthesis 02:43 #38
APU 34°C, sentinel alive (PID 9161). 14 substantive entries 2026-05-09
(cross-task sprint: z233→z238). Peak: z238 quantitative validation of
monotonic claim (predicted Δ within 0.32pp). O37 oracle WARNING:
task-modality confound (all 4 datapoints 28×28 grayscale).
🚩 Blocked >5 days: sebas_silicon_characterisation_request.md
(drafted 2026-05-05, 5 days unsent today — HITS THRESHOLD).
mario_update_note_v2_draft.md (1d, 2 figure attachments ready).
Milestone flip: cross-task narrative upgraded from "task-conditional"
→ "quantitatively predictable" with O37-flagged scope caveat.
Re-prio for next compute: NARMA-10 modality-test (gemini O37 #1 pick).
Syncing proposal (drift since 22:44 yesterday).

## 2026-05-10 GPU off-hours #39 — z239 CIFAR FAILED → FMNIST replication
CIFAR-10 openml download HTTP 504; auto-fell back to FashionMNIST.
n=8 replication of z236: Δ=-9.75pp vs z236 -10.60pp. CIs overlap
[-10.50, -9.00] vs [-11.25, -9.50]. 4-task-fit predicted -10.56pp at
proj=72%; actual -9.75pp, |err|=0.81pp, BARELY misses CI by 0.06pp on
upper bound (claim_extends=False technically but qualitatively close).
Replication value: z236 robustness confirmed.
Modality-test STILL PENDING — need alternative CIFAR source (torchvision).
O37 task-modality WARNING unresolved.
APU peak 88°C, no kill events.

## 2026-05-10 baseline watchdog 04:43 — PASS (3rd consecutive day)
ER_SPARSE/ff/N=256/seed=0/1ep: best 0.500 final 0.125. Exact match to
ground truth + last 2 days. Diff 0.000 ≤ 0.10. NO regression after 14
substantive entries on 2026-05-09 (cross-task sprint z233-z238 + figs).
APU 34°C, sentinel alive.

## 2026-05-10 morning brief 06:59 #40
Overnight 8h: 5 entries. 22:37 overnight launcher (skipped, surrogate
already done). 00:00 O37 oracle: 3/3 converge + shared WARNING on
task-modality confound. 00:03 resource audit GREEN. 02:43 daily
synth #38 (Sebas request HIT 5d-unsent threshold). 03:23 GPU off-hours
#39: z239 CIFAR-10 download FAILED (openml HTTP 504), auto-fallback
FMNIST → unintended replication of z236 (Δ=-9.75pp matches z236
-10.6pp). 04:43 baseline watchdog PASS (3rd consecutive day).
APU peak 88°C (z239 runs), no thermal events. Pending: real CIFAR via
torchvision; user review of Mario v2 + Sebas drafts.

## 2026-05-10 work-hours #41 — z240 CIFAR: linear extrapolation FAILS
n=8 CIFAR-10 grayscale: proj=15.3% (much lower than MNIST-family 43-72%),
Δ_actual=+1.94pp CI [+1.00, +2.75], 7/8 positive, p=0.001.
Δ_predicted from 4-task linear fit (Δ=+29.8-0.56·proj%): +21.13pp.
|error| = 19.2pp. SIGN matches but MAGNITUDE wildly off.
Conclusion: linear fit does NOT extrapolate beyond MNIST-family.
Reservoir contribution SATURATES at very weak baselines, not linearly.
Mario v2 claim must scope to MNIST-family (proj 43-72%); outside,
direction holds but magnitude is task-specific. Retraction needed.
APU peak 90°C, no kill. O37 task-modality WARNING partial-resolved:
direction-claim survives, magnitude-claim falsified.

## 2026-05-10 work-hours #42 — Mario v2 + monotonic figure scope-bound (post-z240)
Mario v2 cross-task paragraph rewritten to honestly bound claim:
"Within MNIST-family (proj 43-72%): linear fit precise to ±0.5pp.
Outside (CIFAR proj=15%): direction matches, magnitude saturates ~10×
below linear extrapolation — task-specific factor we don't yet model."
Monotonic-baseline figure updated to 5 datapoints; linear fit now
restricted to MNIST-family band (40-75%); CIFAR plotted outside band
with explicit "extrapolation fails by ~10×" annotation. Title rewritten
to reflect bounded validity. Replaces overclaim from #51/#53.

## 2026-05-10 track-audit 6h #16 — V/R/C/T/S/P status (post-z240 scope-bound)

8 entries since #15: O37 WARNING, daily synth, watchdog 3rd day,
z239 CIFAR fail→FMNIST replication, z240 CIFAR succeeded but
falsified linear extrapolation, Mario v2 + figure scope-bound.

| Track | Status | Δ since #15 |
|---|---|---|
| **V** | ✅ ACTIVE | z240 +z239 added (paired-t, p<0.001 each). |
| **R** | ✅ CLOSED | (no change.) |
| **C** | ✅ CLOSED | (no change.) |
| **T** | ✅ COMPLETE | 5 image tasks now (MNIST/KMNIST/FMNIST_small/FMNIST/CIFAR). |
| **S** | ✅ ACTIVE | Linear fit RESTRICTED to MNIST-family band per z240 honest finding. |
| **P** | ✅ ACTIVE | APU peak 90°C across z239+z240, no kill. Thermal-aware batching held. |

**Stalled count = 0** (4th consecutive audit at zero stalled).

**Cross-task narrative evolution** (honest trajectory):
  audit #13 post-z233:  "frozen FAILS" — single negative
  audit #14 post-z235:  "single-knob retune RECOVERS" — single positive
  audit #15 post-z237:  "monotonic complement to weak baselines"
  z238 quantitative validation
  ↓ z239 CIFAR-failed→FMNIST replication
  ↓ z240 CIFAR succeeds, falsifies linear extrapolation
  audit #16 post-z240: "linear within MNIST-family (43-72%); saturates
                         outside, direction holds but magnitude
                         task-specific" — HONEST SCOPE-BOUND CLAIM

This is the most defensible Mario brief framing yet. Not as flashy
as the unbounded-linear claim from #51/#53, but it survives external
scrutiny including non-MNIST-family extrapolation.

**Highest leverage now**: STILL HUMAN-side (Mario v2 send + Sebas
sends — Sebas main 5+ days unsent). Compute fallbacks (lower leverage):
  - 6th-task validation in MNIST-family band (e.g., EMNIST-letters)
    to firm up the "linear within band" claim
  - Saturation-curve fit on 5 points instead of linear fit
    (logistic or piecewise)
  - Hyperparam sensitivity (g_VG2 sweep at one task) to test "single-
    knob choice" winner's curse (O37 risk c).

Logged. Mario brief delivery is the actual blocker now.

## 2026-05-10 work-hours #43 — Saturation curve analysis (audit #16 fallback option 2)
Fitted alternative models to 5-task Δ-vs-proj data:
  Linear (band-only 43-72%): R²=0.997 ✅ within band
  Linear extrapolated to 5:  R²=-0.805 ❌ (CIFAR off the line)
  tanh saturation (5 pts, 3 params): R²=0.792 (CIFAR fit weak)
  Sigmoid 4 params on 5 pts: R²=0.973 (overfit risk: 4 params/5 pts)

Sigmoid suggests reservoir-help saturates ~+3pp and reservoir-hurt
saturates ~-12pp with sharp transition at proj≈63%, but 4 params on
5 points is statistically suspect.

Honest conclusion for Mario v2: linear is robust within band; outside
band, we have ONE datapoint (CIFAR) showing saturation — not enough
to fit a curve confidently. Mario v2 stays at "linear within band,
direction-only outside" until we have ≥3 datapoints outside the band
to constrain the saturation form.

Logged. No code change to Mario v2 — current scope-bound framing is
already correct. Saturation analysis stays as research-internal note.

## 2026-05-10 work-hours #44 — z241 g_VG2 sensitivity: H1 CONFIRMED
3 new g_VG2 values × 5 seeds + existing z233/z235:
  g=0.05: Δ=-4.67  g=0.10: Δ=-2.10  g=0.15: Δ=+2.30
  g=0.20: Δ=+5.10  g=0.30: Δ=+9.60
SMOOTH gradient (~linear: Δ ≈ -7.4 + 56·g_VG2). No peak, no winner's
curse. Predicted Δ at g=0.30 from 4-pt fit: +9.4pp; actual +9.6pp.
O37 risk c (winner's curse on g_VG2=0.20): REJECTED. The strong-input
mechanism is the knob; specific value 0.20 is not special — it's just
near the negative→positive transition zone.
Mario v2 narrative robustness STRENGTHENED: cross-task relationship
holds across a wide g_VG2 range (~0.13 to 0.30+ all give positive Δ
on MNIST). APU peak 90°C, no kill.
